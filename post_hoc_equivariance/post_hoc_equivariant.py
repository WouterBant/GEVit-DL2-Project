import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


def get_transforms(images, n_rotations=4, flips=True):
    """ Returns all transformations of the input images """

    B, C, H, W = images.shape
    T = 2*n_rotations if flips else n_rotations  # number of transformations

    # initialize empty transforms tensor
    transforms = torch.empty(size=(B, T, C, H, W))
    transforms[:, 0,...] = images
    idx = 1

    # remember all orientations that need to be flipped
    orientations = [images] if flips else []

    # rotations
    for i in range(1, n_rotations):
        angle = i * (360 / n_rotations)
        rotated_images = TF.rotate(images, angle)  # B, C, H, W
        transforms[:, idx,...] = rotated_images
        idx += 1

        if flips:
            orientations.append(rotated_images)

    # flips
    for transform in orientations:
        flipped_image = TF.hflip(transform)
        transforms[:, idx, ...] = flipped_image
        idx += 1

    return transforms  # B, T, C, H, W


# TODO come up with fancy names for these variations these suck

class PostHocEquivariant(nn.Module):
    """ General class for the different types of equivariant models """

    def __init__(self, model, n_rotations=4, flips=True, finetune_mlp_head=False, finetune_model=False):
        super().__init__()
        self.model = model
        self.n_rotations = n_rotations
        self.flips = flips
        self.finetune_mlp_head = finetune_mlp_head
        self.finetune_model = finetune_model
        self.device = next(model.parameters()).device

    def forward(self, images):
        transforms = get_transforms(images, self.n_rotations, self.flips).to(self.device)
        B, T, C, H, W = transforms.shape
        transforms = transforms.view(B*T, C, H, W)  # process all transformations in one forward pass
        embeddings = self._forward(transforms, B, T)
        logits = self.project_embeddings(embeddings)  # B, num_classes
        return logits

    def _forward(self, transforms, B, T):
        embeddings = self.model.forward(transforms, output_cls=True)  # B*T, n_dim_repr
        embeddings = embeddings.view(B, T, -1)  # B, T, n_dim_repr
        return embeddings

    def project_embeddings(self, embeddings):
        """
        The different methods just differently project the latent dimensions

        Input:
            embeddings - latent dimensions (B, T, n_dim)
                B = number of elements in the batch
                T = number of different transformations of the original input image
                n_dim = the dimension of the latent dimension for a single input
        Output:
            logits (B, n_classes)
        """
        raise NotImplementedError()
    
    def train(self, b):
        super().train(b)
        if not b: return
        if not self.finetune_model:
            for param in self.model.parameters():
                param.requires_grad = False
        if self.finetune_mlp_head:
            for param in self.model.mlp_head.parameters():
                param.requires_grad = True


class PostHocEquivariantMean(PostHocEquivariant):
    """ Mean pooling over the different transformations of the image """

    def __init__(self, model, n_rotations=4, flips=True, finetune_mlp_head=False, finetune_model=False):
        super().__init__(model, n_rotations, flips, finetune_mlp_head, finetune_model)

    def project_embeddings(self, embeddings):
        combined_embeddings = embeddings.mean(dim=1)  # B, n_dim_repr
        outputs = self.model.mlp_head(combined_embeddings)  # just the pretrained one for now, works well
        return outputs


class PostHocEquivariantMax(PostHocEquivariant):
    """ Max pooling over the different transformations of the image """

    def __init__(self, model, n_rotations=4, flips=True, finetune_mlp_head=False, finetune_model=False):
        super().__init__(model, n_rotations, flips, finetune_mlp_head, finetune_model)

    def project_embeddings(self, embeddings):
        combined_embeddings = embeddings.max(dim=1).values  # B, n_dim_repr
        logits = self.model.mlp_head(combined_embeddings)  # just the pretrained one for now, surprisingly works
        return logits


class PostHocEquivariantSum(PostHocEquivariant):
    """ Sum over the different transformations of the image """

    def project_embeddings(self, embeddings):
        combined_embeddings = embeddings.sum(dim=1)  # B, n_dim_repr
        logits = self.model.mlp_head(combined_embeddings)  # just the pretrained one for now
        return logits


class PostHocEquivariantMostProbable(PostHocEquivariant):
    """ Predicts the class with the highest class product probability """

    def __init__(self, model, n_rotations=4, flips=True, finetune_mlp_head=False, finetune_model=False):
        super().__init__(model, n_rotations, flips, finetune_mlp_head, finetune_model)

    def _forward(self, transforms, B, T):  # overrides parent class method
        logits = self.model.forward(transforms, output_cls=False)  # B*T, n_classes
        logits = logits.view(B, T, -1)  # B, T, n_classes
        return logits  # here the logits are the embeddings we use

    def project_embeddings(self, embeddings):
        probs = torch.softmax(embeddings, dim=2)  # over the classes
        mle = torch.prod(probs, dim=1)  # over the different transformations
        mle = mle / mle.sum(dim=1, keepdim=True)  # normalize
        logits = torch.log(mle+1e-8) - torch.log(1-mle+1e-8)  # convert to logits
        return logits


class PostHocMostCertain(PostHocEquivariant):
    """ Takes the transformation that assigns highest probability to a class """

    def __init__(self, model, n_rotations=4, flips=True, finetune_mlp_head=False, finetune_model=False):
        super().__init__(model, n_rotations, flips, finetune_mlp_head, finetune_model)

    def _forward(self, transforms, B, T):  # overrides parent class method
        logits = self.model.forward(transforms, output_cls=False)  # B*T, n_classes
        logits = logits.view(B, T, -1)  # B, T, n_classes
        return logits  # here the logits are the embeddings we use

    def project_embeddings(self, embeddings):  # embeddings are logits here
        probs = torch.softmax(embeddings, dim=2)  # over the classes
        idx_highest_probs = torch.argmax(probs, dim=1, keepdim=True)  # among the transformations
        logits = torch.gather(embeddings, 1, idx_highest_probs)  # pluck the ones with the highest prob to some class
        return logits.squeeze(1)


class PostHocLearnedScoreAggregation(PostHocEquivariant):
    """ Given the latent dimension for a transformation, predict an importance weight """

    def __init__(self, model, scoring_model, n_rotations=4, flips=True, finetune_mlp_head=False, finetune_model=False):
        super().__init__(model, n_rotations, flips, finetune_mlp_head, finetune_model)
        self.scoring_model = scoring_model

    def project_embeddings(self, embeddings):
        B, T, n_dim = embeddings.shape
        embeddings = embeddings.view(B*T, n_dim)
        weights = self.scoring_model(embeddings)  # B*T, 1
        weights = weights.view(B, T, 1)
        normalized_weights = weights / weights.sum(dim=1, keepdim=True)  # weighs over orientations sum to one
        embeddings = embeddings.view(B, T, -1)
        weighted_embeddings = embeddings * normalized_weights  # weight all embeddings
        combined_embeddings = weighted_embeddings.sum(dim=1)
        logits = self.model.mlp_head(combined_embeddings)
        return logits


class PostHocLearnedAggregation(PostHocEquivariant):
    """ Uses a transformer to combine the latent dimensions into a new one """

    def __init__(self, model, aggregation_model, n_rotations=4, flips=True, finetune_mlp_head=False, finetune_model=False):
        super().__init__(model, n_rotations, flips, finetune_mlp_head, finetune_model)
        self.aggregation_model = aggregation_model

    def project_embeddings(self, embeddings):
        combined_embeddings = self.aggregation_model(embeddings)
        logits = self.model.mlp_head(combined_embeddings)
        return logits