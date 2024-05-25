## E(2) Equivariant Vision Transformer

[comment]: <Total blogpost should be a 20 minute read>

### Wouter Bant, Colin Bot, Jasper Eppink, Clio Feng, Floris Six Dijkstra

---
In this blogpost, we dive deeper into E(2) equivariant Vision Transformers, and we propose and evaluate alternative methods for the equivariant attention models discussed in ["E(2)-Equivariant Vision Transformer"](https://proceedings.mlr.press/v216/xu23b.html). This paper proposes a new Group-Equivariant Vision Transformer (GE-ViT), which is claimed to be a new positional encoding for the traditional well-known Vision Transformer (ViT) ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" ](https://arxiv.org/abs/2010.11929)

In particular, in this blogpost, we:
1. Evaluate existing and novel methods to make non-equivariant (attention) models (e.g., [Lippe 2023](https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/11-vision-transformer.html), [Dosovitskiy et al. 2021](https://arxiv.org/pdf/2010.11929)), equivariant by combining the predictions of different transformations of the input, that reflect inductive biases.
2. Propose and evaluate a novel method to make modern ViTs equivariant by using GE-ViTs onto group invariant patch embeddings to project patches to latent embeddings that will be used as input to the equivariant vision transformer model used by [Xu et al.2023](https://arxiv.org/abs/2306.06722).
3. Visualize latent representations in different layers of equivariant and non-equivariant models, to help researchers better understand these models. 
---

## The Importance of Equivariant Models
In this section, we explain the significance of equivariant models and review prior work. 

Equivariance is a fundamental property across various domains including image processing [Krizhevsky and Ilya 2012](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html), 3D point cloud analysis [Li, Chen, and Lee 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_SO-Net_Self-Organizing_Network_CVPR_2018_paper.html), chemistry [Faber et al.](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.117.135502), astronomy [Ntampaka 2016](https://iopscience.iop.org/article/10.3847/0004-637X/831/2/135/meta), and economics [Qin et al. 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/730d61b4d9ff794a028fa3a25b9b891d-Abstract-Conference.html). Equivariance means that when a transformation is applied to the input, the output undergoes a corresponding transformation. Simply put, if the input is shifted, rotated, or scaled, the output will shift, rotate, or scale similarly. This provides geometric guarantees and is parameters efficient as weights are shared across transformations.

Traditional Convolutional Neural Networks (CNNs) exhibit translation equivariance but lack equivariance to rotations in their input data. The translation equivariance of CNNs is reached by weight sharing and the use of a shared Convolutional Kernel. The Translation invariance is reached through pooling ["P.339 Translation equivariance"](https://www.deeplearningbook.org/). The first rotation equivariant neural network was proposed by [Cohen and Welling 2016](https://proceedings.mlr.press/v48/cohenc16.html). Their method augmented the existing translation equivariance of CNNs by incorporating discrete group transformations of the kernel. Their method employed lifting which is the process of increasing the input data into a higher dimensional space. This allows the CNN to capture the dynamics of an input image for multiple rotations with each rotation being an increase in this dimensional space. 

In the field of Natural Language Processing (NLP), [Vaswani et al. 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) introduced transformers, a model that quickly gained significant prominence in its field. Recognizing the potential of this architecture in computer vision, [dosovitskiy et al. 2020](https://arxiv.org/abs/2010.11929) proposed vision transformers (ViTs). However, a limitation of their approach is necessitating positional encodings for each patch of pixels, losing the translation or any other form of equivariance. Despite this drawback, vision transformers demonstrated noteworthy performance, achieving state-of-the-art results on various computer vision benchmarks.

Initial attempts have been made to modify self-attention to become group equivariant. Before the release of the GE-ViT model, the most promising work in the field was proposed by [Romero et al. 2020](https://proceedings.mlr.press/v119/romero20a.html). They proposed Group Equivariant Stand Alone Self-Attention Networks (GSA-nets), which incorporated a different positional encoding strategy and modifications to the group self attention mechanism to ensure equivariance.

<table align="center">
  <tr align="center">
      <td><img src="figures/Rotation.gif" width=600></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure 1.</b> The effect of rotation on the predicted digit for the GE-ViT and the standard ViT</td>
  </tr>
</table>

In Figure 1, the importance of equivariant models becomes evident. The non-equivariant model will change its predictions for every different transformation of the input, while the equivariant model consistently provides the same predictions. Guaranteed consistent outcomes are desirable in many fields such as cell analysis, where a model's ability to deliver consistent predictions regardless of image orientation is crucial.

--- 

## Recap on Vision Transformers (ViTs)

[comment]: <In this section we discuss modern ViTs and older equivariant versions.>

<!-- In recent years, the Transformer architecture ["Attention is all you need"](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) has had a huge impact in natural language processing (NLP). The success of this architecture has paved the way for an adaptation in computer vision, giving rise to Vision Transformers (ViTs). --> 

The original transformer architecture operates using an encoder-decoder structure. The encoder maps an input sequence $(x_1, ..., x_n)$ to a continuous latent variable $z=(z_1, ...,z_n)$. Using this latent variable $z$, the decoder generates an output sequence $y=(y_1, ..., y_m)$ one element at a time. At each generating time step, the model utilizes its previously generated output. The encoder and decoder both employ self-attention and fully connected layers. This process is depicted in Figure 2.

<table align="center">
  <tr align="center">
      <td><img src="figures/Transformer architecture.png" width=300></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure 2.</b> The Transformer architecture with the encoder (left) and decoder (right) (INSERT BRON)</td>
  </tr>
</table>

The encoder consists of a stack of $\mathcal{N}$ block layers. Each layer contains two sub-layers: a multi-head self-attention mechanism and a fully connected feed-forward network. Residual connections are used between each sub-layer.

The decoder is similar to the encoder but includes a third sub-layer, which handles multi-head attention on the output generated so far.

<strong> Attention: </strong>

The attention mechanism allows the transformer to assign different weights to individual input tokens, resulting in improved embeddings. Figure 3 provides a toy example where words with higher attention have higher opacity when encoding the word "it."

<table align="center">
  <tr align="center">
      <td><img src="figures/Attention example.png" width=300></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure 3.</b> An example of attention (INSERT BRON)</td>
  </tr>
</table>

Formally, the attention mechanism maps a query and a set of key-value pairs to an output. The query, key, and value are all vectors. The output is computed by taking the weighted sum of the values, with the weights determined by a scaled dot-product of the query and key vectors followed by a softmax operation. The final architecture uses multi-head attention, which consists of several attention layers computed in parallel. For efficiency, the attention for a set of queries is calculated simultaneously by packing all queries into a matrix Q, all keys into a matrix K, and all values into a matrix V. Figure 4 illustrates the attention mechanism.

<table align="center">
  <tr align="center">
      <td><img src="figures/Multi-head attention.png" width=500></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure 4.</b> Schematic depiction of the attention mechanism (INSERT BRON)</td>
  </tr>
</table>

<!-- This encoding is not used in ViTs  -->
<!-- <strong> Positional Encoding: </strong>

During the encoding block, the order of the input is not taken into account. To be able to know the order of the words or input, positional encodings are used. These positional encodings allow the Transformer to understand the order of the input sequence. These positional encodings have the same dimension $d_{\text{model}}$ as the input embeddings so that the two can be summed. In the original transformer network, the following positional encodings were used:

$$\begin{align} 
    PE_{(pos,2i)} &= sin(pos/10000^{2i/d_{\text{model}}}) \\
    PE_{(pos,2i+1)} &= cos(pos/10000^{2i/d_{\text{model}}})
\end{align}$$

In the above equations, $pos$ is the position and $i$ is the dimension. The exact details about the transformer architecture can be found in ["Attention is All you Need"](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html). -->

<strong> Vision Transformer: </strong>

The ViT closely follows the encoder of the original Transformer architecture ["An Image is Worth...](https://arxiv.org/abs/2010.11929). The standard Transformer receives a 1D sequence of token embeddings as input. For the ViT to handle 2D images, the image $x \in \mathbb{R}^{H \times W \times C}$ is reshaped into a sequence of flattened 2D patches $x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$. In this input $H$ is the height of the input image, $W$ is the width, and $C$ is the number of channels (3 for RGB images). Furthermore, $(P, P)$ is the spatial resolution of each image patch and $N=HW/P^2$ is the number of patches, forming the input sequence length to the transformer. Each patch is flattened and mapped to $D$ dimensions using a trainable linear projection, resulting in individual patch embeddings. Learnable positional encodings are added to these embeddings to inform the attention layers about the structure of the image. A schematic drawing of this architecture is shown in Figure 5.

<table align="center">
  <tr align="center">
      <td><img src="figures/ViT architecture.png" width=600></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure 5.</b> Schematic depiction of the ViT architecture (INSERT BRON)</td>
  </tr>
</table>

From the above scheme and intuition, it is clear that this architecture is not equivariant to translations and rotations as each translation or rotation results in a completely new patch embedding. On top of that, the positional encodings are not constrained to be equivariant to different group transformations of the input.

---

### GE-ViT Architecture
The method proposed by the paper, known as GE-ViT, is a modified version of the GSA-Net model. The structure of GE-ViT and its attention blocks is visualized in Figure 6. 

<table align="center">
  <tr align="center">
      <td><img src="figures/GEVIT.png" width=600></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure 6.</b> Illustration of GE-ViT and Attention Block </td>
  </tr>
</table>

The model processes the input as follows. First, the lifting self-attention layer transforms the input from the standard 2D space ($R^2$) into a higher-dimensional space where group transformations are encoded, as detailed in the section on GE-ViT attention and positional embedding. The resulting features are then normalized using layer normalization and put through a Swish activation function.

Next, these features are processed through $N$ attention block layers to refine the feature representations. The GE self-attention mechanism works for both lifting self-attention and group self-attention.

Finally, the global pooling block aggregates features in a transformation-invariant manner across the spatial dimensions. Initially, max pooling is applied both spatially and over the group elements. Subsequently, spatial averaging is done on the features to reduce dimensionality further. Both of these aggregations are invariant to group transformations (because they are independent of input order). 

### GE-ViT attention & positional embedding
The main update to the GSA-Net method proposed by the GE-ViT paper is an update of the positional encoding that addresses its group equivariance. This update corrects a minor mathematical error in GSA-Net, leading to a slight modification in the generation of positional encodings.

As opposed to the modern ViTs where patch embeddings attend to different patch embeddings, pixels attend to other pixels in GSA-Nets. As attention has a quadratic time complexity in the number of inputs, attention in GSA-Nets is only applied to an NxN local neighborhood around each pixel. The attention computation is done per entire row and column of the local neighborhood instead of individual pixels to reduce the number of computations (the row and column attention scores are summed after).

To achieve group equivariance, a stack of positional encodings is generated including an encoding for every group element with the corresponding group action applied to it. This is similar to how an input image is stacked in a GE-CNN lifting layer because it takes a base ‘image’ (encoding in this case) and applies the group action for every group element. In practice, this means that for a discrete E(2) group with 90-degree rotations with flips (8 elements in total), a stack of 8 positional encodings is generated.

This works as follows:

* For each group element, a ‘positional embedding index’ is generated by taking a base NxN embedding with 3 channels and applying the group action corresponding to the element. The split row and column embedding indices are visualized in figure TODO number. 


![figure](figures/row_embd_idxs.png)
![figure](figures/col_embd_idxs.png)
TODO make a nice table figure

* The embedding index is put through a learned mini-network with 2 convolutional layers to obtain the final position embeddings. An example row embedding when transformed by this mini-net is visualized in figure TODO number

![figure](figures/row_embd_final.png)
TODO make a nice table figure

* As previously mentioned, the attention scores for rows and columns in the local neighborhood are calculated separately and summed after. We attempted to visualize the combination of the row and column embeddings in Figure TODO number

![figure](figures/combined_embd.png)
TODO make a nice table figure

The final attention mechanism for a pixel in the center of an image was visualized in Figure TODO number. We can see that within an attention head, the attention is transformed according to the group element’s rotation (hence why in the figure the difference between $MHA_0, g_0$ and $MHA_0, g_1$ is a 90-degree rotation, the difference between group elements $g_0$ and $g_1$). It's important to remember that this computation is done for every pixel in the image, we will come back to this when discussing strengths and weaknesses, as well as potential improvements on the method.

![figure](figures/GEVIT_attention.gif)
TODO make a nice table figure



---

## Discussion of ["E(2)-Equivariant Vision Transformer"](https://proceedings.mlr.press/v216/xu23b.html)

In researching and reproducing this paper and its proposed architecture, several aspects caught our attention. While their proposed architecture offers advantages over GSA-nets and traditional Vision Transformers, it also has some weaknesses that we would like to highlight.

[comment]: < Here we say that these methods are computationally expensive and some of our findings. eg steerable but also artifact-like differences (show this with a figure). quickly mention we evaluate on the validation set an increased batch size (and proportionally learning rate) because of computational constraints. Display the results we got for their methods here and say we use the reported numbers of the best method in the following parts of the blogpost. >

<strong> Weaknesses and strengths of proposed method </strong>

The authors of the original equivariant vision transformer claim that their positional encodings result in an E(2) equivariant model. This has the benefit of providing consistent results and predictions for rotated images, a crucial property in fields like medical cell analysis. To illustrate this, we display the latent representations of inputs transformed according to an E(2) group action (rototranslation + mirroring) in the following figure.

![figure](figures/GEVIT_latent_representations_2.gif)

Furthermore, Equivariant networks typically benefit from weight sharing and faster generalization. However, the authors did not explore this advantage in depth, so we have chosen to investigate this further.

In addition to strengths, we also identified several weaknesses in their approach and methods:

1. The original paper claims that the group equivariant vision transformer significantly outperforms non-equivariant self-attention networks. We question this assertion and suggest that retraining a Vision Transformer to be rotation equivariant ad-hoc could significantly enhance performance.

2. A notable weakness of the original GE-VIT is its training and inference time. The authors mention using the well-known ViT, but this is not accurate. The original ViT and the GSA-net paper were published simultaneously in the same journal. The GE-VIT authors used the codebase from the GSA-nets, which employs a vision transformer applying attention to each pixel instead of pixel patches, diverging from the standard vision transformer. This causes the model to be very slow at training and inference time as this pixel-wise attention requires lots of computation.

[comment]: <The following is a bit double, so commented out currently: 4. The Vision Transformer in the GSA-Net architecture uses local self-attention for every single pixel which makes the model translation equivariant but also computationally very expensive. The GE-ViT is based on this GSA-Net model, not the well-known original ViT. This can be made more computationally efficient by dividing the images into patches and applying the equivalent things to those patches. Their proposed architecture does calculations with the input of the following size for MNIST and the lifting layer and rotation of 90 degrees. ([8, 9, 4, 28, 28, 9, 9]) = (batch size, num_heads, the 4 rotations, height, width, patch size, patch size) where patch size refers to the local neighborhood that should be taken into account for attention. For a single pixel, it computes attention to 81 different other pixels and then this is done for all 28*28*9*4=28244 pixels making it a total of 81*28244=2,286,144 attention computations just for the lifting layer. Having this much attention computations causes training and inference time to be slow for even relatively low-resolution images such as MNIST data.>

3. Another issue pertains to the implementation used for training and evaluating the model's performance. Upon inspecting their source code, we discovered that the authors used the test set during training to evaluate their method's performance. They reported the epoch with the best test set performance as the result. This practice can lead to overfitting on the test set, which should only be used to evaluate the final model's performance, not to guide training.

4. The original paper states that their approach is steerable because the positional encoding lives in a continuous space. This however appears to be incorrect because rotations of 45 degrees will get different positional encodings than for 90 degrees. This has been visualized in the Figure below.

![not steerable](figures/not_steerable.png)


<strong> Our novel contribution </strong>

Some of the contributions that we want to add to this paper are already briefly discussed in the section above as we want to improve upon all the weaknesses mentioned. Furthermore, we propose a novel alternate architecture that combines a group-equivariant CNN and a transformer and analyze its performance.

- The ViT we propose uses a patch size (the normal one) of 4x4. So 49 patches in total. Now you compute attention globally to the other patches and itself. If all other settings are the same, you do this for 49*9*4=1764 patches in total, making a total of 49*1764=86,436 attention computations for the lifting layer. This should make it about 26 times faster, so for rotation MNIST 10 hours/26=23 minutes.

<!-- - Maybe do interpretability analyses but I feel like we should not do that. -->

[comment]: <How do we want to visualise attention as in the original ViT paper (okey ik ga die paper wel even bestuderen Figure 6 dus kijken)>


## Post Hoc Equivariant Models
### Introduction
Equivariant models offer many advantages, but they are often memory-intensive and require many training epochs to converge. This has hindered their widespread adoption. To address this issue, we have built upon the work of [Basu et al. 2023](https://arxiv.org/abs/2210.06475) to make any image classification model equivariant with minimal or no fine-tuning, which is computationally as efficient as parameter sharing with the advantage of leveraging pretrained knowledge.

[Basu et al. 2023](https://arxiv.org/abs/2210.06475) achieved this by invariantly aggregating the latent dimensions of transformed inputs. In his work, he proposed several methods, including mean pooling and using a neural network to assign importance scores for a weighted average of embeddings. The pipeline for this method is visualized in the image below:

<table align="center">
  <tr align="center">
      <td><img src="figures/posthocaggregation.png" width=600></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure TODO.</b> Pipeline post hoc equivariant models. </td>
  </tr>
</table>

> This image illustrates how post hoc equivariance works. The input image undergoes various transformations to achieve equivariance (in this case, 90-degree rotations). Each transformed image is processed by the same model, which generates latent embeddings or class probabilities. These embeddings (or probabilities) are then aggregated in an invariant way. Note that the model's latent representations are equivariant to the group (because of the lifting to the group) until they are aggregated, which is done using an invariant operation.

Besides these ways of aggregating the embeddings we propose and evaluate the following ways of aggregating the latent dimensions: sum, max pooling, and multi-head attention without positional encodings. Furthermore, we experiment with predicting the class with the highest probability among all transformations and predicting the class with the highest product of probabilities. In the next section, we will more formally discuss these methods.

### Method
In this section, we drew inspiration from [Basu et al. 2023](https://arxiv.org/abs/2210.06475). They proposed a finetuning method called equituning that starts with potentially non-equivariant model M and produces a model $M_G$ equivariant to a group G. 

Given a set $\chi$, a group action of G on X is defined as $\Gamma X$: $G \times \chi$ -> $\chi$. We write $\Gamma X(g,x)$ simply as gx for brevity. A model M: X -> Y is equivariant to G under the group action of G on X and Y if M(gx) = g(M(x)) for all g $\in$ G, x $\in$ $\chi$. This essentially means that any group transformation g to the input $\Gamma X(g,x)$ should reflect an equivalent group transformation of the output  $\Gamma Y(g,M(x))$.

Equituning converts a pre-trained model into an equivariant version by minimizing the distance of features obtained from pretrained and equivariant models. 

<!-- Demonstrate how (show the important process for minization) and how each one is equivariant  -->

Here, we proposed three methods. The output of an equituned model is given by

- $x$ as the input image.
- $g$ as a transformation in the group $G$.
- $g^{-1}$ as a inverse of the transformation in the group $G$.
- $M(x)$ as the output logits obtained from the original input image $x$.
- $M_G(x)$ as the output logits obtained from the transformed input image $gx$.

Mean Pooling: 

$$ M_G(x) = \frac{\sum_{g \in G}{g^{-1}M(gx)}}{|G|} $$

Max Pooling:

$$ (M_G(x))i = \max_{g \in G}({g^{-1}M(gx)})_i $$

Sum Pooling: 

$$ M_G(x) = \sum_{g \in G}{g^{-1}M(gx)} $$

#### Most Likely and Certain
Instead of combining the embeddings to get the final logits, in this approach, we compute logits for each transformation independently. Here, we propose two methods to select the final logits.

Select Most Likely: Combine them to get the final logits.

$$  M_G(x) = \log \left( \prod_{g \in G}\text{softmax}{(M(gx))} \right) $$

Select Most Certain: Select the transformation with the highest probability for each class. It then selects the logits corresponding to these highest probabilities.

$$  M_G(x) = \text{arg max}_{g \in G} (\text{softmax}{(M(gx))}) $$

#### Learned weighted average
Similar to the idea of λ-equitune in (Sourya Basu (2023). Efficient Equivariant Transfer Learning from Pretrained Models), revolves around recognizing that, within a pretrained model M, features M(gx) derived from fixed x are not uniformly crucial across all transformations g $\in$ G. Let λ(gx) denote the significance weight assigned to feature M(gx) for g $\in$ G, x $\in$ X. Assuming a finite G, as in Basu et al. (2023), λ : X → $R^+$ is predefined. The λ-equituned model, denoted as $M^{λ} {G}$, aims to minimize:

$$\min_{ M_G^{λ}(x)} \sum_{g \in G} || λ(gx) M(gx) -  M_G^{λ}(g,x)||^{2}$$

subject to:

$$ M_G^{λ}(gx) = g M_G^{λ}(x)$$ 

for all g $\in$ G.

The solution to the above equation, referred to as λ-equitune, is given by:

$$ M_G^{λ}(x) = \sum_{g \in G}^{|G|} g^{-1}λ(gx)M(gx) \frac{1}{\sum_{g \in G}{λ(gx)}}$$

#### Learned attention aggregation

This method aggregates the embeddings using a transformer and then passes the combined embedding through the model's MLP head to get the final logits. Since the transformer operations (layer normalization, multi-head attention, and feed-forward networks) do not depend on the order of embeddings, the aggregated result is independent of the transformations applied to the input. The final logits are produced by passing the aggregated embeddings through the MLP head. This process is invariant to the transformations since it operates on the aggregated embeddings representing the transformed input space.

   $$
   M_G(x) = \text{Mlp}(\text{Transformer}(M(gx))), g\in G
   $$

Since the aggregation model (transformer) is designed to handle sequences of embeddings in an order-invariant manner (due to the self-attention mechanism), the output should remain consistent under the same group transformations applied to the input and the output:

$$
 M_G(x) = g( M_G(x))
$$

Therefore, the `PH Learned Attention Aggregation` model is equivariant by design because the transformer aggregation maintains the equivariance property through its self-attention mechanism and the consistent application of transformations across the input space. Using the class token ensures that the final output logits are derived in a manner that respects the input transformations.

### Results
We evaluated all approaches through various experiments, examining their zero-shot impact, the effect of fine-tuning only the last layer, and the impact of fine-tuning the entire model.

We conducted the following experiments:
1. Training and evaluating on the full rotation MNIST dataset. This was done to compare our novel methods to GE-ViT.
2. Training on the full standard MNIST dataset and evaluating on the full rotation MNIST dataset. This was done to prove that the models were group equivariant, even when trained on a dataset that didn't include samples that were transformed by the group action.
3. Training on 10% of rotation MNIST and evaluating on the full rotation MNIST dataset. This was done to test the models' data efficiency.
4. Evaluating a pretrained RESNET-50 on Patch Camelyon. This was done to compare our novel methods to GE-ViT, and because we deemed Patch Camelyon to be more interesting than MNIST in terms of its real-life relevance.

> **NOTE:** We provide links to all relevant code and markdowns for reproducing our results in the [Experimental Details](#experimental-details) section.

#### 1. Training and evaluating on the full rotation MNIST dataset
In the first experiment, we trained and evaluated on rotation MNIST as done in [GeViT] and [GSA-Nets] (https://arxiv.org/abs/2010.00977). Below are the report test accuracies of the best models, obtained with a patch size of 5 and 12 rotations.

<table>
<thead>
<tr>
<th align="center">Model</th>
<th align="center">Best Reported Test Accuracy</th>
</tr>
</thead>

<tbody>
<tr>
<td align="center">GSA - Nets</td>
<td align="center">97.97</td>
</tr>
<tr>
<td align="center">GE-ViT</td>
<td align="center">98.01</td>
</tr>
</tbody>
</table>

Below we present the results for the different aggregation methods without fine-tuning, with fine-tuning the last layer, and with fine-tuning the entire model. The first row (ViT) represents the Vision Transformer we trained on rotation MNIST, and we applied our framework to this model.

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th colspan = 2>No finetuning</th>
            <th colspan = 2>MLP finetuning</th>
            <th colspan = 2> Model finetuning </th>
        </tr>
        <tr>
            <th></th>
            <th align="center">Validation Accuracy</th>
            <th align="center">Test Accuracy</th>
            <th >Validation Accuracy</th>
            <th >Test Accuracy</th>
            <th >Validation Accuracy</th>
            <th >Test Accuracy</th>
        </tr>
    </thead>
    <tbody>
<tr>
<td>Normal ViT</td>
<td align="center">97.55</td>
<td align="center">97.23</td>
<td align="center">97.55</td>
<td align="center">97.23</td>
<td align="center">97.55</td>
<td align="center">97.23</td>
</tr>
<tr>
<td>PH Mean Pooling</td>
<td align="center">98.00</td>
<td align="center">98.07</td>
<td align="center">98.20</td>
<td align="center">98.20</td>
<td align="center">98.70</td>
<td align="center">98.24</td>
</tr>
<tr>
<td>PH Max Pooling</td>
<td align="center">97.80</td>
<td align="center">97.82</td>
<td align="center">98.10</td>
<td align="center">98.12</td>
<td align="center">98.65</td>
<td align="center">98.31</td>
</tr>
<tr>
<td>PH Sum Pooling</td>
<td align="center">98.00</td>
<td align="center">98.07</td>
<td align="center">98.20</td>
<td align="center">98.20</td>
<td align="center">98.70</td>
<td align="center">98.24</td>
</tr>
<tr>
<td>PH Most Likely</td>
<td align="center">97.90</td>
<td align="center">98.09</td>
<td align="center">98.15</td>
<td align="center">98.14</td>
<td align="center">98.45</td>
<td align="center">98.18</td>
</tr>
<tr>
<td>PH Most Certain</td>
<td align="center">97.75</td>
<td align="center">97.63</td>
<td align="center">97.95</td>
<td align="center">97.94</td>
<td align="center">98.40</td>
<td align="center">98.10</td>
</tr>
<tr>
<td>PH Learned Weighted Average</td>
<td align="center">96.60</td>
<td align="center">96.46</td>
<td align="center">95.65</td>
<td align="center">95.48</td>
<td align="center">96.60</td>
<td align="center">96.46</td>
</tr>
<tr>
<td>PH Learned Attention Aggregation</td>
<td align="center">96.80</td>
<td align="center">96.75</td>
<td align="center">96.65</td>
<td align="center">96.33</td>
<td align="center">96.80</td>
<td align="center">96.75</td>
</tr>
    </tbody>
</table>

The table above presents interesting results on many accounts. First, it shows that, except for the methods that require learning, all methods improve the results from the base model. This is surprising because, for example, summing or taking the maximum element of the latent embeddings would likely significantly alter the embeddings the final layers saw during training. Nevertheless, the model finds a way to project these modified embeddings to logits accurately. 

Second, as expected, the dominant trend shows that fine-tuning the last layer leads to better results, with further improvements observed when fine-tuning the entire model. Third, it can be seen that multiple models outperform the best-reported baselines, even after only fine-tuning the final layer. All models that aggregate without additional parameters outperform the baselines when we fine-tune the entire model. This means that better results can be achieved without using inherently equivariant models, however, post hoc augmentations are essential in this experiment.

We found that in this experiment, learning to score the embeddings or learning to aggregate the embeddings with multi-head attention led to overfitting, resulting in worse validation and test accuracies.

We also aimed to visualize the behavior of the LearnedAggregation model's transformer block when its input is transformed by a group action to which the model is supposed to be equivariant. For an image that is rotated and flipped in 8 different ways, we visualized how the various tokens (transformed embeddings for each group element) and the CLS token self-attend across the 4 transformer layers. This visualization is shown in Figure [REF]. Notice how the CLS token, which is used for the final scoring, attends to different columns (representing different transformed input embeddings) for each transformation.

<table align="center">
  <tr align="center">
      <td><img src="figures/PostHocLearnedAggregation_attention_visualisation.png" width=600></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure TODO NUMBER.</b> Self-attention weights in the LearnedAggregation post hoc equivariant ViT for input transformed by various group actions.</td>
  </tr>
</table>


 
#### 2. Training on the full standard MNIST dataset and evaluating on the full rotation MNIST dataset
One of the advantages of purely equivariant models is that the training data can be in different orientations than the test data, as long as the transformations between the training and testing examples are within the group the model is equivariant to. This is typically not the case for non-equivariant models. Therefore, we now test how well post hoc methods can improve the performance of non-equivariant models.

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th colspan = 2>No finetuning</th>
            <th colspan = 2>MLP finetuning</th>
            <th colspan = 2> Model finetuning </th>
        </tr>
        <tr>
            <th></th>
            <th >Validation Accuracy</th>
            <th >Test Accuracy</th>
            <th >Validation Accuracy</th>
            <th >Test Accuracy</th>
            <th >Validation Accuracy</th>
            <th >Test Accuracy</th>
        </tr>
    </thead>
    <tbody>

<tr>
<td>Normal ViT</td>
<td align="center">29.75</td>
<td align="center">28.72</td>
<td align="center">29.75</td>
<td align="center">28.72</td>
<td align="center">29.75</td>
<td align="center">28.72</td>
</tr>
<tr>
<td>PH Mean Pooling</td>
<td align="center">43.50</td>
<td align="center">43.31</td>
<td align="center">84.90</td>
<td align="center">85.50</td>
<td align="center">97.40</td>
<td align="center">97.53</td>
</tr>
<tr>
<td>PH Max Pooling</td>
<td align="center">41.65</td>
<td align="center">41.89</td>
<td align="center">86.75</td>
<td align="center">87.13</td>
<td align="center">97.35</td>
<td align="center">97.28</td>
</tr>
<tr>
<td>PH Sum Pooling</td>
<td align="center">43.50</td>
<td align="center">43.31</td>
<td align="center">84.90</td>
<td align="center">85.50</td>
<td align="center">97.40</td>
<td align="center">97.53</td>
</tr>
<tr>
<td>PH Most Likely</td>
<td align="center">30.05</td>
<td align="center">29.30</td>
<td align="center">74.32</td>
<td align="center">72.91</td>
<td align="center">76.89</td>
<td align="center">75.47</td>
</tr>
<tr>
<td>PH Most Certain</td>
<td align="center">47.95</td>
<td align="center">49.00</td>
<td align="center">82.95</td>
<td align="center">82.52</td>
<td align="center">96.30</td>
<td align="center">96.26</td>
</tr>
<tr>
<td>PH Learned Weighted Average</td>
<td align="center">83.80</td>
<td align="center">83.28</td>
<td align="center">87.35</td>
<td align="center">87.00</td>
<td align="center">94.90</td>
<td align="center">94.99</td>
</tr>
<tr>
<td>PH Learned Attention Aggregation</td>
<td align="center">92.75</td>
<td align="center">92.43</td>
<td align="center">92.75</td>
<td align="center">92.41</td>
<td align="center">96.65</td>
<td align="center">96.16</td>
</tr>
        <tr>
    </tbody>
</table>

The non-equivariant model trained on MNIST attained an accuracy of around 29% on rotation MNIST. This model can accurately predict the digits in their normal orientation but struggles with heavily rotated images. The best approach that doesn't require any learning is selecting the transformation that gives the highest probability to one particular class, achieving a test accuracy of 49%. This can be explained by the model being uncertain about transformations of digits that were unseen during training but assigning high probability to digits close to their original orientation. However, a digit 6 rotated by 180 degrees is likely to be predicted as a 9, indicating that this method is not foolproof.

Interestingly, the models that require learning perform much better in this experiment. Aggregating the embeddings with multi-head attention even leads to an accuracy of about 92% while keeping all parameters of the original model fixed. However, similar to the previous results, when fine-tuning the entire model, taking the mean or sum of the embeddings proves to be more effective.

Another interesting insight is the MostProbable model's behavior with input that is rotated to look like another number (e.g. a 6 flipped vertically to look like a 9). It seems to perform well in these edge cases, assigning more probability to the correct class in many of these cases. One such case was visualized in Figure TODO number.

<table align="center">
  <tr align="center">
      <td><img src="figures/meanagg_vs_mostprobable.png" width=600></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure TODO NUMBER.</b> Comparing predicted probabilities from the MeanAggregation and MostProbable post hoc equivariant transformer models in an edge case where a 6 is rotated to possibly be confused with a 9.</td>
  </tr>
</table>



#### 3. Training on 10% of rotation MNIST and evaluating on the full rotation MNIST dataset
One key advantage of equivariant models is their data efficiency, owing to the way inductive biases are incorporated (TODO citation needed). Therefore, we now compare the performance of post hoc methods against the equivariant models used in (TODO paper citation needed) when training on only 10% of the rotated MNIST dataset.

We also evaluated the performance of GE-ViT when trained on a reduced dataset comprising only 10% of the total data. Interestingly, we observed that this model reached convergence after 600 epochs, achieving a test accuracy of only 83.55%. However, this accuracy was notably lower compared to when the model was trained on the complete training set.

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th colspan = 2>No finetuning</th>
            <th colspan = 2>MLP finetuning</th>
            <th colspan = 2> Model finetuning </th>
        </tr>
        <tr>
            <th></th>
            <th >Validation Accuracy</th>
            <th >Test Accuracy</th>
            <th >Validation Accuracy</th>
            <th >Test Accuracy</th>
            <th >Validation Accuracy</th>
            <th >Test Accuracy</th>
        </tr>
    </thead>
    <tbody>

<tr>
<td>Normal ViT</td>
<td align="center">86.25</td>
<td align="center">86.21</td>
<td align="center">86.25</td>
<td align="center">86.21</td>
<td align="center">86.25</td>
<td align="center">86.21</td>
</tr>
<tr>
<td>PH Mean Pooling</td>
<td align="center">88.50</td>
<td align="center">89.00</td>
<td align="center">88.75</td>
<td align="center">89.20</td>
<td align="center">89.30</td>
<td align="center">89.73</td>
</tr>
<tr>
<td>PH Max Pooling</td>
<td align="center">88.15</td>
<td align="center">88.73</td>
<td align="center">88.40</td>
<td align="center">88.96</td>
<td align="center">88.90</td>
<td align="center">89.41</td>
</tr>
<tr>
<td>PH Sum Pooling</td>
<td align="center">88.50</td>
<td align="center">88.99</td>
<td align="center">88.75</td>
<td align="center">89.20</td>
<td align="center">89.30</td>
<td align="center">89.73</td>
</tr>
<tr>
<td>PH Most Likely</td>
<td align="center">88.60</td>
<td align="center">89.03</td>
<td align="center">88.90</td>
<td align="center">89.12</td>
<td align="center">86.75</td>
<td align="center">87.19</td>
</tr>
<tr>
<td>PH Most Certain</td>
<td align="center">87.10</td>
<td align="center">87.88</td>
<td align="center">87.75</td>
<td align="center">88.34</td>
<td align="center">88.50</td>
<td align="center">88.87</td>
</tr>
<tr>
<td>PH Learned Weighted Average</td>
<td align="center">80.30</td>
<td align="center">80.75</td>
<td align="center">78.95</td>
<td align="center">79.10</td>
<td align="center">81.85</td>
<td align="center">82.27</td>
</tr>
<tr>
<td>PH Learned Attention Aggregation</td>
<td align="center">82.25</td>
<td align="center">82.66</td>
<td align="center">82.15</td>
<td align="center">82.63</td>
<td align="center">84.50</td>
<td align="center">84.44</td>
</tr>
    </tbody>
</table>

In line with our findings on the full rotation MNIST dataset, we find that learning to aggregate the embeddings often resulted in overfitting. Also, the other methods consistently enhanced performance, with fine-tuning showing even greater improvements, and mean pooling and summing of latent dimensions led to the highest accuracies. Remarkably, our non-equivariant ViT already surpassed the GE-ViT in performance. Throughout our experiments, we observed that GE-ViT exhibited high sensitivity to minor changes in hyperparameters. We hypothesize that optimizing hyperparameters could potentially yield higher accuracy. However, due to computational constraints, we didn't explore this further.

#### 4. Evaluating a pretrained RESNET-50 on Patch Camelyon
Until now, our analysis has been limited to models trained exclusively on either MNIST or rotation MNIST. However, it's worth noting that the post hoc methods discussed can be applied to pretrained models trained on various datasets. To validate the robustness of our findings, we use a [trained RESNET-50 model](https://huggingface.co/1aurent/resnet50.tiatoolbox-pcam). We are grateful to [TIAToolbox](https://tia-toolbox.readthedocs.io/en/latest/?badge=latest) for offering a wide range of pretrained models for this dataset, we found it challenging to find well-performing models elsewhere.

<table>
<thead>
<tr>
<th align="center">Model</th>
<th align="center">Best Reported Test Accuracy</th>
</tr>
</thead>

<tbody>
<tr>
<td align="center">GSA - Nets</td>
<td align="center">82.26</td>
</tr>
<tr>
<td align="center">GE-ViT</td>
<td align="center">83.82</td>
</tr>
</tbody>
</table>

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th colspan = 2>No finetuning</th>
            <th colspan = 2>finetuning</th>
        </tr>
        <tr>
            <th></th>
            <th >Validation Accuracy</th>
            <th >Test Accuracy</th>
            <th >Validation Accuracy</th>
            <th >Test Accuracy</th>
        </tr>
    </thead>
    <tbody>
<tr>
<td>Normal ViT</td>
<td align="center">87.76</td>
<td align="center">86.61</td>
<td align="center">87.76</td>
<td align="center">86.61</td>
</tr>
<tr>
<td>PH Mean Pooling</td>
<td align="center">88.07</td>
<td align="center">87.27</td>
<td align="center">90.40</td>
<td align="center">87.52</td>
</tr>
<tr>
<td>PH Max Pooling</td>
<td align="center">86.67</td>
<td align="center">86.15</td>
<td align="center">90.083535</td>
<td align="center">86.94</td>
</tr>
<tr>
<td>PH Sum Pooling</td>
<td align="center">88.29</td>
<td align="center">87.43</td>
<td align="center">90.33</td>
<td align="center">87.15</td>
</tr>
<tr>
<td>PH Most Likely</td>
<td align="center">88.07</td>
<td align="center">87.27</td>
<td align="center">90.37</td>
<td align="center">87.14</td>
</tr>
<tr>
<td>PH Most Certain</td>
<td align="center">87.90</td>
<td align="center">87.11</td>
<td align="center">90.04</td>
<td align="center">86.98</td>
</tr>
    </tbody>
</table>
We assessed the pretrained model's performance on PCam's validation and test sets, revealing accuracies of 87.8% and 86.6% respectively. Remarkably, employing mean pooling boosted these accuracies to 90.4% and 87.5% through just 1 epoch of fine-tuning. This could be because the mean pooling equation minimizes the distance between the features obtained by a pretrained model and the equivariant model the most compared to the other two methods. Additionally, both summing and the most probable method yielded favorable results in this experiment, underscoring the efficacy of incorporating inductive biases into pretrained models to enhance performance.

A notable observation is the considerable gap between validation and test accuracies. Given our single-epoch fine-tuning approach on the training set, it's strange why the validation accuracy would differ significantly from the test accuracy. Similar trends were noticed in other PCam experiments, where models exhibited rapid validation accuracy growth but more subdued improvements on the test set. We encourage further study on this dataset as it is a widely used benchmark.

Furthermore, our findings demonstrate the superiority of large models over GSA-nets and GE-ViTs. However, it's crucial to acknowledge the parameter discrepancy, comparing models with 45,000 parameters to one of 23.6 million. Attempting to scale up GE-ViTs to over a million parameters was not feasible for us, exceeding the limitations of our 40GB RAM GPU with a batch size of 1. While this indicates poor scalability, it's worth exploring how these models perform when scaled up, given our evidence that integrating inductive biases significantly enhances performance.

## Speeding up GE-ViT
#### Introduction
GSA-Nets and GE-ViTs are slow because attention is applied on the pixel level. We propose two methods to reduce the spatial resolution of the input to GE-ViT which we dub the artificial image. 

The first approach is to split the image into patches that are lifted to the group. Subsequently, each transformation is projected and the resulting embeddings are averaged over the group producing an embedding for the patch. The image structure is preserved, but the artificial image contains the embeddings for the patches with dimensionality 64. This pipeline is visualized in the image below:
<table align="center">
  <tr align="center">
      <td><img src="figures/modern_eq_vit.png" width=600></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure TODO.</b> Pipeline for creating an artificial image composed of group invariant patch embeddings. </td>
  </tr>
</table>

> As in the normal ViT we create patches, however, these patches are lifted to the group and the resulting transformations are each processed by the same model. Afterward, the images are invariantly aggregated over the group and reshaped to the original image structure but now with fewer pixels. The resulting artificial image is fed to GE-ViT as normal.

The second approach is to process the image with a group convolutional neural network. Here we don't pad the image so the spatial resolution naturally decreases. Here were output 32 E(2) invariant feature maps with spatial resolution 28 by 28. This stack of feature maps is the artificial image on which we apply GE-ViT. This process is visualized below:

<table align="center">
  <tr align="center">
      <td><img src="figures/hybrid.png" width=600></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure TODO.</b> Pipeline for creating an artificial image composed of group invariant feature maps from a GCNN. </td>
  </tr>
</table>

> The image is processed with a GCNN that provides group-invariant feature maps. These feature maps have lower spatial resolution and are the input to GE-ViT.

#### Results

##### Approach 1: projecting patches to group invariant embeddings
Because of computational constraints we train on 1 percent of the data for 50 epochs and report the accuracy after 50 epochs on 10% of the test data. Therefore our results in this section are indicative and not conclusive. 

<table align="center">
  <tr align="center">
      <td><img src="figures/ablation.png" width=600></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure TODO.</b> Accuracy vs. epoch time for different patch sizes. </td>
  </tr>
</table>

> Reducing the spatial dimension of the input significantly decreases the traning time, however, leads to a decrease in performance compared to GE-ViT.

The results of this experiment are visualized in Figure (TODO give figure number). First, it can be observed that increasing the patch size decreases training time significantly. This is because reducing the spatial resolution decreases the number of computations by GE-ViT. Second, we observe the optimal patch size, in terms of accuracy, is 6. Smaller patch sizes likely lead to the splitting up of cells making the classification more challenging. Larger patch sizes can likely not be represented well with a 64-dimensional embedding obtained from a linear projection. If our hypothesis is correct, this shows the downside of using the modern ViT, which uses patches, for particular applications where a few pixels determine the label of the image (as in detection of pathological cells). This is illustrated in the figure below where black arrows indicate note tumor cell infiltration ([image courtesy](https://www.researchgate.net/publication/337826655_WSZG_inhibits_BMSC-induced_EMT_and_bone_metastasis_in_breast_cancer_by_regulating_TGF-b1Smads_signaling)).

<table align="center">
  <tr align="center">
      <td><img src="figures/pathological_cell.png" width=500></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure TODO.</b> Metastatic tissue, black arrows indicate tumor cell infiltration. </td>
  </tr>
</table>

> The image above is not part of the Patch Camelyon dataset, however, is indicative of the difficulty. For the Patch Camelyon dataset is holds that ["A positive label indicates that the center 32x32px region of a patch contains at least one pixel of tumor tissue"](https://github.com/basveeling/pcam/tree/master?tab=readme-ov-file#details).

##### Approach 2: downsizing image with GCNN


## Concluding Remarks
In this blog post, we conducted a thorough evaluation of existing methods while also proposing new approaches to achieve equivariance in pretrained models with minimal to no fine-tuning. Our findings highlight mean pooling of latent dimensions as the most reliable method, consistently delivering strong performance across all experiments. Notably, performance saw incremental gains with fine-tuning the last layer and further enhancements when fine-tuning the entire model. Importantly, this simple yet effective approach outperforms equivariant attention models without altering their original training paradigms.

It's crucial to recognize that this approach only confers equivariance to global transformations. Additionally, since the pretrained models lacked translation equivariance, the final models remain non-translation equivariant, solely equivariant to the O(2) and/or SO(2) groups. However, applying these methods to CNNs, inherently translation equivariant, would yield equivariance to the E(2) and/or SE(2) groups. We advocate for researchers to explore these methods' efficacy when applied to CNNs and across diverse datasets.

Post Hoc Equivariant models:
| Pros                         | Cons                           |
|-------------------------------|---------------------------------|
|              Directly applicable to trained image classification models                |                  Only global invariant               |
|               Simple                |              Not straightforward to apply to different tasks                   |
|              Proven to be effective                 |              Susceptible to interpolation effects as discrete transformations are used                   |
|              Provides geometric guarantees                 |              Slows down model during inference                   |
|              Scalability is determined by the base model                  |              Base model wastes parameters to learn equivariant properties                    |

To address these cons, future work could explore the following ideas:
- *Only global invariant*: apply the transformations to patches of the image as opposed to the whole image.
- *Not straightforward to apply to different tasks*: for graph neural networks one could consider combining the outputs from equivalent slightly different permutations. With only slightly changing the input one can reuse the majority of computations from other permutations.
- *Susceptible to interpolation effects as discrete transformations are used*: for rotations, one could pad the image with black pixels to make the image a circle. Now don't make square patches but rings and split these up into multiple parts to preserve locality. During training mask out the black pixels.
- *Slows down model during inference*: although all transformations are processed simultaneously, one could explore the possibility of predicting the useful transformations with a small neural network to save computations spent on uninformative/misleading transformations.

*Base model wastes parameters to learn equivariant properties*: the only ways to solve this are either by using an equivariant network, or similarly transforming all inputs. The former gives the most guarantees, but are parameters a more important metric than training/inference time?

Further, we explored downsizing the image with (1) projecting patches and (2) downsizing with a GCNN. Both approaches significantly increase the speed of the models but are less accurate. We believe that with the right architecture and learning paradigm the second approach can become competitive with SOTA approaches as this is also the case for their non-equivariant counterparts for non-equivariant image classification ([Graham, B, et. al.](https://arxiv.org/abs/2104.01136)).

We firmly believe that true equivariant models possess the potential to achieve superior performance, requiring less data, particularly with ample computational resources. However, in scenarios where computational resources are limited for training, the post hoc methods presented here offer a viable solution. They can be directly applied to pretrained models, robustly enhancing performance by significant margins.

## Experimental Details

Because of the amount of experiments we conducted, it is not practical to make an extensive list of hyperparameters and design choices. We refer anyone interested directly to our code.

#### Results GE-ViT:
- [Reproduction](src/README.md)

#### Post Hoc Experiments:
- [Implementation of the various methods](src/post_hoc_equivariance/post_hoc_equivariant.py)
- [Implementation of the sub-models used for learned aggregation](src/post_hoc_equivariance/sub_models.py)
- [Most experiments](src/post_hoc_equivariance/post_hoc_experiments.py)
- [Experiments for the pretrained Resnet](src/post_hoc_equivariance/resnet.py)
- [Training for the non-equivariant ViT on rotation MNIST](src/post_hoc_equivariance/train_vit.py)
- [Training for the non-equivariant ViT on normal MNIST](src/post_hoc_equivariance/mnist.py)
- [Reproduction](src/post_hoc_equivariance/README.md)

#### Learned downsizing of image:
- [Experiments + implementation for patches](src/modern_eq_vit/eq_modern_vit.py)
- [Implementation GCNN](src/models/gcnn.py)
- [Implementation hybrid model (GCNN + GE-ViT)](src/models/hybrid.py)
- [Reproduction](src/modern_eq_vit/README.md)

## Individual contributions

Wouter Bant:
Came up with and implemented the post hoc methods and the projection of patches for reduced image resolution. Wrote and made pictures for the experiments and conclusion sections. Oversaw the running of experiments on Snellius.

Colin Bot:
Worked mostly on inspecting and visualizing the GE-ViT inner workings. Implemented code for most model interpretability figures and wrote the section on GE-VIT's equivariant positional embedding. 

Jasper Eppink:
Contributed mainly to the writing of the blogpost. Has written the introduction and the weaknesses. Also aided with thinking about the programming challenges and the novelties that we wanted to research. 

Clio Feng:
Came up with various experiments. Did the math section for the post hoc methods to show equivariance and wrote the section on GE-ViT architecture. Implemented the code for training on MNIST and on a fraction of the dataset. 

Floris Six Dijkstra:
Came up with and implemented the method of downsizing the image with a GCNN.