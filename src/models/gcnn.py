## Standard libraries
import os
import numpy as np
import math
from PIL import Image
from functools import partial

## Imports for plotting
import matplotlib
import matplotlib.pyplot as plt
## PyTorch
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
## Torchvision
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.transforms.functional import hflip

class GroupBase(torch.nn.Module):

    def __init__(self, dimension, identity):
        """ Implements a group.

        @param dimension: Dimensionality of the group (number of dimensions in the basis of the algebra).
        @param identity: Identity element of the group.
        """
        super().__init__()
        self.dimension = dimension
        self.register_buffer('identity', torch.Tensor(identity))

    def elements(self):
        """ Obtain a tensor containing all group elements in this group.
        
        """
        raise NotImplementedError()

    def product(self, h, h_prime):
        """ Defines group product on two group elements.

        @param h: Group element 1
        @param h_prime: Group element 2
        """
        raise NotImplementedError()

    def inverse(self, h):
        """ Defines inverse for group element.

        @param h: A group element from subgroup H.
        """
        raise NotImplementedError()

    def left_action_on_R2(self, h, x):
        """ Group action of an element from the subgroup H on a vector in R2.

        @param h: A group element from subgroup H.
        @param x: Vectors in R2.
        """
        raise NotImplementedError()

    def matrix_representation(self, h):
        """ Obtain a matrix representation in R^2 for an element h.

        @param h: Group element
        """
        raise NotImplementedError()

    def determinant(self, h):
        """ Calculate the determinant of the representation of a group element
        h.

        @param g:
        """
        raise NotImplementedError()
    
    def normalize_group_parameterization(self, h):
        """ Map the group elements to an interval [-1, 1]. We use this to create
        a standardized input for obtaining weights over the group.

        @param g:
        """
        raise NotImplementedError()



class CyclicGroup(GroupBase):

    def __init__(self, order):
        super().__init__(
            dimension=1,
            identity=[0.]
        )

        assert order > 1
        self.order = torch.tensor(order)

    def elements(self):
        """ Obtain a tensor containing all group elements in this group.
        
        @returns elements: Tensor containing group elements of shape [self.order]
        """
        return torch.linspace(
            start=0,
            end=2 * np.pi * float(self.order - 1) / float(self.order),
            steps=self.order,
            device=self.identity.device
        )
    
    def product(self, h, h_prime):
        """ Defines group product on two group elements of the cyclic group C4.

        @param h: Group element 1
        @param h_prime: Group element 2
        
        @returns product: Tensor containing h \cdot h_prime with \cdot the group action.
        """
        # As we directly parameterize the group by its rotation angles, this 
        # will be a simple addition. Don't forget the closure property though!

        ## YOUR CODE STARTS HERE ##
        product = torch.remainder(h + h_prime, 2 * np.pi)
        ## AND ENDS HERE ##

        return product

    def inverse(self, h):
        """ Defines group inverse for an element of the cyclic group C4.

        @param h: Group element
        
        @returns inverse: Tensor containing h^{-1}.
        """
        # Implement the inverse operation. Keep the closure property in mind!

        ## YOUR CODE STARTS HERE ##
        inverse = torch.remainder(-h, 2 * np.pi)
        ## AND ENDS HERE ##

        return inverse
    
    def left_action_on_R2(self, h, x):
        """ Group action of an element from the subgroup H on a vector in R2.

        @param h: A group element from subgroup H.
        @param x: Vectors in R2.
        
        @returns transformed_x: Tensor containing \rho(h)x.
        """
        # Transform the vector x with h, recall that we are working with a left-regular representation, 
        # meaning we transform vectors in R^2 through left-matrix multiplication.
        transformed_x = torch.tensordot(self.matrix_representation(h), x, dims=1)       
        return transformed_x

    def matrix_representation(self, h):
        """ Obtain a matrix representation in R^2 for an element h.

        @param h: A group element.
        
        @returns representation: Tensor containing matrix representation of h, shape [2, 2].
        """
        ## YOUR CODE STARTS HERE ##
        cos_t = torch.cos(h)
        sin_t = torch.sin(h)

        representation = torch.tensor([
            [cos_t, -sin_t],
            [sin_t, cos_t]
        ], device=self.identity.device)
        ## AND ENDS HERE ##

        return representation
    
    def normalize_group_elements(self, h):
        """ Normalize values of group elements to range between -1 and 1.
        The group elements range from 0 to 2pi * (self.order - 1) / self.order,
        so we normalize accordingly.

        @param h: A group element.
        @return normalized_h: Tensor containing normalized value corresponding to element h.
        """
        largest_elem = 2 * np.pi * (self.order - 1) / self.order
        normalized_h = (2*h / largest_elem) - 1.
        return normalized_h

# Some tests to verify our implementation.
c4 = CyclicGroup(order=4)
e, g1, g2, g3 = c4.elements()

assert c4.product(e, g1) == g1 and c4.product(g1, g2) == g3
assert c4.product(g1, c4.inverse(g1)) == e

assert torch.allclose(c4.matrix_representation(e), torch.eye(2))
assert torch.allclose(c4.matrix_representation(g2), torch.tensor([[-1, 0], [0, -1]]).float(), atol=1e-6)

assert torch.allclose(c4.left_action_on_R2(g1, torch.tensor([0., 1.])), torch.tensor([-1., 0.]), atol=1e-7)





class E2Group(GroupBase):##############second iteration

    def __init__(self, order):
        super().__init__(
            dimension=1,
            identity= [0.]
        )

        assert order > 1
        self.order = torch.tensor(order)
        self.e = 0.01 #set to 0 for more accurate (float64), doesnt work with  float 32 precision
    
    def elements(self):
        """ Obtain a tensor containing all group elements in this group.
        
        @returns elements: Tensor containing group elements of shape [self.order, 2]
        """
        return torch.linspace(
            start=0,
            end=4 * np.pi * float(self.order*2 - 1) / float(self.order*2), ##new coding of rotation, >2pi means also a flip, hacky AF
            steps=self.order*2,
            device=self.identity.device,
            dtype = torch.float32)
        
    
    def trans_xh(self, x):
        return (x%(2*np.pi), int(x>=(2*np.pi-self.e)))

    def trans_hx(self, h):
        return  h[0] + h[1] * 2*np.pi 

    def ind_product(self, x, x_p):
        h= self.trans_xh(x)
        h_p = self.trans_xh(x_p)
        if h_p[1]==0:
            rotation = torch.remainder(h[0] + h_p[0], 2 * np.pi)
            flip = h[1]
        elif h_p[1]==1:
            rotation = torch.remainder(-h[0] + h_p[0], 2 * np.pi)
            flip = 1 - h[1]
        else:
            print("that was unaccounted for")
        return self.trans_hx([rotation, flip])

    def product(self, x, x_prime):
        """ Defines group product on two group elements of the cyclic group C4.

        @param h: Group element 1
        @param h_prime: Group element 2
        
        @returns product: Tensor containing h \cdot h_prime with \cdot the group action.
        """
        # As we directly parameterize the group by its rotation angles, this 
        # will be a simple addition. Don't forget the closure property though!
        ## YOUR CODE STARTS HERE ##
        if len(x_prime.shape)==0:
            #print(f"Single, {x}, {x_prime}")
            res = self.ind_product(x, x_prime)
            return res
        else:
            res = []
            for x_p in x_prime:
                ind_res = self.ind_product(x, x_p)
                res.append(ind_res)
            return torch.tensor(res)

        ## AND ENDS HERE ##

    def ind_inverse(self, x):
        if x>=np.pi*2-self.e:
            return x
        else:
            return torch.remainder(-x, 2. * np.pi)


    def inverse(self, h):
        """ Defines group inverse for an element of the cyclic group C4.

        @param h: Group element
        
        @returns inverse: Tensor containing h^{-1}.
        """
        # Implement the inverse operation. Keep the closure property in mind!

        ## YOUR CODE STARTS HERE ##
        #if not isinstance(h, list) or not isinstance(h, torch.tensor) or not isinstance(h, torch.Tensor): h = [h]
        if len(h.shape)==0: 
            return self.ind_inverse(h)
        else:
            res = []
            for x in h:
                res.append(self.ind_inverse(x))
            return torch.tensor(res)
        ## AND ENDS HERE ##
    
    def left_action_on_R2(self, x, vector_in2):
        """ Group action of an element from the subgroup H on a vector in R2.

        @param h: A group element from subgroup H.
        @param x: Vectors in R2.
        
        @returns transformed_x: Tensor containing \rho(h)x.
        """
        # Transform the vector x with h, recall that we are working with a left-regular representation, 
        # meaning we transform vectors in R^2 through left-matrix multiplication.
        transformed_x = torch.tensordot(self.matrix_representation(x), vector_in2, dims=1)       
        return transformed_x

    def matrix_representation(self, x):
        """ Obtain a matrix representation in R^2 for an element h.

        @param h: A group element.
        
        @returns representation: Tensor containing matrix representation of h, shape [2, 2].
        """
        ## YOUR CODE STARTS HERE ##
        h = self.trans_xh(x)
        cos_t = torch.cos(h[0])
        sin_t = torch.sin(h[0])
        flip_matrix = np.eye(2)
        if h[1]==1:
            flip_matrix[1,1]=-1        
        
        #first flip then rotate
        representation = torch.tensor(
            flip_matrix@[[cos_t, -sin_t], 
            [sin_t, cos_t]
        ], device=self.identity.device, dtype = torch.float32)
        ## AND ENDS HERE ##

        return representation
    
    def normalize_group_elements(self, h):
        """ Normalize values of group elements to range between -1 and 1.
        The group elements range from 0 to 2pi * (self.order - 1) / self.order,
        so we normalize accordingly.

        @param h: A group element.
        @return normalized_h: Tensor containing normalized value corresponding to element h.
        """
        largest_elem = 2 * np.pi * (self.order - 1) / self.order
        normalized_h = (2*h / largest_elem) - 1.
        return normalized_h




e2 = E2Group(order=4)
e, g1, g2, g3, g4, g5, g6, g7 = e2.elements()

assert e2.product(e, g1) == g1 and e2.product(g1, g2) == g3

#doesn't pass assertion with float32, does with float64
#assert torch.allclose(e2.product(g1, e2.inverse(g1)), e, atol = 1e-2)

assert torch.allclose(e2.matrix_representation(e), torch.eye(2))
assert torch.allclose(e2.matrix_representation(g2), torch.tensor([[-1, 0], [0, -1]]).float(), atol=1e-6)

assert torch.allclose(e2.left_action_on_R2(g1, torch.tensor([0., 1.])), torch.tensor([-1., 0.]), atol=1e-7)


def bilinear_interpolation(signal, grid):
    """ Obtain signal values for a set of gridpoints through bilinear interpolation.
    
    @param signal: Tensor containing pixel values [C, H, W] or [N, C, H, W]
    @param grid: Tensor containing coordinate values [2, H, W] or [2, N, H, W]
    """
    # If signal or grid is a 3D array, add a dimension to support grid_sample.
    if len(signal.shape) == 3:
        signal = signal.unsqueeze(0)
    if len(grid.shape) == 3:
        grid = grid.unsqueeze(1)
    
    # Grid_sample expects [N, H, W, 2] instead of [2, N, H, W]
    grid = grid.permute(1, 2, 3, 0)
    
    # Grid sample expects YX instead of XY.
    grid = torch.roll(grid, shifts=1, dims=-1)
    
    return torch.nn.functional.grid_sample(
        signal,
        grid,
        padding_mode='zeros',
        align_corners=True,
        mode="bilinear"
    )

def trilinear_interpolation(signal, grid):
    """ 
    
    @param signal: Tensor containing pixel values [C, D, H, W] or [N, C, D, H, W]
    @param grid: Tensor containing coordinate values [3, D, H, W] or [3, N, D, H, W]
    """
    # If signal or grid is a 4D array, add a dimension to support grid_sample.
    if len(signal.shape) == 4:
        signal = signal.unsqueeze(0)
    if len(grid.shape) == 4:
        grid = grid.unsqueeze(1)

    # Grid_sample expects [N, D, H, W, 3] instead of [3, N, D, H, W]
    grid = grid.permute(1, 2, 3, 4, 0)
    
    # Grid sample expects YX instead of XY.
    grid = torch.roll(grid, shifts=1, dims=-1)
    
    return torch.nn.functional.grid_sample(
        signal, 
        grid,
        padding_mode='zeros',
        align_corners=True,
        mode="bilinear" # actually trilinear in this case...
    )


class LiftingKernelBase(torch.nn.Module):
    
    def __init__(self, group, kernel_size, in_channels, out_channels):
        """ Implements a base class for the lifting kernel. Stores the R^2 grid
        over which the lifting kernel is defined and it's transformed copies
        under the action of a group H.
        
        """
        super().__init__()
        self.group = group

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Create spatial kernel grid. These are the coordinates on which our
        # kernel weights are defined.
        self.register_buffer("grid_R2", torch.stack(torch.meshgrid(
            torch.linspace(-1., 1., self.kernel_size),
            torch.linspace(-1., 1., self.kernel_size)
            #indexing='ij'
        )).to(self.group.identity.device))

        # Transform the grid by the elements in this group.
        self.register_buffer("transformed_grid_R2", self.create_transformed_grid_R2())

    def create_transformed_grid_R2(self):
        """Transform the created grid by the group action of each group element.
        This yields a grid (over H) of spatial grids (over R2). In other words,
        a list of grids, each index of which is the original spatial grid transformed by
        a corresponding group element in H.
        
        """
        # Obtain all group elements.

        ## YOUR CODE STARTS HERE ##
        group_elements = self.group.elements()
        ## AND ENDS HERE ##

        # Transform the grid defined over R2 with the sampled group elements.
        # Recall how the left-regular representation acts on the domain of a 
        # function on R2! (Hint: look closely at the equation given under 1.3)
        # We'd like to end up with a grid of shape [2, |H|, kernel_size, kernel_size].

        ## YOUR CODE STARTS HERE ##
        transformed_grids = []
        for element in self.group.inverse(group_elements):
            transformed_grids.append(
                self.group.left_action_on_R2(element, self.grid_R2)
            )

        transformed_grid = torch.stack(transformed_grids, dim=1)
        ## AND ENDS HERE ##

        return transformed_grid


    def sample(self, sampled_group_elements):
        """ Sample convolution kernels for a given number of group elements

        arguments should include:
        :param sampled_group_elements: the group elements over which to sample
            the convolution kernels

        should return:
        :return kernels: filter bank extending over all input channels, 
            containing kernels transformed for all output group elements.
        """
        raise NotImplementedError()


class InterpolativeLiftingKernel(LiftingKernelBase):

    def __init__(self, group, kernel_size, in_channels, out_channels):
        super().__init__(group, kernel_size, in_channels, out_channels)

        # Create and initialise a set of weights, we will interpolate these
        # to create our transformed spatial kernels.
        self.weight = torch.nn.Parameter(torch.zeros((
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size
        ), device=self.group.identity.device))

        # Initialize weights using kaiming uniform intialisation.
        torch.nn.init.kaiming_uniform_(self.weight.data, a=math.sqrt(5))
    
    def sample(self):
        """ Sample convolution kernels for a given number of group elements

        should return:
        :return kernels: filter bank extending over all input channels, 
            containing kernels transformed for all output group elements.
        """
        # First, we fold the output channel dim into the input channel dim; 
        # this allows us to transform the entire filter bank in one go using the
        # torch grid_sample function.

        ## YOUR CODE STARTS HERE ##
        weight = self.weight.view(
            self.out_channels * self.in_channels,
            self.kernel_size,
            self.kernel_size
        )
        ## AND ENDS HERE ##

        # Sample the transformed kernels.
        transformed_weight = []
        for spatial_grid_idx in range(len(self.group.elements())):
            transformed_weight.append(
                bilinear_interpolation(weight, self.transformed_grid_R2[:, spatial_grid_idx, :, :])
            )
        transformed_weight = torch.stack(transformed_weight)
            
        # Separate input and output channels.
        transformed_weight = transformed_weight.view(
            len(self.group.elements()),
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size
        )
        
        # Put out channel dimension before group dimension. We do this
        # to be able to use pytorched Conv2D. Details below!
        transformed_weight = transformed_weight.transpose(0, 1)

        return transformed_weight


class LiftingConvolution(torch.nn.Module):

    def __init__(self, group, in_channels, out_channels, kernel_size, padding):
        super().__init__()

        self.kernel = InterpolativeLiftingKernel(
            group=group,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels
        )
        
        self.padding = padding

    def forward(self, x):
        """ Perform lifting convolution

        @param x: Input sample [batch_dim, in_channels, spatial_dim_1, 
            spatial_dim_2]
        @return: Function on a homogeneous space of the group
            [batch_dim, out_channels, num_group_elements, spatial_dim_1, 
            spatial_dim_2]
        """

        # Obtain convolution kernels transformed under the group.
        
        ## YOUR CODE STARTS HERE ##
        conv_kernels = self.kernel.sample()
        ## AND ENDS HERE ##

        # Apply lifting convolution. Note that using a reshape we can fold the
        # group dimension of the kernel into the output channel dimension. We 
        # treat every transformed kernel as an additional output channel. This
        # way we can use pytorch's conv2d function!

        # Question: Do you see why we (can) do this?

        ## YOUR CODE STARTS HERE ##
        x = torch.nn.functional.conv2d(
            input=x,
            weight=conv_kernels.reshape(
                self.kernel.out_channels * len(self.kernel.group.elements()),#numel
                self.kernel.in_channels,
                self.kernel.kernel_size,
                self.kernel.kernel_size
            ),
            padding=self.padding
        )
        ## AND ENDS HERE ##

        # Reshape [batch_dim, in_channels * num_group_elements, spatial_dim_1, 
        # spatial_dim_2] into [batch_dim, in_channels, num_group_elements, 
        # spatial_dim_1, spatial_dim_2], separating channel and group 
        # dimensions.
        x = x.view(
            -1,
            self.kernel.out_channels,
            len(self.kernel.group.elements()),
            x.shape[-1],
            x.shape[-2]
        )

        return x

class GroupKernelBase(torch.nn.Module):

    def __init__(self, group, kernel_size, in_channels, out_channels):
        """ Implements base class for the group convolution kernel. Stores grid
        defined over the group R^2 \rtimes H and it's transformed copies under
        all elements of the group H.
        
        """
        super().__init__()
        self.group = group

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Create a spatial kernel grid
        self.register_buffer("grid_R2", torch.stack(torch.meshgrid(
            torch.linspace(-1., 1., self.kernel_size),
            torch.linspace(-1., 1., self.kernel_size)#,indexing='ij'
        )).to(self.group.identity.device))

        # The kernel grid now also extends over the group H, as our input 
        # feature maps contain an additional group dimension
        self.register_buffer("grid_H", self.group.elements())
        self.register_buffer("transformed_grid_R2xH", self.create_transformed_grid_R2xH())

    def create_transformed_grid_R2xH(self):
        """Transform the created grid over R^2 \rtimes H by the group action of 
        each group element in H.
        
        This yields a set of grids over the group. In other words, a list of 
        grids, each index of which is the original grid over G transformed by
        a corresponding group element in H.
        """
        # Sample the group H.
        
        ## YOUR CODE STARTS HERE ##
        group_elements = self.group.elements()
        ## AND ENDS HERE ##

        # Transform the grid defined over R2 with the sampled group elements.
        # We again would like to end up with a grid of shape [2, |H|, kernel_size, kernel_size].
        
        ## YOUR CODE STARTS HERE ##
        transformed_grid_R2 = []
        for g_inverse in self.group.inverse(group_elements):
            transformed_grid_R2.append(
                self.group.left_action_on_R2(g_inverse, self.grid_R2)
            )
        
        transformed_grid_R2 = torch.stack(transformed_grid_R2, dim=1)
        ## AND ENDS HERE ##

        # Transform the grid defined over H with the sampled group elements. We want a grid of 
        # shape [|H|, |H|]. Make sure to stack the transformed like above (over the 1st dim).

        ## YOUR CODE STARTS HERE ##
        transformed_grid_H = []
        for g_inverse in self.group.inverse(group_elements):
            transformed_grid_H.append(
                self.group.product(
                    g_inverse, self.grid_H
                )
            )
        transformed_grid_H = torch.stack(transformed_grid_H, dim=1)
        ## AND ENDS HERE ##

        # Rescale values to between -1 and 1, we do this to please the torch
        # grid_sample function.
        transformed_grid_H = self.group.normalize_group_elements(transformed_grid_H)
        # Create a combined grid as the product of the grids over R2 and H
        # repeat R2 along the group dimension, and repeat H along the spatial dimension
        # to create a [3, |H|, |H|, kernel_size, kernel_size] grid
        transformed_grid = torch.cat(
            (
                transformed_grid_R2.view(
                    2,
                    len(group_elements),##group_elements.numel() initially
                    1,
                    self.kernel_size,
                    self.kernel_size,
                ).repeat(1, 1, len(group_elements), 1, 1),##group_elements.numel() initially
                transformed_grid_H.view(
                    1,
                    len(group_elements), ##group_elements.numel() initially
                    len(group_elements),##group_elements.numel() initially
                    1,
                    1,
                ).repeat(1, 1, 1, self.kernel_size, self.kernel_size)
            ),
            dim=0
        )
        return transformed_grid


    def sample(self, sampled_group_elements):
        """ Sample convolution kernels for a given number of group elements

        arguments should include:
        :param sampled_group_elements: the group elements over which to sample
            the convolution kernels

        should return:
        :return kernels: filter bank extending over all input channels, 
            containing kernels transformed for all output group elements.
        """
        raise NotImplementedError()


class InterpolativeGroupKernel(GroupKernelBase):

    def __init__(self, group, kernel_size, in_channels, out_channels):
        super().__init__(group, kernel_size, in_channels, out_channels)

        # create and initialise a set of weights, we will interpolate these
        # to create our transformed spatial kernels. Note that our weight
        # now also extends over the group H.

        ## YOUR CODE STARTS HERE ##
        self.weight = torch.nn.Parameter(torch.zeros((
            self.out_channels,
            self.in_channels,
            len(self.group.elements()), # this is different from the lifting convolution, used to be numel()
            self.kernel_size,
            self.kernel_size
        ), device=self.group.identity.device))
        ## AND ENDS HERE ##

        # initialize weights using kaiming uniform intialisation.
        torch.nn.init.kaiming_uniform_(self.weight.data, a=math.sqrt(5))
    
    def sample(self):
        """ Sample convolution kernels for a given number of group elements

        should return:
        :return kernels: filter bank extending over all input channels, 
            containing kernels transformed for all output group elements.
        """
        # First, we fold the output channel dim into the input channel dim; 
        # this allows us to transform the entire filter bank in one go using the
        # interpolation function.
       
        ## YOUR CODE STARTS HERE ##
        weight = self.weight.view(
            self.out_channels * self.in_channels,
            len(self.group.elements()),  #numel
            self.kernel_size,
            self.kernel_size
        )
        ## AND ENDS HERE ## 
        
        transformed_weight = []
        # We loop over all group elements and retrieve weight values for
        # the corresponding transformed grids over R2xH.
        for grid_idx in range(len(self.group.elements())):
            transformed_weight.append(
                trilinear_interpolation(weight, self.transformed_grid_R2xH[:, grid_idx, :, :, :])
            )
        transformed_weight = torch.stack(transformed_weight)
        
        # Separate input and output channels.
        transformed_weight = transformed_weight.view(
            len(self.group.elements()), #numel
            self.out_channels,
            self.in_channels,
            len(self.group.elements()), #numel
            self.kernel_size,
            self.kernel_size
        )

        # Put out channel dimension before group dimension. We do this
        # to be able to use pytorched Conv2D. Details below!
        transformed_weight = transformed_weight.transpose(0, 1)
        
        return transformed_weight


class GroupConvolution(torch.nn.Module):

    def __init__(self, group, in_channels, out_channels, kernel_size, padding):
        super().__init__()

        self.kernel = InterpolativeGroupKernel(
            group=group,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels
        )
        
        self.padding = padding
        

    def forward(self, x):
        """ Perform lifting convolution

        @param x: Input sample [batch_dim, in_channels, group_dim, spatial_dim_1, 
            spatial_dim_2]
        @return: Function on a homogeneous space of the group
            [batch_dim, out_channels, num_group_elements, spatial_dim_1, 
            spatial_dim_2]
        """

        # We now fold the group dimensions of our input into the input channel
        # dimension.

        ## YOUR CODE STARTS HERE ##
        x = x.reshape(
            -1,
            x.shape[1] * x.shape[2],
            x.shape[3],
            x.shape[4]
        )
        ## AND ENDS HERE ##

        # We obtain convolution kernels transformed under the group.

        ## YOUR CODE STARTS HERE ##
        conv_kernels = self.kernel.sample()
        ## AND ENDS HERE ##

        # Apply group convolution, note that the reshape folds the 'output' group 
        # dimension of the kernel into the output channel dimension, and the 
        # 'input' group dimension into the input channel dimension.

        # Question: Do you see why we (can) do this?

        ## YOUR CODE STARTS HERE ##
        x = torch.nn.functional.conv2d(
            input=x,
            weight=conv_kernels.reshape(
                self.kernel.out_channels * len(self.kernel.group.elements()), #numel
                self.kernel.in_channels * len(self.kernel.group.elements()), #numel
                self.kernel.kernel_size,
                self.kernel.kernel_size
            ),
            padding=self.padding
        )
        ## AND ENDS HERE ##

        # Reshape [batch_dim, in_channels * num_group_elements, spatial_dim_1, 
        # spatial_dim_2] into [batch_dim, in_channels, num_group_elements, 
        # spatial_dim_1, spatial_dim_2], separating channel and group 
        # dimensions.
        x = x.view(
            -1,
            self.kernel.out_channels,
            len(self.kernel.group.elements()),#numel
            x.shape[-1],
            x.shape[-2],
        )

        return x


class GroupEquivariantCNN(torch.nn.Module):

    def __init__(self, group, in_channels, out_channels, kernel_size, num_hidden, hidden_channels,output_dimensionality=24):
        super().__init__()
        self.out_channels=out_channels
        self.output_dimensionality = output_dimensionality
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]*num_hidden
        # Create the lifing convolution.

        ## YOUR CODE STARTS HERE ##
        
        self.lifting_conv = LiftingConvolution(
            group=group,
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size.pop(0),
            padding=0
        )
        ## AND ENDS HERE ##

        # Create a set of group convolutions.
        self.gconvs = torch.nn.ModuleList()

        ## YOUR CODE STARTS HERE ##
        
        for ks in kernel_size:
            self.gconvs.append(
                GroupConvolution(
                    group=group,
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=ks,
                    padding=0
                )
            )
        ## AND ENDS HERE ##

        # Create the projection layer. Hint: check the import at the top of
        # this cell.
        
        ## YOUR CODE STARTS HERE ##
        # self.projection_layer = torch.nn.AdaptiveAvgPool3d(1)  # CHANGED THIS
        ## AND ENDS HERE ##

        # And a final linear layer for classification.
        #self.final_linear = torch.nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x):
        
        # Lift and disentangle features in the input.
        # print("input: ", x.shape)
        x = self.lifting_conv(x)
        # print("afer lifting: ", x.shape)
        x = torch.nn.functional.layer_norm(x, x.shape[-4:])
        x = torch.nn.functional.relu(x)

        # Apply group convolutions.
        for i, gconv in enumerate(self.gconvs):
            x = gconv(x)
            # print(f"after gconf {i}: ", x.shape)
            x = torch.nn.functional.layer_norm(x, x.shape[-4:])
            x = torch.nn.functional.relu(x)
        
        # to ensure equivariance, apply max pooling over group and spatial dims.
        # print("before projection: ", x.shape)
        # x = self.projection_layer(x).squeeze(2)
        x = x.mean(dim=-3)  # CHANGED THIS
        # print("after projection: ", x.shape)
        x = torch.tanh(x)  # CHANGED THIS

        #x = self.final_linear(x)
        return x


h_params = {"in_channels": 1,
            "out_channels": 16,
            "kernel_size": 5,
            "num_hidden": 2,
            "hidden_channels":16, # to account for the increase in trainable parameters due to the extra dimension in our feature maps, remove some hidden channels.
            "group":CyclicGroup(order=4)}

def get_gcnn(order=4):
    model = GroupEquivariantCNN(in_channels = 3,
                                out_channels = 64,
                                kernel_size = 5,
                                num_hidden = 20,
                                hidden_channels = 64,
                                group = CyclicGroup(order=4).to("cuda:0")) #E2Group(order=order))
    return model




