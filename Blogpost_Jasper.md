## E(2) Equivariant Vision Transformer

[comment]: <Total blogpost should be a 20 minute read>


### Wouter Bant, Colin Bot, Jasper Eppink, Clio Feng, Floris Six Dijkstra

---
In this blogpost, we dive deeper into E(2) equivariant Vision Transformers and we propose and evaluate alternative methods for the equivariant attention models discussed in ["E(2)-Equivariant Vision Transformer"](https://proceedings.mlr.press/v216/xu23b.html). This paper proposes a new Group-Equivariant Vision Transformer (GE-ViT), which introduces a new positional encoding for the traditional well known Vision Transformer (ViT) ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" ](https://arxiv.org/abs/2010.11929)

In particular, in this blogpost, we present:
1. Evaluate existing and novel methods to make non equivariant (attention) models, equivariant by combining the predictions of different transformations, that reflect inductive biases, of the input.
2. Evaluate a novel method to make modern ViTs (TODO cite google) equivariant by combining equivariant CNNs (TODO cite) to project patches to latent embeddings that will be uses as input to the equivariant vision transformer model used by (TODO cite paper).
3. Visualize different layers of equivariant and non equivariant, with the aim to help researchers better understand these models. 
---

## The Importance of Equivariant Models
In this section we motivate why one should be interested in equivariant models and discuss prior work. Equivariance is a fundamental property in various domains including image processing [Krizhevsky and Ilya 2012](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html), 3D point cloud analysis [Li, Chen and Lee 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_SO-Net_Self-Organizing_Network_CVPR_2018_paper.html), chemistry [Faber et al.](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.117.135502), astronomy [Ntampaka 2016](https://iopscience.iop.org/article/10.3847/0004-637X/831/2/135/meta), and economics [Qin et al. 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/730d61b4d9ff794a028fa3a25b9b891d-Abstract-Conference.html). Equivariance in machine learning refers to the property of a function where applying a transformation to the input results in a corresponding transformation to the output. In simpler terms, if you shift, rotate, or scale the input, the output will shift, rotate, or scale in the same way, making the model's predictions consistent and reliable.


Traditional Convolutional Neural Networks (CNNs) exhibit translation equivariance but lack equivariance to rotations in input data. [Cohen and Welling 2016](https://proceedings.mlr.press/v48/cohenc16.html) introduced the first equivariant neural network, which augmented the existing translation equivariance of CNNs by incorporating translation to discrete groups.
In the realm of Natural Language Processing (NLP), [Vaswani et al. 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) introduced transformers, a model that gained significant prominence in its field. Recognizing the potential of this architecture in computer vision, [dosovitskiy et al. 2020](https://arxiv.org/abs/2010.11929) proposed the original vision transformer architecture. However, a limitation of their approach is necessitating positional encodings for each pixel patch, losing the translation or any other form of equivariance. Despite this drawback, vision transformers demonstrated noteworthy performance, achieving state-of-the-art results in various domains. To allow ViTs to be equivariant to affine groups, new positional encodings need to be proposed to replace the original positional encodings. 

Iniitial attempts have been made to modify the self-attention to become equivariant. Before the release of the GE-ViT model, The most promising work in the field was proposed by [Romero et al. 2020](https://proceedings.mlr.press/v119/romero20a.html). They proposed Group Equivariant Stand Alone Self-Attention Networks (GSA-nets), which incorporated a different positional encoding strategy and modifications to the attention mechanism to ensure equivariance.

<table align="center">
  <tr align="center">
      <td><img src="figures/rotation.gif" width=600></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure 1.</b> The effect of rotation on the predicted digit for the GE-ViT and the standard ViT</td>
  </tr>
</table>

In Figure 1, the significance of equivariance becomes evident. Consistent outcomes in the above digit prediction is desirable, highlighting the importance of a model that maintains its predictions irrespective of image rotation. This attribute holds particular significance in fields such as cell analysis, where a model's ability to deliver consistent predictions regardless of image orientation is crucial.

--- 

## Recap on Vision Tranformers (ViTs)

[comment]: <In this section we discuss modern ViTs and older equivariant versions.>

In recent years, The Transformer architecture ["Attention is all you need"](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) has had a huge impact in natural language processing (NLP). The succes of this architecture has paved the way for an adaptation in computer vision, giving rise to Vision Transformers (ViTs).

The transformer architecture works by having an encoder-decoder structure. The encoder maps an input sequence $(x_1, ..., x_n)$ to a continous latent variable denoted as $z=(z_1, ...,z_n)$. Using this latent variable $z$, the decoder generates an output sequence $y=(y_1, ..., y_m)$ one element at a time. During each generating time step, the model utilises its previously generated output. Within this encoder and decoder structure, the architectures uses self-attention and fully connected layers. The attention mechanism will be explained in the next paragraph. The full architecture is depicted in Figure 2.

<table align="center">
  <tr align="center">
      <td><img src="figures/Transformer architecture.png" width=300></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure 2.</b> The Transformer architecture with the encoder (left) and decoder (right) (INSERT BRON)</td>
  </tr>
</table>

The encoder works by having a stack of $N$ blocks layers on top of each other. Each layer contains two sub-layers. The first sub-layer is a multi-head self-attention mechanism and the second sub-layer a fully connected feed-forward network. Between each sublayer, residual connections are used.

The decoder is very similar to the encoder however this encapsulates a third sub-layer. This third sub-layer is responsible for the Multi-Head attention on the output generated so far in the process.

<strong> Attention: </strong>

The attention mechanism allows the transformer to assign a certain weight to the individual input tokens. Using this attention results in improved embeddings. A toy example of this can be found in Figure 3. In this graph words with higher attention have a higher opacity when encoding the word it.

<table align="center">
  <tr align="center">
      <td><img src="figures/Attention example.png" width=300></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure 3.</b> An example of attention (INSERT BRON)</td>
  </tr>
</table>

More formally, the attention mechanism can be seen as mapping a query and a set of key-value pairs to an output. The query, key and value are all vectors. The output is computed by taking the weighted sum of the values. The weight for each value is determined by a multiplication between the query and the key and taking softmax. This procedure is called Scaled Dot-Product attention. In the final architecture Multi-Head Attention however is used which consists of several attention layers which are computed in parallel. In practice the attention for a set of queries are calculated simultaneously. This is done by packing together all queries in a matrix Q, all keys in matrix K and all values in matrix V. In Figure 4, a schematic depiction of the attention mechanism is show.

<table align="center">
  <tr align="center">
      <td><img src="figures/Multi-head attention.png" width=500></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure 4.</b> Schematic depiction of the attention mechanism (INSERT BRON)</td>
  </tr>
</table>

<strong> Positional Encoding: </strong>

During the encoding block, the order of the input is not taken into account. As being able to know the order of the words or input, positional encodings are used. These positional encodings allow the Transformer to understand the order of the input sequence. These positional encodings have the same dimension $d_{\text{model}}$ as the input embeddings, so that the two can be summed. In the original transformer network, the folowing positional encodings were used:

$$\begin{align} 
    PE_{(pos,2i)} &= sin(pos/10000^{2i/d_{\text{model}}}) \\
    PE_{(pos,2i+1)} &= cos(pos/10000^{2i/d_{\text{model}}})
\end{align}$$
In the above equations, $pos$ is the position and $i$ is the dimension. The exact details about the transformer architecture can be found in ["Attention is All you Need"](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)

<strong> Vision Transformer: </strong>

<table align="center">
  <tr align="center">
      <td><img src="figures/ViT architecture.png" width=600></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure 5.</b> Schematic depiction of the ViT architecture (INSERT BRON)</td>
  </tr>
</table>

---
## Using positional encodings to make the model equivariant (Colin)

I (Jasper) propose to insert Colins part about how the positional encodings work here

[comment]: <Here we should display how positional encoding makes the model equivariant.>
---
## Discussion of (TODO cite paper) (Jasper, NOG NIET AF)
Here we say that these methods are comp. expensive and some of our findings. eg steerable but also artifact like differences (show this with a figure). quickly mention we evaluate on validation set a increase batch size (and proportionally learning rate) because of computational constraints. Display the results we got for their methods here and say we use the reported numbers of the best method in the following parts of the blogpost. 


<strong> Weaknesses and strengths of proposed method </strong>

The authors of the original equivaraint vision transformer mention that their proposed positional encodings for the (pixels or patches) results in an E(2) equivariant network. This has as an advantage that this results in consistent results and predictions even for rotated images. This is one very strong strength as such properties are very important in medical cell analysis where you do not want different predictions for the same cells by only rotating an image.

Furthermore, typically one of the advantages of equivaraint networks is that they use weight sharing and generalise faster (Hier nog een bron voor vinden). This is something that the authors didn't focus on, however this advantage is something that we want to look at.

One of the main weaknesses that we seeked to explore with their proposed method are the following. 
1. In the original paper, the author mentions that using the group equivariant vision transformer significantly outperforms non-equivariant self-attention networks. We doubt this claim and believe that retraining a Vision-Transformer and making it rotation equivariant ad-hoc could significantly improve performance. (Note that one downside of this approach is that it is not translation invariant right, can this be changed using e.g. the weird attention they used???)
2. Next up, one of the weaknesses of the original GE-VIT is it training and inference time. In the original paper, the authors mention that they use the well-known ViT, however this is not the case. The original well-known ViT and the original GSA net paper were published in the same journal on the same date. The Vision Transformer they use applies local self-attention for every single pixel which makes the model translation equivariant however also computationally very expensive. This can be made more computationally efficient by dividing the images in patches and apply the equivarent things to those patches. Their proposed architecture uses for MNIST and the lifting layer and rotation of 90 degrees the following input. ([8, 9, 4, 28, 28, 9, 9]) = (batch size, num_heads, the 4 rotations, height, width, patch size, patch size) where patch size refers to the local neighborhood that should be taken into account for attention. Aka for a single pixel it computes attention to 81 different other pixels and then you do this for each 28*28*9*4=28244 pixels making a total of 81*28244=2,286,144 attention computations just for the lifting layer. Having this many attention computations causes training and inference time to be slow for even relatively low resolution images.

3. A third weakness that we discovered has to do with the implementation that the authors used to train and evaluate the performance of their models. When expecting their source code, we found that the authors used the test set during training for evaluating the performance of their method. In the end after training, the epoch with the best performance on the test set was reported as the result. This is not a good practice as this causes overfitting on the test set, while a test set should only be used to predict performance on the final model.
4. The original paper states that their approach is steerable because the positional encoding lives in a continuous space. This however appears to be incorrect because rotations of 45 degrees will get different positional encodings than for 90 degrees. See results in the google docs. This is likely caused by the interpolation effect.

<strong> Our novel contribution </strong>

Some of the contributions that we want to add to this paper are already briefly discussed in the section above as we want to improve upon all the weaknesses mentioned. Furthermore, we also propose a novel architecture which utilizes a combination of a group-equivariant CNN together with a transformer and see if this is able to outperform their baseline

- The ViT we propose uses a patch size (the normal one) of 4x4. So 49 patches in total. Now you compute attention globally with respect to the other patches and yourself. If all other settings are the same we have that you do this for 49*9*4=1764 patches in total, making a total of 49*1764=86,436 attention computations for the lifting layer. This should make it about 26 times faster, so for rotmnist 10 hours/26=23 minutes.

- Maybe do interpretablility analyses but I feel like we should not do that.

[comment]: <How do we want to visualise attention as in the original ViT paper (okey ik ga die paper wel even bestuderen Figure 6 dus kijken)>



---

#### Individual contributions

Below the contributions for each team member are noted

Wouter Bant:


Colin Bot:


Jasper Eppink:
Contributed mainly to the writing of the blogpost. As written the introduction and the weaknesses. Also aided with thinking about the programming challenges and the novelties that we wanted to research. 


Clio Feng:


Floris Six Dijkstra
