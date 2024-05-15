## E(2) Equivariant Vision Transformer

[comment]: <Total blogpost should be a 20 minute read>

#### Introduction

[comment]: <Also add one paragraph of related work, should be similar to reviews on OpenReview.net>

[comment]: <Find out how the positional encoding truly is implemented and how this causes the equivariance property>

[comment]: < Mention the difference between E(2) and SE(2) equivarance, answer: SE(2) equivariance also adds reflection equivariance.>

Equivariance, a fundamental property in various domains including image processing [Krizhevsky and Ilya 2012](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html), 3D point cloud analysis [Li, Chen and Lee 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_SO-Net_Self-Organizing_Network_CVPR_2018_paper.html), chemistry [Faber et al.](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.117.135502), astronomy [Ntampaka 2016](https://iopscience.iop.org/article/10.3847/0004-637X/831/2/135/meta), and economics [Qin et al. 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/730d61b4d9ff794a028fa3a25b9b891d-Abstract-Conference.html), has garnered significant attention in the realm of machine learning. Traditional Convolutional Neural Networks (CNNs) exhibit translation equivariance but lack equivariance to rotations in input data. [Cohen and Welling 2016](https://proceedings.mlr.press/v48/cohenc16.html) introduced the first equivariant neural network, which augmented the existing translation equivariance of CNNs by incorporating translation to discrete groups.

In the realm of Natural Language Processing (NLP), [Vaswani et al. 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) introduced transformers, a model that gained significant prominence in its field. Recognizing the potential of this architecture in computer vision, [dosovitskiy et al. 2020](https://arxiv.org/abs/2010.11929) proposed the original vision transformer architecture. However, a limitation of their approach is necessitating positional encodings for each pixel patch, losing the translation or any other form of equivariance. Despite this drawback, vision transformers demonstrated noteworthy performance, achieving state-of-the-art results in various domains. To allow ViTs to be equivariant to affine groups, new positional encodings need to be proposed to replace the original positional encodings. 

Iniitial attempts have been made to modify the self-attention to become equivariant. Some examples include ...... The most promising work in the field is proposed by [Romero et al. 2020](https://proceedings.mlr.press/v119/romero20a.html). They proposed Group Equivariant Stand Alone Self-Attention Networks (GSA-nets), which incorporated a different positional encoding strategy and modifications to the attention mechanism to ensure equivariance.

[comment]: <Dont forget to add the research question> 


#### Weaknesses and strengths of proposed method

The authors of the original equivaraint vision transformer mention that their proposed positional encodings for the (pixels or patches) results in an E(2) equivariant network. This has as an advantage that this results in consistent results and predictions even for rotated images. This is one very strong strength as such properties are very important in medical cell analysis where you do not want different predictions for the same cells by only rotating an image.

Furthermore, typically one of the advantages of equivaraint networks is that they use weight sharing and generalise faster (Hier nog een bron voor vinden). This is something that the authors didn't focus on, however this advantage is something that we want to look at.

One of the main weaknesses that we seeked to explore with their proposed method are the following. 
1. In the original paper, the author mentions that using the group equivariant vision transformer significantly outperforms non-equivariant self-attention networks. We doubt this claim and believe that retraining a Vision-Transformer and making it rotation equivariant ad-hoc could significantly improve performance. (Note that one downside of this approach is that it is not translation invariant right, can this be changed using e.g. the weird attention they used???)
2. Next up, one of the weaknesses of the original GE-VIT is it training and inference time. In the original paper, the authors mention that they use the well-known ViT, however this is not the case. The original well-known ViT and the original GSA net paper were published in the same journal on the same date. The Vision Transformer they use applies local self-attention for every single pixel which makes the model translation equivariant however also computationally very expensive. This can be made more computationally efficient by dividing the images in patches and apply the equivarent things to those patches. Their proposed architecture uses for MNIST and the lifting layer and rotation of 90 degrees the following input. ([8, 9, 4, 28, 28, 9, 9]) = (batch size, num_heads, the 4 rotations, height, width, patch size, patch size) where patch size refers to the local neighborhood that should be taken into account for attention. Aka for a single pixel it computes attention to 81 different other pixels and then you do this for each 28*28*9*4=28244 pixels making a total of 81*28244=2,286,144 attention computations just for the lifting layer. Having this many attention computations causes training and inference time to be slow for even relatively low resolution images.

3. A third weakness that we discovered has to do with the implementation that the authors used to train and evaluate the performance of their models. When expecting their source code, we found that the authors used the test set during training for evaluating the performance of their method. In the end after training, the epoch with the best performance on the test set was reported as the result. This is not a good practice as this causes overfitting on the test set, while a test set should only be used to predict performance on the final model.
4. The original paper states that their approach is steerable because the positional encoding lives in a continuous space. This however appears to be incorrect because rotations of 45 degrees will get different positional encodings than for 90 degrees. See results in the google docs. This is likely caused by the interpolation effect.

#### Our novel contribution

Some of the contributions that we want to add to this paper are already briefly discussed in the section above as we want to improve upon all the weaknesses mentioned. Furthermore, we also propose a novel architecture which utilizes a combination of a group-equivariant CNN together with a transformer and see if this is able to outperform their baseline

- The ViT we propose uses a patch size (the normal one) of 4x4. So 49 patches in total. Now you compute attention globally with respect to the other patches and yourself. If all other settings are the same we have that you do this for 49*9*4=1764 patches in total, making a total of 49*1764=86,436 attention computations for the lifting layer. This should make it about 26 times faster, so for rotmnist 10 hours/26=23 minutes.

- Maybe do interpretablility analyses but I feel like we should not do that.

[comment]: <How do we want to visualise attention as in the original ViT paper (okey ik ga die paper wel even bestuderen Figure 6 dus kijken)>



#### Results

Here I have created tables for the results of our experiments. The results can be divided into 4 sections. In which different ways of training are used


##### Results on Mnist 

All results in the table below were trained on Mnist and evaluated on RotMnist which is Mnist with 90 degree rotations.

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
            <td> </td>
            <td colspan=6, align="center">  Baseline original models </td>
        </tr>
        <tr>
            <td>GSA - Nets</td>
            <td> </td>
            <td> </td>
            <td align="right"> - </td>
            <td align="right"> - </td>
            <td align="right"> - </td>
            <td align="right"> - </td>
        </tr>
        <tr>
            <td>GE-ViT </td>
            <td> </td>
            <td> </td>
            <td align="right"> - </td>
            <td align="right"> - </td>
            <td align="right"> - </td>
            <td align="right"> - </td>
        </tr>
        <tr>
            <td> </td>
            <td colspan=6, align="center">  Our proposed architectures </td>
        </tr>
<tr>
<td>VisionTransformer</td>
<td align="right">29.75</td>
<td align="right">28.724</td>
<td align="right">29.75</td>
<td align="right">28.724</td>
<td align="right">29.75</td>
<td align="right">28.724</td>
</tr>
<tr>
<td>PostHocEquivariantMean</td>
<td align="right">43.5</td>
<td align="right">43.314</td>
<td align="right">84.9</td>
<td align="right">85.496</td>
<td align="right">97.4</td>
<td align="right">97.526</td>
</tr>
<tr>
<td>PostHocEquivariantMax</td>
<td align="right">41.65</td>
<td align="right">41.888</td>
<td align="right">86.75</td>
<td align="right">87.128</td>
<td align="right">97.35</td>
<td align="right">97.28</td>
</tr>
<tr>
<td>PostHocEquivariantSum</td>
<td align="right">43.5</td>
<td align="right">43.314</td>
<td align="right">84.9</td>
<td align="right">85.496</td>
<td align="right">97.4</td>
<td align="right">97.526</td>
</tr>
<tr>
<td>PostHocEquivariantMostProbable</td>
<td align="right">30.05</td>
<td align="right">29.298</td>
<td align="right">10</td>
<td align="right">9.998</td>
<td align="right">10</td>
<td align="right">9.998</td>
</tr>
<tr>
<td>PostHocMostCertain</td>
<td align="right">47.95</td>
<td align="right">48.992</td>
<td align="right">82.95</td>
<td align="right">82.522</td>
<td align="right">96.3</td>
<td align="right">96.26</td>
</tr>
<tr>
<td>PostHocLearnedScoreAggregation</td>
<td align="right">83.8</td>
<td align="right">83.28</td>
<td align="right">87.35</td>
<td align="right">86.996</td>
<td align="right">94.9</td>
<td align="right">94.988</td>
</tr>
<tr>
<td>PostHocLearnedAggregation</td>
<td align="right">92.75</td>
<td align="right">92.426</td>
<td align="right">92.75</td>
<td align="right">92.408</td>
<td align="right">96.65</td>
<td align="right">96.16</td>
</tr>
        <tr>
    </tbody>
</table>

#### Conclusion


#### Individual contributions

Below the contributions for each team member are noted

Wouter Bant:


Colin Bot:


Jasper Eppink:
Contributed mainly to the writing of the blogpost. As written the introduction and the weaknesses. Also aided with thinking about the programming challenges and the novelties that we wanted to research. 


Clio Feng:


Floris Six Dijkstra
