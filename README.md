# Various Keras Layers that can be used with TensorFlow 2.x

## Conv2D
Standard Convolution layer that comes with some changes and extension.
- bias is disabled by default
- Weight Normalization as an alternative to batch normalization with comparable results

Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks [arXiv:1602.07868](http://arxiv.org/abs/1602.07868)

## SparseConv2D and PartialConv2D
Sparse/Partial Convolution layers allow CNNs to process sparse sensory data by making the convolution operation invariant against the sparsity. The sparsity-related information is propagate through the network and the layer output is normalized depending on the number of information-carrying elements in the convolution window.

The layers come with some extensions compared to the original version proposed in the paper:
- SparseConv2D was changed to be consistent with Conv2D initializers, which leads to better convergence behavior
- sparsity can be propagated as float values instead of binary masks, which leads to better convergence behavior and looks more like a diffusion process
- both layers support Weight Normalization

Sparsity Invariant CNNs [arXiv:1708.06500](https://arxiv.org/abs/1708.06500)  
Image Inpainting for Irregular Holes Using Partial Convolutions [arXiv:1804.07723](https://arxiv.org/abs/1804.07723)

## GroupConv2D
Group Convolution provides CNNs with discreter rotation in- and equivariance by sharing weights over symmetries. Depending on the application, Group Convolution leads to better results and fast convergence. The computation performed in the layer is still slower compared to normal convolution, but the expanded kernel can be loaded into a regular Conv2D layer. (Thanks to Taco Cohen for pointing that out.)

Group Equivariant Convolutional Networks [arXiv:1602.07576](https://arxiv.org/abs/1602.07576)  
Rotation Equivariant CNNs for Digital Pathology [arXiv:1806.03962](https://arxiv.org/abs/1806.03962)

## DeformableConv2D
Deformable Convolution learns input-dependent spacial offsets where the input elements of the convolution are sampled from the input feature map. It can be interpretet as learning an input-dependent receptive field or a dynamic dilation rate. Adding Deformable Convolution usually leads to better object detection and segmentation models.

Deformable Convolutional Networks[arXiv:1703.06211](https://arxiv.org/abs/1703.06211)

## DepthwiseConv2D
Depthwise Convolution layers perform the convolution operation for each feature map separately. Compared to conventional Conv2D layers, they come with significantly fewer parameters and lead to smaller models. A DepthwiseConv2D layer followed by a 1x1 Conv2D layer is equivalent to the SeperableConv2D layer provided by Keras.

Xception: Deep Learning with Depthwise Separable Convolutions [arXiv:1610.02357](https://arxiv.org/abs/1610.02357)

## MaxPoolingWithArgmax2D and MaxUnpooling2D
In convolutional encoder-decoder architectures, one may want to invert the max pooling operation without loosing spatial information. This is exactly what these layers do. MaxPoolingWithArgmax2D is a max pooling layer that addidionally outputs the pooling indices and MaxUnpooling2D uses them for unpooling.

SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation [arXiv:1511.00561](https://arxiv.org/abs/1511.00561)

## AddCoords2D
CoordConv adds the spatial information about the location where the convolution kernel is applied as additional features to its input. A fairly similar approach is known as Semi-convolutional Operators.

An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution [arXiv:1807.03247](https://arxiv.org/abs/1807.03247)  
Semi-convolutional Operators for Instance Segmentation [arXiv:1807.10712](https://arxiv.org/abs/1807.10712)

## Blur2D
Blurring can be used as replacement for MaxPool2D or AvgPool2D. It anti-aliases the feature maps and suppresses high frequencies resulting from downsampling with max pooling.

The layers come with changes compared to the original implementation:
- boundary effects are handled more properly, similar as in AvgPool2D

Why do deep convolutional networks generalize so poorly to small image transformations? [arXiv:1805.12177](https://arxiv.org/abs/1805.12177)  
Making Convolutional Networks Shift-Invariant Again [arXiv:1904.11486](https://arxiv.org/abs/1904.11486)

## LayerNormalization

Layer Normalization is an alternative to Batch Normalization. The statistic used for normalization is calculated over the channel dimension. Compared to Batch Normalization, the results are usually slightly worse, but it can be applied in situations in which it is difficult to apply Batch Normalization.

Layer Normalization [arXiv:1607.06450](http://arxiv.org/abs/1607.06450)

## InstanceNormalization

Instance Normalization is an alternative to Batch Normalization and has interesting properties regarding style transfer. The statistic used for normalization is calculated per instance over the spatial dimensions.

Instance Normalization: The Missing Ingredient for Fast Stylization [arXiv:1607.08022](https://arxiv.org/abs/1607.08022)

## Scale
Learned linear scaling of the features: `scale * x + shift`.
