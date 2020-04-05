# Various Keras Layers that can be used with TensorFlow 2.x

## SparseConv2D and PartialConv2D
Sparse/Partial Convolution layers allow CNNs to process sparse sensory data by making the convolution operation invariant against the sparsity. The sparsity-related information is propagate through the network and the layer output is normalized depending on the number of information-carrying elements in the convolution window.

The layers come with some extensions compared to the original version proposed in the paper:
- SparseConv2D was changed to be consistent with Conv2D initializers, which leads to better convergence behavior
- sparsity can be propagated as float values instead of binary masks, which leads to better convergence behavior and looks more like a diffusion process

Sparsity Invariant CNNs [arXiv:1708.06500](https://arxiv.org/abs/1708.06500)  
Image Inpainting for Irregular Holes Using Partial Convolutions [arXiv:1804.07723](https://arxiv.org/abs/1804.07723)

## DepthwiseConv2D
Depthwise Convolution layers perform the convolution operation for each feature map separately. Compared to conventional Conv2D layers, they come with significantly fewer parameters and lead to smaller models. A DepthwiseConv2D layer followed by a 1x1 Conv2D layer is equivalent to a SeperableConv2D layer provided by Keras.

Xception: Deep Learning with Depthwise Separable Convolutions [arXiv:1610.02357](http://arxiv.org/abs/1610.02357)

## MaxPoolingWithArgmax2D and MaxUnpooling2D
In convolutional encoder-decoder architectures, one may want to invert the max pooling operation without loosing spatial information. This is exactly what these layers do. MaxPoolingWithArgmax2D is a max pooling layer that addidionally outputs the pooling indices and MaxUnpooling2D uses them for unpooling.

SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation [arXiv:1511.00561](http://arxiv.org/abs/1511.00561)

## AddCoords2D
CoordConv adds the spatial information about the location where the convolution kernel is applied as additional features to its input.

An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution [arXiv:1807.03247](https://arxiv.org/abs/1807.03247)
