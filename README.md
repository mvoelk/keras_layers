# Various Keras Layers that can be used with TensorFlow Eager Execution

## SparseConv2D
Sparse/Partial Convolution layers allow CNNs to process sparse sensory data. To make the convolution operation invariant against sparsity. The sparsity related information is propagate through the network and the layer output is normalized depending on the number of information-carrying elements in the convolution window.

Sparsity Invariant CNNs [arXiv:1708.06500](https://arxiv.org/abs/1708.06500)  
Image Inpainting for Irregular Holes Using Partial Convolutions [arXiv:1804.07723](https://arxiv.org/abs/1804.07723)

## MaxPoolingWithArgmax2D and MaxUnpooling2D
In convolutional encoder-decoder architectures, one may want to invert the max pooling operation without loosing spatial information. This is exactly what these layers do. MaxPoolingWithArgmax2D is a max pooling layer that addidionally outputs the pooling indices and MaxUnpooling2D uses them for unpooling.

SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation [arXiv:1511.00561](http://arxiv.org/abs/1511.00561)

## AddCoords2D
CoordConv adds the spatial information about the location where the convolution kernel is applied as additional features to its input.

An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution [arXiv:1807.03247](https://arxiv.org/abs/1807.03247)
