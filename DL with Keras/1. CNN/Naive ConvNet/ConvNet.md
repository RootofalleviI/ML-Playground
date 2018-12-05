# Intro to ConvNets

## Naive ConvNet

### ConvNet Input

```python
from keras import layers, models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
```

A ConvNet takes as input tensors of shape `(image_height, image_width, image_channels)` (not including the batch dimension). In this case, our input size are 28 pixels * 28 pixels 1-channel grayscale images.


### Pooling Layer
 
``` python 
"""
________________________________________________________________
Layer (type) 					Output Shape 		Param #
================================================================
conv2d_1 (Conv2D) 				(None, 26, 26, 32) 	320
________________________________________________________________
maxpooling2d_1 (MaxPooling2D) 	(None, 13, 13, 32) 	0
________________________________________________________________
conv2d_2 (Conv2D) 				(None, 11, 11, 64) 	18496
________________________________________________________________
maxpooling2d_2 (MaxPooling2D) 	(None, 5, 5, 64) 	0
________________________________________________________________
conv2d_3 (Conv2D) 				(None, 3, 3, 64) 	36928
================================================================
Total params: 55,744
Trainable params: 55,744
Non-trainable params: 0
"""
```

The output of every `Conv2D` and `MaxPooling2D` layer is a 3D tensor of shape `(height, width, channels)`. The width and height dimensions tend to shring as you go deeper in the network. The number of channels is controlled by the first argument passed to the `Conv2D` layers (32 or 64).


### Dense Layer

``` python
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activaiton='softmax'))
```

Finally, we feed the last output tensor (of shape `(3, 3, 64)`) into a densely connected classifier network -- a stack of `Dense` layers. Since they only accept 1D vectors, we need to first flatten the 3D outputs to 1D and then add a few `Dense` layers on top.


### Summary

```
Layer (type) 					Output Shape 		Param #
================================================================
conv2d_1 (Conv2D) 				(None, 26, 26, 32) 	320
________________________________________________________________
maxpooling2d_1 (MaxPooling2D) 	(None, 13, 13, 32) 	0
________________________________________________________________
conv2d_2 (Conv2D) 				(None, 11, 11, 64) 	18496
________________________________________________________________
maxpooling2d_2 (MaxPooling2D) 	(None, 5, 5, 64) 	0
________________________________________________________________
conv2d_3 (Conv2D) 				(None, 3, 3, 64) 	36928
________________________________________________________________
flatten_1 (Flatten) 			(None, 576) 		0
________________________________________________________________
dense_1 (Dense) 				(None, 64) 			36928
________________________________________________________________
dense_2 (Dense) 				(None, 10) 			650
================================================================
Total params: 93,322
Trainable params: 93,322
Non-trainable params: 0
```

### Training

```python
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_iamges = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_iamges, train_labels, epochs=5, batch_size=64)
```


## A Deeper Look

### The Convolution Operation

The fundamental difference between a dense layer and a convolution layer is: `Dense` layers learn global patterns in their input feature space, whereas convolution layers learn local patterns. This key characteristics gives convnets two interesting properties:
1. The patterns they learn are translation invariant.
2. They can learn spatial hierarchies of patterns.

Convolutions operate over 3D tensors, called **feature maps**, with two spatial axes (height and width) as well as a depth/channel axis. For an RGB images, `dim(depth) = 3` because there are three color channels: red, green, and blue. For a black-and-white picture, like the MNIST digits, `dim(depth) = 1` because we only have the level of gray.

The convolution operation extracts patches from its input feature map and applies the same transformation to all of these patches, producing an *output feature map*. This output is still a 3D tensor: it has a width and a height. Its depth can be arbitrary, because the output depth is a parameter of the layer, and the different channels in the depth axis now stand for *filters*, which encode special aspects of the input data. 

In the MNIST example, the first conv layer takes a feature map of size `(28, 28, 1)` and outputs a feature map of size `(26, 26, 32)`: it computes 32 filters over its input and each of these 32 output channels contains a $26 \times 26$ grid of values, which is a *response map* of the filter over the input, indicating the response of that filter pattern at different local in the input. That is what *feature map* means: every dimension in the depth axis is a feature/filter, and the 2D tensor `output[:, :, n]` is the 2D spatial `map` of the response of this filter over the input.

Convolutions are defined by two key parameters:
1. Size of the patches extracted from the inputs, e.g., $3 \times 3$.
2. Depth of the output feature map, i.e., the number of filters computed by the convolution, e.g., 32.

A convolution works by *sliding* these windows of size $3 \times 3$ or $5 \times 5$ over the 3D input feature map, stopping at every possible location, and extracting the 3D patch of surrounding features of shape `(window_height, window_width, input_depth)`. Each such 3D patch is then transformed (via a tensor product with the same learned weight matrix, called teh **convolutional kernel**) into a 1D vector of shape `(output_depth,)`. All of these vectors are then spatially reassembled into a 3D output map of shape `(height, width, output_depth)`. Every spatial location in the output feature map corresponds to the same location in the input feature map.

Note that the output width and height may differ from the input width and height due to two reasons:
1. Border effects, which can be countered by padding the input feature map.
	- In Keras `Conv2D` layers, padding is configurable via the `padding` argument, which can either be `valid` (no padding)` or `same` (use padding to preserve the shape).
2. The use of strides, the distance between two successive windows.
	- A *strided convolution* means we configure a stride > 1.


### Max-pooling

Max poolling consists of extracting windows from the input feature maps and ouputting the max value of each channel. It's conceptually similar to a convolution, except that instead of transforming local patches via a learned linear transformation (the convolution kernel), they're transformed via a hardcoded `max` tensor operation. A big difference is that max pooling is usually done with $2 \times 2$ windows and stride 2, in order to downsample teh feature maps by a factor of 2. On the other hand, convoluiton is typically done with $3\times 3$ windows and no stride (stride 1).

Without max pooling, it's hard for the model to learn a spatial hierarchy of features, and the final feature map may be too large and result in expensive computation and overfitting. In short, the reason to use downsampling is to reduce the number of feature-map coefficients to process, as well as to introduce spatial-filter hierarchies by making successive convoluiton layers look at increasingly large windows (in terms of the fraction of the original input they cover).

