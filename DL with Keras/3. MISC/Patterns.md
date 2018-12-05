# Patterns

## Batch Normalization

**Normalization** is a broad category of methods that seek to make different samples seen by a ML model more similar to each other, which helps the model learn and generalize well to new data. The method we have used so far makes the assumption that the data follows a Gaussian distribution:

```
normalized_data = (data - np.mean(data, axis=...)) / np.std(data, axis=...)
```

Previously, we normalized data before feeding it into model, but this process should be a concern after every transformation operated by the network. Batch normalization is a type of layer that adaptively normalize data even as teh mean and variance chane over time during training. It works by internally maintaining an exponential moving average of the batch-wise mean and variance of the data seen during training. The main effect of batch normalization is that it helps with gradient propagation and thus allows for deeper networks.

The `BatchNormalization` layer is typically used after a convolutional or densely connected layer:

```python
conv_model.add(layers.Conv2D(32, 3, activation='relu'))
conv_model.add(layers.BatchNormalization())

conv_model.add(layers.Dense(32, activation='relu'))
conv_model.add(layers.BatchNormalization())
```

The layer takes an `axis` argumented that specifies the features axis that should be normalized. This argument defaults to -1, the last axis in the input tensor. 

## Depthwise Separable Convolution

The **depthwise separable convolution** layer performs a spatial convolution on each channel of its input, independently, before minxing output channels via a pointwise convoluation (1 x 1 convolution). This is equivalent to separating the learning of spatial features and learning of channel-wise features, which makes sense if you assume that spatial locations in the input are highly correlated, but different channels are fairely independent. 

This layer requires significantly fewer parameters and involves fewer computations, thus resulting in a smaller, speedier model. And because it's a more representationally efficient way to perform convolution, it tends to learn better representation using less data, resulting in better-performing models.

```python
from keras.models import Sequential, Model
from keras import layers

h = 64
w = 64
c = 3
num_classes = 10

model = Sequential()
model.add(layers.SeparableConv2D(32, 3, activation='relu', input_shape=(h,w,c)))
model.add(layers.SeparableConv2D(64, 3, activation='relu'))
model.add(layers.MaxPooling2D(2))

model.add(layers.SeparableConv2D(64, 3, activation='relu'))
model.add(layers.SeparableConv2D(128, 3, activation='relu'))
model.add(layers.MaxPooling2D(2))

model.add(layers.SeparableConv2D(64, 3, activation='relu'))
model.add(layers.SeparableConv2D(128, 3, activation='relu'))
model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
```
