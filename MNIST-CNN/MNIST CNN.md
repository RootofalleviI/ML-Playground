# MNIST: CNN

[TOC]

### Overview of CNN

##### Regular Feed-forward NN vs. CNN

- Feed-forward NNs transform an input by putting it through a series of fully connected hidden layers.
- CNNs have a different architecture:
  1. **3D Layers**: Layers are organized in 3 dimension -- width, height, and depth (a sense of "stack") .
  2. **Not Fully Connected**: The neurons in one layer do not connect to all the neurons in the next layer but only to a small region of it.
  3. **Single Vector of Output**: The final output will be reduced to a single vector of probability scores, organized along the depth dimension.
- CNNs have two components:
  - The *Feature Extraction* Part: the CNN performs a series of **convolutions** and **pooling** operations during which the **features are detected**.
  - The *Classification* Part: the fully connected layers will serve as a **classifier** on top of these extracted features and assign a **probability** for the object on the image being what the algorithm predicts it is.



##### Feature Extraction: Convolution Layer

- The convolution is performed on the input data with the use of a **filter** or **kernel** to then produce a **feature map**.
- The size of the step the convolution filter moves each time is called **stride**. A stride size is usually 1, meaning the filter slides pixel by pixel. By increasing the stride size, your filter is sliding over the input with a larger interval and thus has less overlap between the cells.
- Because the size of the feature map is always smaller than the input, we can add a layer of zero-value pixels surrounding the input to ensure our feature map to not shrink. **Padding** also improves performance and makes sure the kernal and stride size will fit in the input.



##### Feature Extraction: Pooling Layer

- After a convolution layer, we add a **pooling layer** to continuously reduce the dimensionality and in turn reduce the number of parameters and computation in the network. This shortens the training time and controls overfitting.
- The most frequent type of pooling is **max pooling**, which takes the maximum value in each window. These window sizes need to be specified beforehand. This decreases the feature map size while at the same time keeping the significant information.



##### Feature Extraction Summary: Hyperparameters

- The filter size (how large filter windows are)
- The filter count (how many filters we want to use)
- Stride (how big are the steps of the filter)
- Padding (use padding or not)



##### Classification

- The classification part consists of a few fully connected layers. We flatten our 3D data to 1D in order to feed it into the classification layers.
- These fully conected layers are in principle the same as a regular NN.



### CNN for MNIST

##### Architecture

```python
# neural network structure for this sample:
#
# · · · · · · · · · ·      (input data, 1-deep)                 X [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 5x5x1=>4 stride 1        W1 [5, 5, 1, 4]        B1 [4]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                           Y1 [batch, 28, 28, 4]
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x4=>8 stride 2        W2 [5, 5, 4, 8]        B2 [8]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                             Y2 [batch, 14, 14, 8]
#     @ @ @ @ @ @       -- conv. layer 4x4x8=>12 stride 2       W3 [4, 4, 8, 12]       B3 [12]
#     ∶∶∶∶∶∶∶∶∶∶∶                                               Y3 [batch, 7, 7, 12] => reshaped to YY [batch, 7*7*12]
#      \x/x\x\x/        -- fully connected layer (relu)         W4 [7*7*12, 200]       B4 [200]
#       · · · ·                                                 Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)      W5 [200, 10]           B5 [10]
#        · · ·                                                  Y [batch, 10]
```



##### Initialization, Depth, Weights, and Bias

```python
# Same as before
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])
step = tf.placeholder(tf.int32)
```

```python
# Three conv layers with their channel counts, plus a fully connected layer
C1 = 6 # first conv layer output depth
C2 = 12 # second conv layer output depth
C3 = 24 # third conv layer output depth
N = 200 # size of fully connected layer
```

```python
# These are the variables what we want to learn 
W1 = tf.Variable(tf.truncated_normal([6, 6, 1, C1], stddev=0.1))  # 6x6 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.constant(0.1, tf.float32, [C1]))
W2 = tf.Variable(tf.truncated_normal([5, 5, C1, C2], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1, tf.float32, [C2]))
W3 = tf.Variable(tf.truncated_normal([4, 4, C2, C3], stddev=0.1))
B3 = tf.Variable(tf.constant(0.1, tf.float32, [C3]))

W4 = tf.Variable(tf.truncated_normal([7 * 7 * C3, N], stddev=0.1))
B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))
```



##### Construct the Model

```python
stride = 1  # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2  # output is 14x14
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2  # output is 7x7
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# Reshape the output from the third conv layer before feeding into the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * C3])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
YY4 = tf.nn.dropout(Y4, pkeep) # Adding dropout for better performance
Ylogits = tf.matmul(YY4, W5) + B5
Y = tf.nn.softmax(Ylogits)
```

- `SAME` means add padding to preserve the same dimensionality.
- The other option is `VALID`, which means no padding is added.



##### Cross-entropy and Accuracy

```python
# Same as before
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# Same as before
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```



---

> **Credit**: [TensorFlow and deep learning, without a PhD](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist) (Part 11~13)
>
> **Part 1**: 







