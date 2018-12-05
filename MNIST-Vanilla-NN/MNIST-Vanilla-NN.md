# MNIST: Vanilla NN

[TOC]

### One-layer Neural Network: Theory

- Handwritten digits in the MNIST dataset are $28 \times 28$ pixel greyscales images. We use them as inputs for a 1-layer neural network.



##### Activation Function: Softmax

$$
\sigma:\R^k \to \left\{\sigma \in \R^k \mid \sigma_i > 0, \sum_{i=1}^K \sigma_i = 1 \right\} \\
\sigma(\mathbf z)_j = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}} \quad\text{for } j = 1, \ldots, k. 
$$

- **Input**: a $K$-dimensional vector $\mathbf z$ of arbitrary real values.
- **Output**: a $K$-dimensional vector $\sigma(\mathbf z)$ of real values, where each entry is in the range $(0, 1)$, and all the entries add up to 1.
- **Purpose**: the output of the softmax function can be used to represent a *categorical distribution* -- that is, a probability distribution over $K$ different possible outcomes.
- **Usage in ML**: multiclass classification, e.g., *multinomial logistic regression* (aka *softmax regression*), *naive Bayes classifiers*, and *ANN*.



##### Training Strategy

$$
\text{softmax}(XW+b) = Y
$$

$$
\begin{bmatrix}
X_{0, 0} & X_{0, 1} & \cdots & X_{0, 783} \\
X_{1, 0} & X_{1, 1} & \cdots & X_{1, 783} \\
\vdots & \vdots & & \vdots\\
X_{99, 0} & X_{99, 1} & \cdots & X_{99, 783}
\end{bmatrix}_{100, 784}

\begin{bmatrix}
W_{0, 0} & W_{0, 1} & \cdots & W_{0, 9} \\
W_{1, 0} & W_{1, 1} & \cdots & W_{1, 9} \\
\vdots & \vdots & & \vdots\\
W_{783, 0} & W_{783, 1} & \cdots & W_{783, 9}
\end{bmatrix}_{784, 10}

+ 
\text{Broadcast}\left\{\begin{bmatrix}
b_0 \\
b_1 \\
\vdots \\
b_9
\end{bmatrix}^T\right\}
= 

\begin{bmatrix}
Y_{0, 0} & Y_{0, 1} & \cdots & Y_{0, 9} \\
Y_{1, 0} & Y_{1, 1} & \cdots & Y_{1, 9} \\
\vdots & \vdots & & \vdots\\
Y_{99, 0} & Y_{99, 1} & \cdots & Y_{99, 9}
\end{bmatrix}_{100, 10}
$$

- $X \in M_{100, 784}$: the input matrix with batch size 100 where each row represents a flattened number consisting of 784 pixels. 
  - $X_{i, j}$ denotes the $j$th pixel in the $i$th image sample, e.g., $X_{100, 5}$ denotes the 5th pixel of the 100th image.
- $W \in M_{784, 10}$: the weight matrix where the pixel at $(i, j)$ represents the weight the $i$th pixel contribute to the number $j$th.
  - As an example, if the 720th pixel strongly implies that the number is a 7 but not a 3, then $W_{719, 7}$ is large and $W_{719, 3}$ is small.
- $b \in \R^{10}$: the bias vector -- 10 neurons means 10 bias constants. 
  - The broadcasting of NumPy helps us add the $\R^{10}$ vector b$$ to each row of the matrix product to get the result matrix $R$.
- $Y \in M_{100, 10}$: outcome, where the $(i, j)$th entry represents the weighted sum reflecting how likely for the $i$th sample to be the number $j$.
  - As an example, if the 32th image is very likely to be a 7 but not a 3, then $Y_{32, 7}$ is large and $Y_{32, 3}$ is small.
- Finally, we apply the softmax activation function to obtain the formula describing the 1-layer neural network, applied to 100 images.



##### Loss Function: Cross-Entropy

- We need a loss function to measure how good our prediction is.

- The cross-entropy function is given by
  $$
  F(Y_i, Y_i') = - \sum Y'_i \log(Y_i)
  $$
  where

  - $Y_i'$ denotes the actual, one-hot encoded probabilities. 

  - $Y_i$ denotes the computed, softmax-plus-bias'ed probabilities.

  - As an example, say we have 
    $$
    \begin{align*}
    \text{Number} &= \begin{bmatrix} 0\phantom{.1} & 1\phantom{.1} & 2\phantom{.1} & 3\phantom{.1} & 4\phantom{.1} & 5\phantom{.1} & 6\phantom{.1} & 7\phantom{.1} & 8\phantom{.1} & 9\phantom{.1} \end{bmatrix}\\
    \text{Computed probability: } Y_i &= \begin{bmatrix}0.1 & 0.2 & 0.1 & 0.3 & 0.2 & 0.1 & 0.9 & 0.2 & 0.1 & 0.1\end{bmatrix} \\
    \text{Actual probability: }Y_i' &= \begin{bmatrix} 0\phantom{.1} & 0\phantom{.1} & 0\phantom{.1} & 0\phantom{.1} & 0\phantom{.1} & 0\phantom{.1} & 1\phantom{.1} & 0\phantom{.1} & 0\phantom{.1} & 0\phantom{.1} \end{bmatrix}
    \end{align*}
    $$
    The actual label of our image sample is 6 and our model predicts the likelyhood of the number being a 6 is 0.9, significantly greater than the probabilities of other choices. Our model did a decent job! But how decent?

- The goal of training is to adjust weights and biases to minimize the loss function, i.e., the cross entropy given the weights, biases, pixels, and known labels of sample images. We use gradient descent for this purpose.

---

### One-layer Neural Network: Code

##### Define TensorFlow Variables and Placeholders

```python
import tensorflow as tf

# Placeholder for training data
X = tf.placeholder(tf.float32, [None, 28, 28, 1])

# Variables our model needs to learn
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
```

- Variables are all the parameters that you want the training algorithm to determine for you. In our case, our weights $W$ and biases $b$.
- Placeholders are parameters that will be filled with actual data during training, in our case, $X$. 
  - Each image has $28 \times 28$ pixels $\times 1$ grayscale value. 
  - The `None` represents the number of images in the mini-batch and will be known at training time.



##### Construct the Model

```python
# Place holder for current labels
Y_ = tf.placeholder(tf.float32, [None, 10])

# Model: Y = softmax(X * W + b)
Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W) + b)

# Loss function we want to minimize: F = - sum(Y' * log(Y))
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# Optimizer: learning rate = 0.003
optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

# Percent of corrent answers found in batch
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_corrent, tf.float32))
```



##### Execute the Model

```python
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    # Load batch of images and correct answers
    batch_X, batch_Y = mnist.train.next_batch(100)
    
    # Assign values to the placeholders
    train_data = {X: batch_X, Y_: batch_Y}
    
    # Train the model
    sess.run(train_step, feed_dict=train_data)    
```

- **Deferred Exection Model**: TensorFlow was built for distributed computing. It has to know what you are going to compute, your execution graph, before it starts actually sending compute tasks to various computers. That is why it has a deferred execution model where you first use TensorFlow functions to create a computation graph in memory, then start an execution `Session` and perform actual computation using `Session.run`. At this point the graph cannot be changed anymore.



##### Check Performance

```python
train_data = {X: batch_X, Y_: batch_Y}
a, c = sess.run([accuracy, cross_entropy], feed_dict=training_data)

test_data = {X: mnist.test.images, Y_: mnist.test.labels}
a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
```

---

### More Layers: 5-layer Neural Network

##### Activation Function: Sigmoid

- We keep **softmax** as the activation function on the last layer because it works best for classification. On intermediate layers however we will use the **sigmoid** activation function.

$$
S(x) = \frac{1}{1+e^{-x}}
$$

- The sigmoid function is monotonic and is constrainted by a pair of horizontal asymptotes as $x \to \pm \infty$.



##### Adding Layers to the Model

```python
# · · · · · · · · · ·          (input data, flattened pixels)       X [batch, 784]   # 784 = 28*28
# \x/x\x/x\x/x\x/x\x/       -- fully connected layer (sigmoid)      W1 [784, 200]      B1[200]
#  · · · · · · · · ·                                                Y1 [batch, 200]
#   \x/x\x/x\x/x\x/         -- fully connected layer (sigmoid)      W2 [200, 100]      B2[100]
#    · · · · · · ·                                                  Y2 [batch, 100]
#     \x/x\x/x\x/           -- fully connected layer (sigmoid)      W3 [100, 60]       B3[60]
#      · · · · ·                                                    Y3 [batch, 60]
#       \x/x\x/             -- fully connected layer (sigmoid)      W4 [60, 30]        B4[30]
#        · · ·                                                      Y4 [batch, 30]
#         \x/               -- fully connected layer (softmax)      W5 [30, 10]        B5[10]
#          ·                                                        Y5 [batch, 10]
```

```python
# We need more variables for weights and biases
W1 = tf.Variable(tf.truncated_normal([784, 200], stddev=0.1)) 
B1 = tf.Variable(tf.zeros([200]))
W2 = tf.Variable(tf.truncated_normal([200, 100], stddev=0.1))
B2 = tf.Variable(tf.zeros([100]))
W3 = tf.Variable(tf.truncated_normal([100, 60], stddev=0.1))
B3 = tf.Variable(tf.zeros([60]))
W4 = tf.Variable(tf.truncated_normal([60, 30], stddev=0.1))
B4 = tf.Variable(tf.zeros([30]))
W5 = tf.Variable(tf.truncated_normal([30, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))

# Now we add extra layers
XX = tf.reshape(X, [-1, 784])
Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1)
Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4)

# See below: Cross-Entropy with Softmax
Ylogits = tf.matmul(Y4, W5) + B5 
Y = tf.nn.softmax(Ylogits)
```

- **Truncated Normal Distribution**: the probability distribution derived from that of a normally distributed random variable by bounding the random variable from either below or both (or both). 
- The `tf.truncated_normal` is a Tensorflow function that produces random values following the Guassian distribution between $-2\sigma$ (`-2 * stddev`) and $+2\sigma$ (`+2 * stddev`). In our case, all weights are initialized with random values between $-0.2$ and $+0.2$.



##### Cross-Entropy with Softmax

- Recall what's happening in our last layer:
  $$
  \text{Weighted Sum Plug Bias} \xrightarrow{\large\text{Softmax Function}} \text{Categorical Distribution} \xrightarrow{\large\text{Cross-Entropy}} \text{Computing Cost}
  $$
  The cross-entropy involves a logarithm, computed on the output of the softmax layer. Since softmax is essentially an exponential, which is never a zero, we should be fine but with 32-bit precision floating-point operations, $\exp(-100)$ is already a genuine zero.

- To solve this problem, we separate the logit (the weighted sum plus bias) and softmax then calculate cross-entropy in a safe way.

```python
# Logit
Ylogits = tf.matmul(Y4, W5) + B5 

# Apply softmax
Y = tf.nn.softmax(Ylogits)

# Calculate cross-entropy
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)

# Bring the test and training cross-entropy to the same scale
cross_entropy = tf.reduce_mean(cross_entropy)*100
```



##### Optimizer: AdamOptimizer

- There may exist many *saddle points* in very high dimensional spaces. They are the points that are not local minima but where the gradient is nevertheless zero, which may cause the gradient descent optimizer to stuck there. One of the optimizer provided by TensorFlow which can solve this problem is `tf.train.AdamOptimizer`.



##### Activation Function: Rectified Linear Unit

- The sigmoid works well for classification because it is steep in the middle and tends to converge quickly to either end (0 or 1) of the graph. However, it suffers from the *vanishing gradient problem* -- towards either end of the sigmoid function, the $Y$ value tend to respond very little to change in $X$, i.e., the gradient at that region is going to be small. As a consequence, the network may refuse to learn further or the learning speed will be drastically slow.

$$
f(x) = x^+ = \max(0, x)
$$

- The **ReLU** activation function is non-linear and is stackable. It also provides us sparsity -- half of the randomly-initialized input, the negative ones, will be casted to zero -- and thereby makes the computations efficient. 
- It does, however, have its own *dying ReLU problem* -- for activations in the horizontal region of ReLU, gradient will be zero as the weights do not get adjusted during descent; if several neurons simply do not respond to changes, they are essentially dead. For this reason, when working with ReLUs, it is best to initialize biases to small positive values so none of them "die" (or worse, remain dead) in the beginning of the training.

```python
W = tf.Variable(tf.truncated_normal([784, 200], stddev=0.1))
B = tf.Variable(tf.ones([200])/10) # Initialize all biases to 0.1
```



##### Learn Rate Decay

```python
step = tf.placeholder(tf.int32)
lr = 0.0001 + tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
```

- We apply exponential decay to the learning rate:

$$
\text{learning rate} = 0.0001 + 0.003 \ast e^{-\frac{\text{step}}{2000}}
$$



##### Dropout

- At each training iteration, you drop random neurons from the network. You choose a probability `pkeep` for a neuron to be kept (usually between $50\%$ and $75\%$) and then at each iteration of the training loop, you randommly remove neurons with all their weights and biases.
- This regularization technique prevents overfitting as it forces a NN to learn more robust features that are useful in conjunction with many different random subsets of the other neurons.
- As a remark, you need to boost the output of the remaining neurons in proportion to make sure activations on the next layer do not shift. When testing the performance on the test data you would put all your neurons back (by setting `pkeep = 1`).
- TensorFlow offers a dropout function to be used on the outputs of a layer of neurons. It randomly zeroes-out some of the outputs and boosts the remaining ones by `1/pkeep`.

```python
# Probability of keeping a node during dropout
# - pkeep=1.0 at test time
# - pkeep=0.75 at training time
pkeep = tf.placeholder(tf.float32)

Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
Y1d = tf.nn.dropout(Y1, pkeep)

Y2 = tf.nn.relu(tf.matmul(Yd, W2) + B2)
Y2d = tf.nn.dropout(Y2, pkeep)
```

---

>  **Credit**: [TensorFlow and deep learning, without a PhD](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist) (Part 1~10)



