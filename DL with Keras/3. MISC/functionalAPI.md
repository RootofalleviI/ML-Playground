# Functional API

## Intro

In the functional API, you directly manipulate tensors, and you use layers as *functions* that take tensors and return tensors:

```python
from keras import Input, layers

input_tensor = Input(shape=(32,))             # A tensor
dense = layers.Dense(32, activation='relu')   # A layer is a function
output_tensor = dense(input_tensor)           # A layer takes a tensor and returns a tensor
```

An example comparing a `Sequential()` to functional API:

```python
from keras.models import Sequential, Model
from keras.import layers
from keras import Input

# Sequential model
seq_model = Sequential()
seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
seq_model.add(layers.Dense(32, activation='relu'))
seq_model.add(layers.Dense(10, activation='softmax'))

# Functional equivalent
input_tensor = Input(shape=(64,))
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)

model = Model(input_tensor, output_tensor)
model.summary()
```

## Multi-input model

The functional API can be used to build models that have multiple inputs. Typically, such models at some point merge their different input branches using a layer that can combine several tensors: by adding them, concatenating them, etc. This is usually done via a Keras merge operation such as `keras.layers.add`, `keras.layers.concatenate`, etc.

Here is a functional API implementation of a two-input question-answering model:

```python
from keras.models import Model
from keras import layers, Input

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

# The text input is a variable-length sequence of integers. Note that we can optionally name the inputs.
text_input = Input(shape=(None,), dtype='int32', name='text')

# Embeds the inputs into a sequence of vectors of size 64
embedded_text = layers.Embedding(64, text_vocabulary_size)(text_input)

# Encodes the vectors in a single vector via an LSTM
encoded_text = layers.LSTM(32)(embedded_text)

# Process the question the same way
question_input = Input(shape=(None,), dtype='int32', name='question')
embedded_text = layers.Embedding(32, question_vocabulary_size)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

# Now concatenate the encoded question and encoded text
concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)

# Adds a softmax classifier on top of it
answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)

# At model instantiation, you specify the two inputs and the output
model = Model([text_input, question_input], answer)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
```

To train the model, we can either feed the model a list of Numpy arrays, or feed it a dictionary that maps input names to Numpy arrays. The latter option is available only if you give names to your inputs.

```python
import numpy as np

num_samples = 1000
max_length = 100

# Generate dummy data
text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))
question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))
answers = np.random.randint(0, 1, size=(num_samples, answer_vocabulary_size)) # One-hot encoded, not integers

# Fitting using a list of inputs
model.fit([text, question], answers, epochs=10, batch_size=128)

# Fitting using a dictionary of inputs
model.fit({'text':text, 'question':question}, answers, epochs=10, batch_size=128)
```

## Multi-output model

In the same way, you can use the functional API to build models with multiple outputs. Here is an example of functional API implementation of a three-output model:

```python
from keras import layers, Input
from keras.models import Model

vocabulary_size = 50000
num_income_groups = 10

posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)

x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)

# We name the outputs
age_prediction = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(num_income_groups, activation='softmax', name='income')(x)
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)

model = Model(posts_input, [age_prediction, income_prediction, gender_prediction])
```

Importantly, training such a model requires the ability to specific different loss functiosn for different outputs of the network. However, because GD requires you to minimize a *scalar*, you must combine these losses into a single value in order to train a model. The simplest way to combine different losses is to sum them all. In Keras, you can use either a list or a dictionary of losses in `compile` to specific different objects for different outputs; the resulting loss values are summed into a global loss, which is minimized during training. 

Note that imbalanced loss contributions will cause the model representations to be optimized preferentially for the task with the largest individual loss, at the expense of the other tasks. To remedy this, you can assign different levels of important to the loss values in their contribution to the final loss. This is useful in particular if the losses' values use different scales. 

```python
# Multiple losses
model.compile(
  optimizer='rmsprop', 
  loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'], 
  loss_weights=[0.25, 1., 10.]
)

# Equivalently, we can use a dictionary
model.compile(
  optimizer='rmsprop', 
  loss={'age':'mse', 'income':'categorical_crossentropy', 'gender':'binary_crossentropy'},
  loss_weights={'age': 0.25, 'income': 1., 'gender': 10.}  
)
```

Now we can feed the input data:

```python
model.fit(posts, [age_targets, income_targets, gender_targets], epochs=10, batch_size=64)
model.fit(posts, {'age':age_targets, 'income':income_targets, 'gender':gender_targets}, epochs=10, batch_size=64)
```

## Directed acyclic graphs of layers

With the functioanl API, you can build NNs in Keras with arbitrary *directed acyclic graphs* of layers. 

### Inception Modules

*Inception* is a popular type of network architecture for CNN. It consists of a stack of modules that themselves look like small independent networks, split into several parallel branches. The most basic form of a Inception modeule has three to four branches starting with 1 x 1 convolution, followed by a 3 x 3 convolution, and ending with the concatenation of the resulting features. This setup helps the network separately learn spatial features and channel-wise features, which is more efficient than learning them jointly. 

**Purpose of 1 x 1 convolutions**: A convolution operating on a 1 x 1 window becomes equivalent to running each tile vector through a `Dense` layer: it will compute features that mix together information from the channels of the input tensor, but it won't mix information across space (because it's looking at one tile at a time). Such 1 x 1 convolutions (aka *pointwise* convolutions) are featured in Inception modules, where they contribute to factoring out channel-wise features learning and space-wise feature learning -- a reasonable thing to do if you assume that each channel is highly autocorrelated across space, but different channels may not be highly correlated with each other.

Here is an example of implementing Inception modules:

```python
from keras import layers

# Every branch has the same stride=2, which is necessary to keep all branch outputs the same size so you can concatenate them.
branch_a = layers.Conv2D(128, 1, activation='relu', strides=2)(x)

# In this branch, the striding occurs in the spatial convolution layer.
branch_b = layers.Conv2D(128, 1, activation='relu')(x)
branch_b = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_b)

# In this branch, the striding occurs in the average pooling layer.
branch_c = layers.AveragePooling2D(3, strides=2)(x)
branch_c = layers.Conv2D(128, 3, activation='relu')(branch_c)

branch_d = layers.Conv2D(128, 1, activation='relu')(x)
branch_d = layers.Conv2D(128, 3, activation='relu')(branch_d)
branch_d = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_d)

# Concatenates the branch outputs to obtain the module output
output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)
```

### Residual Connections

A *residual connections* consists of making the output of an earlier layer available as input to a later layer, effectively creating a shortcut in a sequential model. Rather than being concatenated to the later activation, the earlier output is summed with the layer activation.

### Representational bottlenecks in DL

In a `Sequential` model, each successive representation layer is built on top of the previous one, which means it only has access to information contained in the activation of the previous layer. If one layer is too small, then the model will be constrained by how much information can be crammed into the activations of this layer.

## Layer weight sharing

One more important feature of the functional API is the ability to reuse a layer instsance several times. When you call a layer instance twice, instead of instantiating a new layer for each call, you reuse the same weights with every call. This allows you to build models that have shared branches -- several branches that all share hte same knowledge and perform the same operations. That is, they share the same representations and learn these representations simultaneously for different sets of inputs. 

Consider the problem where you assess the semantic similarity between two sentences. Because of symmetry, you can process both with a single LSTM layer; the representations of this LSTM layer (its weights) are learned baased on both inputs simultaneously.

```python
from keras import layers, Input
from keras.models import Model

# Instantiate one LSTM layer
lstm = layers.LSTM(32)

# Building the left branch: inputs are variable-length sequences of vectors of size 128
left_input = Input(shape=(None, 128))
left_output = lstm(left_input))

# Building the right branch: you reuse its weights
right_input = Input(shape=(None, 128))
right_output = lstm(right_input)

merged = layers.concatenate([left_output, right_output],axis=-1)
predictionar
