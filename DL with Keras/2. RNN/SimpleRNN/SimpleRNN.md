# RNN

## RNN Overview

A **recurrent neural network** processes sequences by iterating through the sequence elements while maintaining a *state* containing information relative to what it has seen so far. In effect, an RNN is a type of NN that has an internal loop. Its state is reset between processing two different, independent sequences, so you still consider one sequence a single data point: a single input to the network. What changes is that this data point is no longer processed in a single step; rather, the network internally loops over sequence elements. 

### Pseudocode RNN

```python
state_t = 0                           # The state at time t
for input_t in input_sequence:        # Iterate over sequence elements 
    output_t = f(input_t, state_t)    # Calculate the output at time t given input_t
    state_t = output_t                # The previous output becomes the state for the next iteration
```

The function `f` is defined as: `activation(dot(W, input_t) + dot(U, state_t) + b)`.

### Numpy implementation of a simple RNN

```python
# A single forward pass of RNN

import numpy as np

time steps = 100       # Number of timesteps in the input sequence
input_features = 32    # Dim of input feature space
output_features = 64   # Dim of output feature space

inputs = np.random.random((timesteps, input_features))  # Random noise
state_t = np.zeros((output_features,))                  # Initial state: all zeros

W = np.random.random((output_features, input_features)) 
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))

successive_outputs = []
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)     
    successive_outputs.append(output_t)
    state_t = output_t

# Output shape: (timesteps, output_features)
final_output_sequence = np.concatenate(successive_outputs, axis=0) 
```

In summary, an RNN is a `for` loop that reuses quantities computed during the previous iteration of the loop.


## RNN in Keras

The process naively implemented in Numpy corresponds to an actual Keras layer -- the `SimpleRNN` layer. One minor difference is that `SimpleRNN` processes batches of sequences, i.e., it takes inputs of shape `(batch_size, timesteps, input_features)` rather than `(timesteps, features)`.

Like all recurrent layers in Keras, `SimpleRNN` can be run in two different modes, either the full sequences of successive outputs for each timestep (a 3D tensor of shape `(batch_size, timesteps, output_features)`) or only the last output for each input sequence (a 2D tensor of shape `(batch_size, output_features)`). These two modes are controlled by the `return_sequences` ctor arg:

```python
from keras.models import Sequential
from keras.layers import SimpleRNN, Embedding

model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32))
model.summary() 

# The output shape of SimpleRNN would have (None, 32) as we only return the final result

model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.summary()

# The output shape will now have (None, None, 32) as we return a sequence rather than the final output.
```

When you stack many RNN layers to increase the representational power of the network, you need to get all the intermediate layeres to return full sequence of outputs.

## Issue with RNN

Although it should theoretically be able to retain at time `t` information about inputs seen many times before, in practice, such long-term dependencies are impossible to learn due to vainishing gradient. 
