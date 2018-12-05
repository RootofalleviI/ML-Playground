# Overfitting and Underfitting

The fundamental issue in ML is the tension between optimization and generalization. *Optimizing* refers to the process of adjusting a model to get the best performance possible on the trianing data, while *generalization* refers to how well the trained model performs on data it has never seen before. 

At the beginning of training, optimization and generalization are correlated: the loewr the loss on training data, the lower the loss on test data. While this is happening, the model is said to be *underfit* -- there is still progress to be made as the network hasn't yet modeled all relevant patterns in the training data. But after a certain number of iterations on the training data, generalization stops improving, and validation metrics stall then begin to degrade -- the model is starting to overfit. That is, it's beginning to learn patterns that are specific to the training data but that are misleading or irrelevant when it comes to new data.

To prevent a model from learning misleading or irrelevant pattern found in the training data, the best solution is to *get more training data*. When this isn't possible, the next-best solution is to modulate the quantity of information that youre model is allowed to store or to add constraints on what information it's allowed to store. If a network can only afford to memorize a small number of patterns, the optimization process will force it to focus on the most prominent patterns, which have a better chance of generalizing well. This way of fighting overfitting is called *regularization*.

## Reducing the network's size

The simplest way to prevent overfitting is to reduce the size of the model. Intuitively, a model with more parameters has more *memorization capacity*. However, it's often hard to find the perfect balance between too much capacity and not enough capacity.

## Add weight regularization

A common way to mitigate overfitting is to put constraints on the complexity of a network by forcing its weights to take only small values, which makes the distribution of weight values more *regular*. This is called *weight regularization*, and it is done by adding to the loss function of the network a *cost* associated with having large weights. This cost comes in two favors:

- *L1 regularization* - the cost is proportional to the *absolute value of the weight coefficients*, i.e., the *L1 norm* of the weights.
- *L2 regularization* - the cost is proportional to the *square of the value of the weight coefficients*, i.e., the *L2 norm* of the weights. 

In Keras, you can pass regularizers into a layer the same way you configure the activation function:

``` python
from keras import regularizers

model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer=regularizer.l2(0.001), # l2 with lambda=0.001
          activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, kernel_regularizer=regularizer.l1(0.001), # l1 with lambda=0.001
          activation='relu'))
model.add(layers.Dense(16, kernel_regularizer=regularizer.l1_l2(l1=0.001, l2=0.001), # l1 and l2 simultaneously
          activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

Note that because this penalty is *only added at training time*, the loss for this network will be much higher at training than at test time.

## Adding dropout

Dropout, applied to a layer, consists of randomly *dropping out* (setting to zero) a number of output features of the layer during training. The *dropout rate* is the fraction of the features that are zeroed out; it's usually set between 0.2 and 0.5. At test time, no units are dropped out; instead, the layer's output values are scaled down by a factor equal to the dropout rate, to balance for the fact that more units are active than at training time.

In Keras, you can introduce dropout in a network via the `Dropout` layer, which is applied to the output of the layer *right before* it:

```python
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
```


## Summary

To prevent overfitting in NN:
1. Get more training data.
2. Reduce the capacity of the network.
3. Add weight regularization.
4. Add dropout.
