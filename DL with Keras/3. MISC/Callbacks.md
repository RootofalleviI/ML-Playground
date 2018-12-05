# Callbacks

## Intro

A **callback** is an object (a class instance implementing specific methods) that is passed to the model in the call to `fit` and that is called by the model at various points during training. It has access to all the available data about the state of the model and its performance, and it can take action: 

- *Model checkpointing* - saving the current weights of the model at different points during training.
- *Early stopping* - interrupting training when the validation loss is no longer improving.
- *Dynamically adjusting the value of certain parameters during training* - e.g., learning rate.
- *Logging training and validation metrics during training, or visualizing the representations learned by the model as they're updated* - the Keras progress bar is a callback itself.

## Built-in callbacks

A list of built-in callbacks can be found in `keras.callbacks`:

- `keras.callbacks.ModelCheckpoint`
- `keras.callbacks.EarlyStopping`
- `keras.callbacks.LearningRateScheduler`
- `keras.callbacks.ReduceLROnPlateau`
- `keras.callbacks.CSVLogger`

### The `ModelCheckpoint` and `EarlyStopping` callbacks

You can use the `EarlyStopping` callback to interrupt training once a target metric being monitored has stopped improving for a fixd number of epochs. This callback is typically used in combination with `ModelCheckpoint`, which lets you continually save the model during training.

```python
import keras

# Define a list of callbacks to pass into the model.fit method
callbacks_list = [

	# Interrupt training when improvement stops
	keras.callbacks.EarlyStopping(
		monitor='acc', # Monitor the accuracy
		patience=1,    # Interrupt training when accuracy has stopped improving for more than one epoch
	),

	# Save the current weights after every epoch
	keras.callbacks.ModelCheckpoint(
		filepath='my_model.h5', # Path to save the model
		monitor='val_loss',     # Only save the model once val_loss has been improved
		save_best_only=True,    # Only save the best
	)
]

model.compile(
	optimizer='rmsprop',
	loss='binary_crossentropy',
	metrics=['acc'] # Your EarlyStopping monitors acc, so it should be part of the model's metrics
)

model.fit(
	x, y,
	epochs=0,
	batch_size=32,
	callbacks=callbacks_list,
	validation_data=(x_val, y_val) # Your ModelCheckpoint needs val_loss so you need to define validation data
)
```

### The `ReduceLROnPlateau` callback

You can use this callback to reduce learning rate when the validation loss has stopped improving. Reducing or increasing the learning rate in the case of a *loss plateau* is an effective strategy to get out of local minima during training.

```python
callbacks_list = [
	keras.callbacks.ReduceLROnPlateau(
		monitor='val_loss', # Monitors the model's validation loss
		factor=0.1, 		# Divides the learning rate by 10 when triggered
		patience=10,		# Trigger the callback after val_loss has stopped improving for 10 epochs
	)
]

model.fit(
	x, y,
	epochs=10,
	batch_size=32,
	callbacks=callbacks_list,
	validation_data=(x_val, y_val) # You need to pass validation data
)
```

## Writing your own callback

Callbacks are implemented by inheriting `keras.callbacks.Callback`. You can implement any number of the following transparently named methods, which are called at various points during training:

- `on_epoch_begin`, `on_epoch_end`
- `on_batch_begin`, `on_batch_end`
- `on_train_begin`, `on_train_end`

These methods are all called with a `logs` argument, which is a dictionary containing information about the previous batch, epoch, or training run -- training and validation metrics, etc. Additionally, the callback has access to the following attributes:

- `self.model` - the model instance from which the callback is being called
- `self.validation_data` - the value of what was passed to `fit` as validation data

### A custom callback

Here is a simple custom callback that saves to disk as Numpy arrays the activations of every layer of the model at the end of every epoch, computed on the first sample of the validation set:

```python
import keras
import numpy as np

class ActivationLogger(keras.callbacks.Callback):

	def set_model(self, model):
		self.model = model
		layer_outputs = [layer.output for layer in model.layers]
		self.activations_model = keras.models.Model(model.input, layer_outputs)

	def on_epoch_end(self, epoch, logs=None):
		if self.validation_data is None:
			raise RuntimeError('Requires validation_data.')
		validation_sample = self.validation_data[0][0:1]
		activation = self.activations_model.predict(validation_sample)
		f = open('activation_at_epoch_' + str(epoch) + '.npz', 'w')
		np.savez(f, activations)
		f.close()
```
