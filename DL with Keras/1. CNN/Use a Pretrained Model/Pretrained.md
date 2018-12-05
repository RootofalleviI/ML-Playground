# Using a Pretrained ConvNet

## Pretrained ConvNet

A *pretrained network* is a saved network that was previously trained on a large dataset, typically on a large-scale image classification task. If this original dataset is large enough and general enough, then the spatial hierarchy of features learned by the pretrained network can effectively act as a generic model of the visual word, and hence its features can prove useful for many different computer-vision problems, even though these new problems may involve completely different classes than those of the original task. Such portability of learned features across different problems is a key advantage of deep learning compared to other older, shallow-learning approaches, and it makes deep learning very effective for small-data problems.

There are two ways to use a pretrained network: *feature extraction* and *fine-tuning*.

## Feature Extraction

Feature extraction consists of using the representations learned by a previous network to extract interesting features from new samples. These features are then run through a new classifier, which is trained from scratch.

A ConvNet used for image classificatiom comprise two parts: a *convolutional base* and a dense classifier. We only reuse the convolutional base because it is likely to be more generic and therefore more reusable, but the representations learned by the classifier will necessarily be specific to the set of classes on which the model was trained.

Note that level of generality (and therefore reusability) of the representations extracted by specific convolution layers depends on the depth of the layer in the model. Layers that come earlier in the model extract local, highly generic features maps (such as visual edges, colors, and textures), whereas layers that are higher up extract more-abstract concepts. Thus, if your new dataset differs a lot from the dataset on which the original model was trained, you may be better of using only the first few layers of the model to do feature extraction, rather than using the entire convolutional base.

### Instantiating the VGG16 Convolutional Base

```python
from keras.application import VGG16

conv_base = VGG16(
  weights='imagenet',
  include_top=False,
  input_shape=(150, 150, 3)
)
```

- `weights`: specifies the weight checkpoing from which to initialize the model.
- `include_top=False`: we don't need the dense classifier, just the convolutional base.
- `input_shape`: specifies the shape of the input tensors, optional as the network will figure out by itself if you don't pass this in.

At this point, there are two ways you could proceed:

- Running the convolutional base over your dataset, recording its output to a Numpy array on disk, and then using this data as input to a standalone, densely connected classifier. This solution is fast and cheap but does not support data augmentation.

- Extending the model you have (`conv_base`) by adding `Dense` layers on top, and running the whole thing end to end on the input data. This allows you to use data augmentation but is far more expensive than the first approach -- do not attempt this if you don't have access to a GPU.


### Approach I (CPU): feature extraction without data augmentation

```python
# import appropriate  modules
# define appropriate constants

def extract_features(directory, smaple_count):
  features = np.zeros(shape(shape_count, 4, 4, 512))
  labels = np.zero(shape=(sample_count))
  generator = datagen.flow_from_directory(
    directory,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary'
  )
  i = 0
  for inputs_batch, labels_batch in generator:
    features_batch = conv_base.predict(inputs_batch)
    features[i * batch_size : (i + 1) * batch_size] = feature_batch
    labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size = sample_count:
      break # generators yield data indefinitely in a loop, thus you must break after every image has been seen once.
  return features, labels

# Extract features
train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

# Flatten tensors
train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

# Construct and train the dense classifier
from keras import models, layers, optimizers

model = model.Sequential()
model.add(layers.Dense(256, activaiton='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
  optimizer=optimizers.RMSprop(lr=2e-5),
  loss='binary_crossentropy',
  metrics=['acc']
)

history = model.fit(
  train_features,
  train_labels,
  epochs=30,
  batch_size=20,
  validation_data=(validation_features, validation_labels)
)
```


### Approach II (GPU): feature extraction with data augmentation

```python
from keras import models, layers

# Add dense layers on top of conv_base
model = model.Sequential()
model.add(conv_base) # base layers
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

Before you compile and train the model, it's very important to **freeze** the convoluitonal base. Freezing a layer or a set of layers means preventing their weights from being updated during training. If you don't do this, then the representations that were previously learned by the convolutional base will be modified during training. Because the `Dense` layers on top are randomly initialized, very large weight updates would be propagated through the network, effectively destroying the representations previously learned.

In Keras, you freeze a network by setting its `trainable` attribute to `False`: 

```python
conv_base.trainable = False
```

With this setup, only the weights from the two `Dense` layers that you added will be trained. Now you can start adding data augmentation and train you model:

```python
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

# Train data augmentation
train_datagen = ImageDataGenerator(
  rescale=1./255,
  rotation_range=40,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  fill_mode='nearest'
)

# The validation data shouldn't be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
  train_dir,
  target_size=(150, 150),
  batch_size=20,
  class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
  validation_dir,
  target_size=(150, 150),
  batch_size=20,
  class_mode='binary'
)

model.compile(
  loss='binary_crossentropy',
  optimizer=optimizers.RMSprop(lr=2e-5),
  metrics=['acc']
)

history = model.fit_generator(
  train_generator,
  steps_per_epoch=100,
  epochs=30,
  validation_data=validation_generator,
  validation_steps=50
)
```

## Fine-tuning

Fine-tuning cnosists of unfreezing a few of the top layers of a frozen model base used for feature extraction, and jointly training both the newly added part of the model (in this case, the dense classifier) and these top layers. This is called *fine-tuning* because it slightly adjusts the more abstract representations of the model being reused, in order to make them more relevant for the problem at hand.

Recall that we had to freeze base layers before. For the same reason, it's only possible to fine-tune the top layers of the convolutional base once the classifier on top has already been trained. Otherwise, the error signal propagating through the network during training might destroy the previously-learned weights. Thus, the steps for fine-tuning a network are as follow:

1. Add you custom network on top of an already-trained base network.
2. Freeze the base network and train the new layers.
3. Unfreeze some layers in the base network and jointly train both these layers and the part you added.

The first two steps are already completely during feature extraction section. Now we unfreeze the base layer and free individual layers inside it.

We will start by training the last three layers.
- Why not fine-tune more layers?
  - Earlier layers encode more-generic, reusable features, whereas layers higher up encode more-specialized features. It's more useful to fine-tune the more specialized features, because these are the ones that need to be repurposed on your new problem. There would be fast-decreasing returns in fine-tuning loewr layers.
- Why not fine-tune the entire convolutional base?
  - The more parameters you're training, the more you're at risk of overfitting.

```python
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
  if layer.name == 'block5_conv1':
    set_trainable = True
  if set_trainable:
    layer.trainable = True 
  else:
    layer.trainable = False

model.compile(
  loss='binary_crossentropy',
  optimizer=optimizers.RMSprop(lr=1e-5),
  metrics=['acc']
)

history = model.fit_generator(
  train_generator,
  steps_per_epoch=100,
  epochs=100,
  validation_data=validation_generator,
  validation_steps=50
)
```

Note that, even though loss curve doesn't show any real improvement, the accuracy might still improve, because the loss curve displays the average of pointwise loss values, but what matters for accuracy is the distribution of the loss values, not their average, as accuracy is the result of a binary thresholding of the class probability predicted by the model.
