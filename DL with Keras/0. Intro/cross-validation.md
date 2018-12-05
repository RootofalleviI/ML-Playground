# Cross-Validation

## Simple Hold-out Validation

``` Python
num_validation_samples = 10000

# Shuffling the data
np.random.shuffle(data)

# Define the validation set
validation_data = data[:num_validation_samples]
data = data[num_validation_samples:]

# Define the training set
training_data = data[:]

# Construct the model
model = get_model()
model.train(training_data)
validation_score = model.evluate(validation_data)

# At this stage, tune your model,
# retrain it, and tune it again ...

# Train the tuned model on all data
model = get_model()
model.train(np.concatenate([training_data, validation_data]))

test_score = model.evaluate(test_data)
```

This is the simplest evaluation protocol, and it suffers from one flaw: if little data is available, then your validation and test sets may contain too few samples to be statistically representative of the data at hand. To solve this, we introduce **k-fold validation**.

## K-fold Validation

With this approach, you split your data into $K$ partitions of equal size; for each partition $i$, train a model on the remaining $K-1$ partitions, and evaluate it on partition $i$. You final score is then the averages of the $K$ scores obtined. 

``` Python 
k = 4
num_validation_samples = len(data) // k

np.random.shuffle(data)

validation_scores = []
for fold in range(k):

  # Select the validation_data partition
  validation_data = data[num_validation_samples * fold:num_validation_samples * (fold + 1)]
  
  # Use the remainder of the data as training data
  training_data = data[:num_validation_samples * fold] + data[num_validation_samples * (fold + 1):]

  # Create an untrained model instance  
  model = get_model()
  model.train(training_data)
  validation_score = model.evaluate(validation_data)
  validation_scores.append(validation_score)

# Compute the average of the validation scores of the k folds as the final validation score
validation_score = np.average(validation_scores) 

# Finally, train the final model on all non-test data available
model = get_model()
model.train(data)
test_score = model.evaluate(test_data)
```

If you have relatively data available and need to evaluate your model as precisely as possible, you can apply k-fold validation multiple times, shuffling the data every time before splitting it $K$ ways. The final score is the average of the scored obtained at each run of k-fold validation. This is called **iterated k-fold validation with shuffling**. Note that, you end up training and evaluating $P \times K$ models (where $P$ is the number of iterations you use), which can be very expensive.

## Other things to keep in mind

### Data Representativeness
You want both your training set and test set to be representative of the data at hand. For this reason, you usually should randomly shuffle your data before splitting it into training and test sets.

### The Axis of Time
If your data has a sense of "time", e.g., tomorrow's whether, stock movements, etc., you should not randomly shuffle becuase doing so would create a *temporal leak* -- your model will effectively be trained on the data from future. In such situations, you should always make sure all data in your test set if *posterior* to the data in the training set.

### Redundancy in Data
If some data points appear twice, then splitting may result in redundancy between the training and validation sets. In effect, you'll be testing on part of your training data, which is the worst thing you can do. Thus, make sure you training set and validation set are disjoint.
