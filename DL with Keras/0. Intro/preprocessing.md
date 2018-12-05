# Preprocessing

## Data Preprocessing

**Goal**: make the raw data at hand more amenable to NN.

1. **Vectorization**: Turn input data into tensors before feeding into NN.
2. **Value normalization**: In general, it isn't safe to feed into a NN data that takes relatively large values or data that is heterogeneous. Doing so can trigger large gradient updates that will prevent the network from converging. To make learning easier for the model, your data should have the following characteristics:
  - Take small values: typically, most values should be in the 0~1 range.
  - Be homogeneous: all features should take values in roughly the same range.
  - Additionally, you can normalize each feature independently to have a mean of 0 and standard deviationof 1.
3. **Handling missing values**: In general, it's safe to input missing values as 0 with NN, with the condition that 0 isn't already a meaningful value. The NN will learn from exposure to the data that the value 0 means *missing data* and will start ignoring the value. As a remark, if your training data does not contain missing values, the network won't have learned to ignore them. In this situation, you should artificially generate training samples with missing entries -- copy some training samples several times, and drop some of the features that you expect are likely to misisng in the test data.

---
## Feature Engineering

**Goal**: use your own knowledge about the data and about the ML algorithm at hand to make the algorithm work better by applying hardcoded transformations to the data before it goes into the model.

Before deep learning, feature engineering used to be critical, because classical shallow algorithms didn't have hypothesis spaces rich enough to learn useful features by themselves. Fortunately, modern deep learning removes the need for most feature engineering, because NNs are capable of automatically extracting useful features from raw data. However, you should still keep in mind about feature engineering for two reasons:

1. Good features still allow you to solve problems more elegantly while using fewer resources.
2. Good features let you solve a problem with far less data.



## Feature Learning
