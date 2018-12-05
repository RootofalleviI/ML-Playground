# Text Processing

## Text vectorization

*Vectorizing* text is the process of transforming text into numeric tensors. This can be done in multiple ways:
1. Segment text into words and transform each word into a vector.
2. Segment text into characters and transform each character into a vector.
3. Extract n-grams -- overlapping groups of multiple consecutive words or characters -- of words or characters and transform each n-gram into a vector. 

The different units into which you can break down text are called **tokens**, and breaking text into such tokens is called **tokenization**.All text-vectorization processes consist of applying some tokenization scheme and then associating numeric vectors with the generated tokens. These vectors, packed into sequence tensors, are fed into deep neural networks. 

## N-grams and Bag-of-words

Word n-grams are groups of $N$ (or fewer) of consecutive words that you can extract from a sentence. The same concept may also be applied to characters instead of words. The term *bag* refers to the fact that you are dealing with a *set* of tokens rather than a list or sequence -- the tokens have no specific order. The family of tokenization methods is called *bag-of-words*.

Because bag-of-words does not preserve order of sequences, it tends to be used in shallow language-processing models rather than DL. Extracting n-grams is a form of feature engineering, and DL tends to replace it with hierarchical feature learning. 1D Convnets and RNNs are capable of learning representations for groups of words and characters without being explicitly told about the existence of such gruops, by looking at continuous word or character sequences.


## One-hot Encoding 

One-hot encoding is the most common way to turn a token into a vector. It consists of associating a unique integer index with every word and then turning this integer index $i$ into a binary vector of size $N$; the vector is all zeros except for the $i$th entry, which is 1.

In Keras:
```python
from keras.preprocessing.text import Tokenizer

samples = ['That cat sat on the mat.', 'The dog ate my homework']

# The tokenizer only takes into account the 1,000 most common words.
tokenizer = Tokenizer(num_words=1000)

# Builds the word index
tokenizer.fit_on_texts(samples) 

# Turn strings into lists of integer indices
sequences = tokenizer.texts_to_sequences(samples)

# Or, you can get the one-hot binary representation directly.
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

# Recover the word index that was computed
print('Found %s unique tokens.' % len(tokenizer.word_index))
```

A variant of one-hot encoding is called *one-hot hashing*, which you can use when the number of unique tokens in your vocabulary is too large to handle explicitly. Instead of assigning an index to each word and keeping a reference of there indices in a dictionary, you can hash words into vectors of fixed size. This is typically done with very lightweight hashing function. 

The main advantage of this method is that it does away with maintaining an explicit word index and thus saves memory and allows online encoding of the data (you can generate token vectors right away before you've seen all the available data). The one drawback is that it's susceptible to *hash collisions* -- two different words may end up with the same hash.

## Word Embeddings

Another popular way to associate a vector with a word is the used of dense *word vectors*, also called *word embeddings*. Whereas the vectors obtained through one-hot encoding are binary, sparse, and very high-dimensional, word embeddings are low-dimensional floating-point vectors (dense as opposed to sparse). Unlike the word vectors obtained via one-hot encoding, word embedding are learned from data.

There are two ways to obtain word embeddings:
1. Learn word embeddings jointly with the main task. In this setup, you start with random word vectors and then learn word vectors in the same way you learn the weights of a NN.
2. Load into your model word embeddings that were precomputed using a different ML task than the one you're trying to solve. This is called *pretrained word embeddings*.

### Learning Word Embeddings with the Embedding Layer

The geometric relationships between word vectors should reflect the ssamantic relationships between these words; word embeddings are meant to map human language into a geometric space. 

In Keras:
```python
from keras.layers import Embedding

# number of possible tokens: 1000, dimensionality of embeddings: 64
embedding_layer = Embedding(1000, 64) 
```

The `Embedding` layer is best understood as a dictionary that maps integer indices (which stand for specific words) to dense vectors. It takes integers as input, looks up these integers in an internal dictionary, and returns the associated vector. It takes as input a 2D tensor of integers, of shape `(samples, sequence_langth)`, where each entry is a sequence of integers. It can embed sequences of variable lengths, but all sequences in a batch must have the same length, thus sequences shorter than others should be padded with zeros and sequences that are longer should be truncated. This layer returns a 3D floating-point tensor of shape `(samples, sequence_length, embedding_dimensionality)`. Such a 3D tensor can be processed by an RNN layer or a 1D convolution layer.

When you instantiate an `Embedding` layer, its weights (its internal dictionary of token vectors) are initially random. During training, tehse word vectors are gradually adjusted via backpropagation, structuring the space into something the downstream model can exploit. Once fully trained, the embedding space will show a lot of structure -- a kind of structure specialized for the specific problem for which you're training your model.

### Pretrained Word Embeddings

Instead of learning word embeddings jointly with the problem you want to solve, you can load embedding vectors ffrom a precomputed embedding space that you know is highly structured and exhibits useful properties -- that captures generic aspects of language strcture. In Keras, we can take advantage of popular models such as Word2vec and GloVe.
