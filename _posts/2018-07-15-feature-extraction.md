---
layout:         post
title:          Feature Extraction with scikit-learn
subtitle:
card-image:     /mldl/assets/images/cards/cat2.gif
date:           2018-07-15 09:00:00
tags:           [machine&nbsp;learning]
categories:     [machine&nbsp;learning]
post-card-type: image
mathjax:        true
---

* <a href="#DictVectorizer">DictVectorizer</a>
* <a href="#FeatureHasher">FeatureHasher</a>
* <a href="#CountVectorizer">CountVectorizer</a>
* <a href="#TfidfVectorizer">TfidfVectorizer</a>
* <a href="#n-grams">n-grams</a>
* <a href="#HashingVectorizer">HashingVectorizer</a>
* <a href="#out-of-core learning">out-of-core learning</a>


The `sklearn.feature_extraction` module can be used to extract features in a format supported by machine learning algorithms from datasets consisting of formats such as text and image.  Feature extraction is very different from Feature selection: the former consists in transforming arbitrary data, such as text or images, into numerical features usable for machine learning. The latter is a machine learning technique applied on these features.

## <a name="DictVectorizer">DictVectorizer</a>

The class `DictVectorizer` can be used to convert feature arrays represented as lists of standard Python dict objects to the NumPy/SciPy representation used by scikit-learn estimators.

While not particularly fast to process, Python’s dict has the advantages of being convenient to use, being sparse (absent features need not be stored) and storing feature names in addition to values.

`DictVectorizer` implements what is called one-of-K or “one-hot” coding for categorical (aka nominal, discrete) features. Categorical features are “attribute-value” pairs where the value is restricted to a list of discrete of possibilities without ordering (e.g. topic identifiers, types of objects, tags, names…).

In the following, “city” is a categorical attribute while “temperature” is a traditional numerical feature:

```python
measurements = [{'city': 'Dubai', 'temperature': 33.}, {'city': 'London', 'temperature': 12.}, {'city': 'San Francisco', 'temperature': 18.},]

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
vec.fit_transform(measurements).toarray() # return: array([[  1.,   0.,   0.,  33.], [  0.,   1.,   0.,  12.], [  0.,   0.,   1.,  18.]])
vec.get_feature_names() # return: ['city=Dubai', 'city=London', 'city=San Francisco', 'temperature']
```

## <a name="FeatureHasher">FeatureHasher</a>

The class `FeatureHasher` is a high-speed, low-memory vectorizer that uses a technique known as feature hashing, or the “hashing trick”. Instead of building a hash table of the features encountered in training, as the vectorizers do, instances of `FeatureHasher` apply a hash function to the features to determine their column index in sample matrices directly. The result is increased speed and reduced memory usage, at the expense of inspectability; the hasher does not remember what the input features looked like and has no `inverse_transform` method.

Since the hash function might cause collisions between (unrelated) features, a signed hash function is used and the sign of the hash value determines the sign of the value stored in the output matrix for a feature. This way, collisions are likely to cancel out rather than accumulate error, and the expected mean of any output feature’s value is zero. This mechanism is enabled by default with `alternate_sign=True` and is particularly useful for small hash table sizes (n_features < 10000). For large hash table sizes, it can be disabled, to allow the output to be passed to estimators like `sklearn.naive_bayes.MultinomialNB` or `sklearn.feature_selection.chi2` feature selectors that expect non-negative inputs.

`FeatureHasher` accepts either mappings (like Python’s dict and its variants in the collections module), (feature, value) pairs, or strings, depending on the constructor parameter `input_type`. Mappings are treated as lists of (feature, value) pairs, while single strings have an implicit value of 1, so ['feat1', 'feat2', 'feat3'] is interpreted as [('feat1', 1), ('feat2', 1), ('feat3', 1)]. If a single feature occurs multiple times in a sample, the associated values will be summed (so ('feat', 2) and ('feat', 3.5) become ('feat', 5.5)). The output from `FeatureHasher` is always a scipy.sparse matrix in the CSR format.

## <a name="CountVectorizer">CountVectorizer</a>

The CountVectorizer provides a simple way to both tokenize a collection of text documents and build a vocabulary of known words, but also to encode new documents using that vocabulary. You can use it as follows:

1. Create an instance of the `CountVectorizer` class.
2. Call the `fit()` function in order to learn a vocabulary from one or more documents.
3. Call the `transform()` function on one or more documents as needed to encode each as a vector.

An encoded vector is returned with a length of the entire vocabulary and an integer count for the number of times each word appeared in the document. Because these vectors will contain a lot of zeros, we call them sparse. Python provides an efficient way of handling sparse vectors in the `scipy.sparse` package. The vectors returned from a call to transform() will be sparse vectors, and you can transform them back to numpy arrays to look and better understand what is going on by calling the `toarray()` function.

Below is an example of using the `CountVectorizer` to tokenize, build a vocabulary, and then encode a document.

```python
from sklearn.feature_extraction.text import CountVectorizer
text = ["The quick brown fox jumped over the lazy dog."]
vectorizer = CountVectorizer()
vectorizer.fit(text)
print(vectorizer.vocabulary_)  # {'dog': 1, 'fox': 2, 'over': 5, 'brown': 0, 'quick': 6, 'the': 7, 'lazy': 4, 'jumped': 3}
vector = vectorizer.transform(text)
print(vector.shape)   # (1, 8)
print(type(vector))   # <class 'scipy.sparse.csr.csr_matrix'>
print(vector.toarray())   # [[1 1 1 1 1 1 1 2]]
```

## <a name="TfidfVectorizer">TfidfVectorizer</a>

In a large text corpus, some words will be very present (e.g. “the”, “a”, “is” in English) hence carrying very little meaningful information about the actual contents of the document. If we were to feed the direct count data directly to a classifier those very frequent terms would shadow the frequencies of rarer yet more interesting terms.

In order to re-weight the count features into floating point values suitable for usage by a classifier it is very common to use the tf–idf transform.

Tf means term-frequency while tf–idf means term-frequency times inverse document-frequency: $$\text{tf-idf(t,d)}=\text{tf(t,d)} \times \text{idf(t)}$$.

Using the `TfidfTransformer`’s default settings, `TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)` the term frequency, the number of times a term occurs in a given document, is multiplied with idf component, which is computed as: $$\text{idf}(t) = log{\frac{1 + n_d}{1+\text{df}(d,t)}} + 1$$, where $$n_d$$ is the total number of documents, and $$\text{df}(d,t)$$ is the number of documents that contain term t. The resulting tf-idf vectors are then normalized by the Euclidean norm: $$v_{norm} = \frac{v}{||v||_2} = \frac{v}{\sqrt{v{_1}^2 + v{_2}^2 + \dots + v{_n}^2}}.$$

```python
from sklearn.feature_extraction.text import TfidfVectorizer
text = ["The quick brown fox jumped over the lazy dog.",
		"The dog.",
		"The fox"]
vectorizer = TfidfVectorizer()
vectorizer.fit(text)
print(vectorizer.vocabulary_)   # {'fox': 2, 'lazy': 4, 'dog': 1, 'quick': 6, 'the': 7, 'over': 5, 'brown': 0, 'jumped': 3}
print(vectorizer.idf_)  # [ 1.69314718 1.28768207 1.28768207 1.69314718 1.69314718 1.69314718 1.69314718 1. ]
vector = vectorizer.transform([text[0]])
print(vector.shape)  # (1, 8)
print(vector.toarray())  # [[ 0.36388646 0.27674503 0.27674503 0.36388646 0.36388646 0.36388646 0.36388646 0.42983441]]
```

## <a name="n-grams">n-grams</a>

A collection of unigrams (what bag of words is) cannot capture phrases and multi-word expressions, effectively disregarding any word order dependence. Additionally, the bag of words model doesn’t account for potential misspellings or word derivations.

N-grams to the rescue! Instead of building a simple collection of unigrams (n=1), one might prefer a collection of bigrams (n=2), where occurrences of pairs of consecutive words are counted.

One might alternatively consider a collection of character n-grams, a representation resilient against misspellings and derivations. We can set the `ngram_range` parameter in `CountVectorizer` or `TfidfVectorizer`.

## <a name="HashingVectorizer">HashingVectorizer</a>

The above vectorization scheme is simple but the fact that it holds an in-memory mapping from the string tokens to the integer feature indices (the vocabulary_ attribute) causes several problems when dealing with large datasets:

* the larger the corpus, the larger the vocabulary will grow and hence the memory use too,
* fitting requires the allocation of intermediate data structures of size proportional to that of the original dataset.
* building the word-mapping requires a full pass over the dataset hence it is not possible to fit text classifiers in a strictly online manner.
* pickling and un-pickling vectorizers with a large `vocabulary_` can be very slow (typically much slower than pickling / un-pickling flat data structures such as a NumPy array of the same size),
* it is not easily possible to split the vectorization work into concurrent sub tasks as the vocabulary_ attribute would have to be a shared state with a fine grained synchronization barrier: the mapping from token string to feature index is dependent on ordering of the first occurrence of each token hence would have to be shared, potentially harming the concurrent workers’ performance to the point of making them slower than the sequential variant.

It is possible to overcome those limitations by combining the “hashing trick” (Feature hashing) implemented by the `sklearn.feature_extraction.FeatureHasher` class and the text preprocessing and tokenization features of the `CountVectorizer`.

This combination is implementing in `HashingVectorizer`, a transformer class that is mostly API compatible with `CountVectorizer`. `HashingVectorizer` is stateless, meaning that you don’t have to call `fit` on it:

```python
from sklearn.feature_extraction.text import HashingVectorizer
text = ["The quick brown fox jumped over the lazy dog."]
vectorizer = HashingVectorizer(n_features=20)
vector = vectorizer.transform(text)
print(vector.shape)   # (1, 20)
print(vector.toarray())  # [[ 0.          0.          0.          0.          0.          0.33333333
                         #     0.         -0.33333333  0.33333333  0.          0.          0.33333333
                         #     0.          0.          0.         -0.33333333  0.          0.
                         #    -0.66666667  0.        ]]
```

The values of the encoded document correspond to normalized word counts by default in the range of -1 to 1, but could be made simple integer counts by changing the default configuration.

## <a name="out-of-core learning">out-of-core learning</a>

An interesting development of using a `HashingVectorizer` is the ability to perform out-of-core scaling. This means that we can learn from data that does not fit into the computer’s main memory.

A strategy to implement out-of-core scaling is to stream data to the estimator in mini-batches. Each mini-batch is vectorized using `HashingVectorizer` so as to guarantee that the input space of the estimator has always the same dimensionality. The amount of memory used at any time is thus bounded by the size of a mini-batch. Although there is no limit to the amount of data that can be ingested using such an approach, from a practical point of view the learning time is often limited by the CPU time one wants to spend on the task.

References:

* [<u>scikit-learn feature extraction doc</u>](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
* [<u><How to Prepare Text Data for Machine Learning with scikit-learn/u>](https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/)
