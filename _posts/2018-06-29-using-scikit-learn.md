---
layout:         post
title:          Using scikit-learn
subtitle:
card-image:     /mldl/assets/images/cards/cat20.gif
date:           2018-06-29 09:00:00
tags:           [machine&nbsp;learning]
categories:     [machine&nbsp;learning, scikit-learn]
post-card-type: image
mathjax:        true
---

* <a href="#Scientific computation tools in Python">Scientific computation tools in Python
* <a href="#Representation and Visualization of Data">Representation and Visualization of Data
* <a href="#Training and Testing Data">Training and Testing Data
* <a href="#Supervised Learning -- Classification">Supervised Learning -- Classification
* <a href="#Scikit-learn's estimator interface">Scikit-learn's estimator interface
* <a href="#Supervised Learning -- Regression">Supervised Learning -- Regression
* <a href="#Unsupervised Learning -- Transformation">Unsupervised Learning -- Transformation
* <a href="#Unsupervised Learning - Clustering">Unsupervised Learning - Clustering
* <a href="#Feature Extraction">Feature Extraction
* <a href="#Text Feature Extraction">Text Feature Extraction
* <a href="#Text classification for SMS spam detection">Text classification for SMS spam detection
* <a href="#Cross Validation">Cross Validation
* <a href="#Model Complexity and GridSearchCV">Model Complexity and GridSearchCV
* <a href="#Pipeline estimators">Pipeline estimators
* <a href="#Performance Metrics, Model Evaluation, and Dealing with Class Imbalances">Performance Metrics, Model Evaluation, and Dealing with Class Imbalances
* <a href="#In Depth - Linear Models">In Depth - Linear Models
* <a href="#In Depth - Support Vector Machines">In Depth - Support Vector Machines
* <a href="#In Depth - Decision Trees and Forests">In Depth - Decision Trees and Forests
* <a href="#Feature Selection">Feature Selection
* <a href="#Unsupervised learning - Hierarchical and density-based clustering algorithms">Unsupervised learning - Hierarchical and density-based clustering algorithms
* <a href="#Unsupervised learning - Non-linear dimensionality reduction (Manifold Learning)">Unsupervised learning - Non-linear dimensionality reduction (Manifold Learning)
* <a href="#Out-of-core Learning Large Scale Text Classification for Sentiment Analysis">Out-of-core Learning Large Scale Text Classification for Sentiment Analysis


## <a name="Scientific computation tools in Python">Scientific computation tools in Python</a>

### Numpy ([<u>Numpy tutorial</u>](http://cs231n.github.io/python-numpy-tutorial/#numpy))

Numpy is the core library for scientific computing in Python. It provides a high-performance multidimensional array object, and tools for working with these arrays.

#### Arrays

A numpy array is a grid of values, all of the same type, and is indexed by a tuple of nonnegative integers. The number of dimensions is the _rank_ of the array; the _shape_ of an array is a tuple of integers giving the size of the array along each dimension.

We can initialize numpy arrays from nested Python lists, and access elements using square brackets:

```python
import numpy as np

a = np.array([1, 2, 3])   # Create a rank 1 array
print(type(a))            # Prints "<class 'numpy.ndarray'>"
print(a.shape)            # Prints "(3,)"
print(a[0], a[1], a[2])   # Prints "1 2 3"
a[0] = 5                  # Change an element of the array
print(a)                  # Prints "[5, 2, 3]"

b = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array
print(b.shape)                     # Prints "(2, 3)"
print(b[0, 0], b[0, 1], b[1, 0])   # Prints "1 2 4"
```

Numpy also provides many functions to create arrays:

```python
a = np.zeros((2,2))   # Create an array of all zeros
print(a)              # Prints "[[ 0.  0.]
                      #          [ 0.  0.]]"

b = np.ones((1,2))    # Create an array of all ones
print(b)              # Prints "[[ 1.  1.]]"

c = np.full((2,2), 7)  # Create a constant array
print(c)               # Prints "[[ 7.  7.]
                       #          [ 7.  7.]]"

d = np.eye(2)         # Create a 2x2 identity matrix
print(d)              # Prints "[[ 1.  0.]
                      #          [ 0.  1.]]"

e = np.random.random((2,2))  # Create an array filled with random values
print(e)                     # Might print "[[ 0.91940167  0.08143941]
                             #               [ 0.68744134  0.87236687]]"
```

#### Array indexing

Numpy offers several ways to index into arrays.

**Slicing**: Similar to Python lists, numpy arrays can be sliced. Since arrays may be multidimensional, you must specify a slice for each dimension of the array:

```python
# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]

# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
print(a[0, 1])   # Prints "2"
b[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
print(a[0, 1])   # Prints "77"
```

You can also mix integer indexing with slice indexing. However, doing so will yield an array of lower rank than the original array.

```python
# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:
row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"

# We can make the same distinction when accessing columns of an array:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)  # Prints "[ 2  6 10] (3,)"
print(col_r2, col_r2.shape)  # Prints "[[ 2]
                             #          [ 6]
                             #          [10]] (3, 1)"
```

**Integer array indexing**: When you index into numpy arrays using slicing, the resulting array view will always be a subarray of the original array. In contrast, integer array indexing allows you to construct arbitrary arrays using the data from another array. Here is an example:

```python
a = np.array([[1,2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and
print(a[[0, 1, 2], [0, 1, 0]])  # Prints "[1 4 5]"

# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))  # Prints "[1 4 5]"

# When using integer array indexing, you can reuse the same
# element from the source array:
print(a[[0, 0], [1, 1]])  # Prints "[2 2]"

# Equivalent to the previous integer array indexing example
print(np.array([a[0, 1], a[0, 1]]))  # Prints "[2 2]"
```

One useful trick with integer array indexing is selecting or mutating one element from each row of a matrix:

```python
# Create a new array from which we will select elements
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

print(a)  # prints "array([[ 1,  2,  3],
          #                [ 4,  5,  6],
          #                [ 7,  8,  9],
          #                [10, 11, 12]])"

# Create an array of indices
b = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print(a[np.arange(4), b])  # Prints "[ 1  6  7 11]"

# Mutate one element from each row of a using the indices in b
a[np.arange(4), b] += 10

print(a)  # prints "array([[11,  2,  3],
          #                [ 4,  5, 16],
          #                [17,  8,  9],
          #                [10, 21, 12]])
```

**Boolean array indexing**: Boolean array indexing lets you pick out arbitrary elements of an array. Frequently this type of indexing is used to select the elements of an array that satisfy some condition. Here is an example:

```python
a = np.array([[1,2], [3, 4], [5, 6]])

bool_idx = (a > 2)   # Find the elements of a that are bigger than 2;
                     # this returns a numpy array of Booleans of the same
                     # shape as a, where each slot of bool_idx tells
                     # whether that element of a is > 2.

print(bool_idx)      # Prints "[[False False]
                     #          [ True  True]
                     #          [ True  True]]"

# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(a[bool_idx])  # Prints "[3 4 5 6]"

# We can do all of the above in a single concise statement:
print(a[a > 2])     # Prints "[3 4 5 6]"
```

#### Datatypes

Every numpy array is a grid of elements of the same type. Numpy provides a large set of numeric datatypes that you can use to construct arrays. Numpy tries to guess a datatype when you create an array, but functions that construct arrays usually also include an optional argument to explicitly specify the datatype. Here is an example:

```python
x = np.array([1, 2])   # Let numpy choose the datatype
print(x.dtype)         # Prints "int64"

x = np.array([1.0, 2.0])   # Let numpy choose the datatype
print(x.dtype)             # Prints "float64"

x = np.array([1, 2], dtype=np.int64)   # Force a particular datatype
print(x.dtype)                         # Prints "int64"
```

#### Array math

Basic mathematical functions operate elementwise on arrays, and are available both as operator overloads and as functions in the numpy module:

```python
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]
print(x + y)
print(np.add(x, y))

# Elementwise difference; both produce the array
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print(x - y)
print(np.subtract(x, y))

# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]
print(x * y)
print(np.multiply(x, y))

# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(x / y)
print(np.divide(x, y))

# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(x))
```

Note that unlike MATLAB, * is elementwise multiplication, not matrix multiplication. We instead use the dot function to compute inner products of vectors, to multiply a vector by a matrix, and to multiply matrices. dot is available both as a function in the numpy module and as an instance method of array objects:

```python
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
print(v.dot(w))
print(np.dot(v, w))

# Matrix / vector product; both produce the rank 1 array [29 67]
print(x.dot(v))
print(np.dot(x, v))

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x, y))
```

Numpy provides many useful functions for performing computations on arrays; one of the most useful is `sum`:

```python
x = np.array([[1,2],[3,4]])

print(np.sum(x))  # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"
```

Apart from computing mathematical functions using arrays, we frequently need to reshape or otherwise manipulate data in arrays. The simplest example of this type of operation is transposing a matrix; to transpose a matrix, simply use the T attribute of an array object:

```python
x = np.array([[1,2], [3,4]])
print(x)    # Prints "[[1 2]
            #          [3 4]]"
print(x.T)  # Prints "[[1 3]
            #          [2 4]]"

# Note that taking the transpose of a rank 1 array does nothing:
v = np.array([1,2,3])
print(v)    # Prints "[1 2 3]"
print(v.T)  # Prints "[1 2 3]"
```

#### Broadcasting (reference: [<u>Broadcasting arrays in Numpy</u>](https://eli.thegreenplace.net/2015/broadcasting-arrays-in-numpy/#id8))

Broadcasting is a powerful mechanism that allows numpy to work with arrays of different shapes when performing arithmetic operations. Frequently we have a smaller array and a larger array, and we want to use the smaller array multiple times to perform some operation on the larger array.

For example, suppose that we want to multiply a constant vector to each row of a matrix. We could do this using an outer loop. Or, we could expand the smaller rank array to have the same rank as dimensions as the larger rank array and then do the multiplication. But a simple, fast, and efficient way of doing this is to use broadcasting.

```python
a = ny.array([
  [0.3, 2.5, 3.5],
  [2.9, 27.5, 0],
  [0.4, 1.3, 23.9],
  [14.4, 6, 2.3]])

b = np.array([9, 4, 4])

result = a * b

print(result) # result is: [[   2.7,   10. ,   14. ],
              #             [  26.1,  110. ,    0. ],
              #             [   3.6,    5.2,   95.6],
              #             [ 129.6,   24. ,    9.2]]
```

Broadcasting is often described as an operation between a "smaller" and a "larger" array. This doesn't necessarily have to be the case, as broadcasting applies also to arrays of the same size, though with different shapes. Therefore, I believe the following definition of broadcasting is the most useful one.

Element-wise operations on arrays are only valid when the arrays' shapes are either equal or compatible. The equal shapes case is trivia. What does "compatible" mean, though?

To determine if two shapes are compatible, Numpy compares their dimensions, starting with the trailing ones and working its way backwards (For the shape (4, 3, 2) the trailing dimension is 2, and working from 2 "backwards" produces: 2, then 3, then 4). If two dimensions are equal, or if one of them equals 1, the comparison continues. Otherwise, you'll see a ValueError raised (saying something like "operands could not be broadcast together with shapes ...").

When one of the shapes runs out of dimensions (because it has less dimensions than the other shape), Numpy will use 1 in the comparison process until the other shape's dimensions run out as well.

Once Numpy determines that two shapes are compatible, the shape of the result is simply the maximum of the two shapes' sizes in each dimension.

Put a little bit more formally, here's a pseudo-algorithm:

```
Inputs: array A with m dimensions; array B with n dimensions
p = max(m, n)
if m < p:
    left-pad A's shape with 1s until it also has p dimensions
else if n < p:
    left-pad B's shape with 1s until is also has p dimensions
result_dims = new list with p elements
for i in p-1 ... 0:
    A_dim_i = A.shape[i]
    B_dim_i = B.shape[i]
    if A_dim_i != 1 and B_dim_i != 1 and A_dim_i != B_dim_i:
        raise ValueError("could not broadcast")
    else:
        result_dims[i] = max(A_dim_i, B_dim_i)
```

The definition above is precise and complete; to get a feel for it, we'll need a few examples.

In the following example, following the broadcasting rules outlined above, the shape (3,) is left-padded with 1 to make comparison with (4, 3) possible:

```
(4, 3)                   (4, 3)
         == padding ==>          == result ==> (4, 3)
(3,)                     (1, 3)
```

Here's another example, broadcasting between a 3-D and a 1-D array:

```
(3,)                       (1, 1, 3)
           == padding ==>             == result ==> (5, 4, 3)
(5, 4, 3)                  (5, 4, 3)
```

Note, however, that only left-padding with 1s is allowed. Therefore:

```
(5,)                       (1, 1, 5)
           == padding ==>             ==> error (5 != 3)
(5, 4, 3)                  (5, 4, 3)
```

Broadcasting is valid between higher-dimensional arrays too:

```
5, 4, 3)                     (1, 5, 4, 3)
              == padding ==>                == result ==> (6, 5, 4, 3)
(6, 5, 4, 3)                  (6, 5, 4, 3)
```

Also, it's perfectly valid to broadcast arrays with the same number of dimensions, as long as they are compatible:

```
(5, 4, 1)
           == no padding needed ==> result ==> (5, 4, 3)
(5, 1, 3)
```

Functions that support broadcasting are known as universal functions. You can find the list of all universal functions in the [<u>documentation</u>](http://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs). Broadcasting typically makes your code more concise and faster, so you should strive to use it where possible.

### Scipy sparse matrices

Sparse matrices are common in machine learning. While they occur naturally in some data collection processes, more often they arise when applying certain data transformation techniques such as One-hot encoding, CountVectorizing, or TfidfVectorizing.

Let’s step back for a second. Just what the heck is a sparse matrix and how is it different than other matrices? Turns out there are two major types of matrices: dense and sparse. Sparse matrices have lots of zero values. Dense matrices do not. Because sparse matrices have lots of zero values, we can apply special algorithms that will do two important things:

1. compress the memory footprint of our matrix object
2. speed up many machine learning routines

Since storing all those zero values is a waste, we can apply data compression techniques to minimize the amount of data we need to store. That is not the only benefit, however. Users of sklearn will note that all native machine learning algorithms require data matrices to be in-memory. Said another way, the machine learning process breaks down when a data matrix (usually called a dataframe) does not fit into RAM. One of the perks of converting a dense data matrix to sparse is that in many cases it is possible to compress it so that it can fit in RAM.

Additionally, consider multiplying a sparse matrix by a dense matrix. Even though the sparse matrix has many zeros, and zero times anything is always zero, the standard approach requires this pointless operation nonetheless. The result is slowed processing time. It is much more efficient to operate only on elements that will return nonzero values. Therefore, any algorithm that applies some basic mathematical computation like multiplication can benefit from a sparse matrix implementation.

Sklearn has many algorithms that accept sparse matrices. The way to know is by checking the fit attribute in the documentation. Look for this: X: {array-like, sparse matrix}.

Scipy has the **Compressed Sparse Row (CSR)** algorithm which converts a dense matrix to a sparse matrix, allowing us to significantly compress our example data.

#### How CSR Works

![sklearn1](/mldl/assets/images/2018-06-29/sklearn1.jpg)

CSR requires three arrays. The first array stores the cumulutive count of nonzero values in all current and previous rows. The second array stores column index values for each nonzero value. And the third array stores all nonzero values. I realize that may be confusing, so let’s walk through an example.

Refer to the diagram above. The first step is to populate the first array. It always starts with 0. Since there are two nonzero values in row 1, we update our array like so [0 2]. There are 2 nonzero values in row 2, so we update our array to [0 2 4]. Doing that for the remaining rows yields [0 2 4 7 9]. By the way, the length of this array should always be the number of rows + 1. Step two is populating the second array of column indices. Note that the columns are zero-indexed. The first value, 1, is in column 0. The second value, 7, is in column 1. The third value, 2, is in column 1. And so on. The result is the array [0 1 1 2 0 2 3 1 3]. Finally, we populate the third array which looks like this [1 7 2 8 5 3 9 6 4]. Again, we are only storing nonzero values.

Believe it or not, these three arrays allow us to perfectly reconstruct the original matrix. From here, common mathematical operations like addition or multiplication can be applied in an efficient manner. Note: the exact details of how the mathematical operations work on sparse matrix implementations are beyond the scope of this post. Suffice it to say there are many wonderful resources online if you are interested.

We can create and manipulate sparse matrices in Scipy as follows:

```python
rnd = np.random.RandomState(seed=321)
X = rnd.uniform(low=0.0, high=1.0, size=(10, 5))
# set the majority of elements to zero
X[X < 0.7] = 0
# turn X into a CSR (Compressed-Sparse-Row) matrix
X_csr = sparse.csr_matrix(X)
print(X_csr)
# Converting the sparse matrix back to a dense array
print(X_csr.toarray())
```

(You may have stumbled upon an alternative method for converting sparse to dense representations: numpy.todense; toarray returns a NumPy array, whereas todense returns a NumPy matrix. In this tutorial, we will be working with NumPy arrays, not matrices; the latter are not supported by scikit-learn.)

The CSR representation can be very efficient for computations, but it is not as good for adding elements. For that, the LIL (List-In-List) representation is better:

```python
# Create an empty LIL matrix and add some items
X_lil = sparse.lil_matrix((5, 5))

for i, j in np.random.randint(0, 5, (15, 2)):
    X_lil[i, j] = i + j

# Converting the lil matrix to a dense array
X_dense = X_lil.toarray()

# Often, once an LIL matrix is created, it is useful to convert it to a CSR format (many scikit-learn algorithms require CSR or CSC format)

X_csr = X_lil.tocsr()
```

## <a name="Representation and Visualization of Data">Representation and Visualization of Data</a>

### Loading the Iris Data with Scikit-learn

scikit-learn embeds a copy of the iris CSV file along with a helper function to load it into numpy arrays:

```python
from sklearn.datasets import load_iris
iris = load_iris()
iris.keys() # returns dict_keys(['target_names', 'target', 'data', 'feature_names', 'DESCR'])
#The features of each sample flower are stored in the data attribute of the dataset:
n_samples, n_features = iris.data.shape # (150, 4)
#The information about the class of each sample is stored in the target attribute of the dataset:
targets = iris.target
#The class names are stored in the the attribute target_names:
names = iris.target_names
```

This data is four dimensional, but we can visualize one or two of the dimensions at a time using a simple histogram or scatter-plot.

```python
%matplotlib inline
import matplotlib.pyplot as plt

x_index = 3
colors = ['blue', 'red', 'green']

for label, color in zip(range(len(iris.target_names)), colors):
    plt.hist(iris.data[iris.target==label, x_index],
             label=iris.target_names[label],
             color=color)

plt.xlabel(iris.feature_names[x_index])
plt.legend(loc='upper right')
plt.show()
```

```python
x_index = 3
y_index = 0

colors = ['blue', 'red', 'green']

for label, color in zip(range(len(iris.target_names)), colors):
    plt.scatter(iris.data[iris.target==label, x_index],
                iris.data[iris.target==label, y_index],
                label=iris.target_names[label],
                c=color)

plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.legend(loc='upper left')
plt.show()
```

Instead of looking at the data one plot at a time, a common tool that analysts use is called the **scatterplot matrix**. Scatterplot matrices show scatter plots between all features in the data set, as well as histograms to show the distribution of each feature.

```python
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
pd.tools.plotting.scatter_matrix(iris_df, figsize=(8, 8))
```

### Loading Digits Data with Scikit-learn

```python
from sklearn.datasets import load_digits
digits = load_digits()
digits.keys() # dict_keys(['target_names', 'target', 'data', 'images', 'DESCR'])
n_samples, n_features = digits.data.shape # (1797, 64)
digits.target.shape # (1797,)
digits.target_names # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
digits.images.shape # (1797, 8, 8)
# We have two versions of the data array: data and images. They're related by a simple reshaping
print(np.all(digits.images.reshape((1797, 64)) == digits.data)) # True
```

Let's visualize the data.

```python
# set up the figure
fig = plt.figure(figsize=(6, 6))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# plot the digits: each image is 8x8 pixels
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')

    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))
```

### Other available data

Scikit-learn makes available a host of datasets for testing learning algorithms. They come in three flavors:

* **Packaged Data**: these small datasets are packaged with the scikit-learn installation, and can be downloaded using the tools in `sklearn.datasets.load_*`
* **Downloadable Data**: these larger datasets are available for download, and scikit-learn includes tools which streamline this process. These tools can be found in `sklearn.datasets.fetch_*`
* **Generated Data**: there are several datasets which are generated from models based on a random seed. These are available in the `sklearn.datasets.make_*`

## <a name="Training and Testing Data">Training and Testing Data</a>

Now we need to split the data into training and testing. Luckily, this is a common pattern in machine learning and scikit-learn has a pre-built function to split data into training and testing sets for you.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=321)
```

### Stratified Split

Especially for relatively small datasets, it's better to stratify the split. Stratification means that we maintain the original class proportion of the dataset in the test and training sets. For example, after we randomly split the dataset as shown in the previous code example, we have the following class proportions in percent:

```python
print('All:', np.bincount(y) / float(len(y)) * 100.0)
print('Training:', np.bincount(y_train) / float(len(y_train)) * 100.0)
print('Test:', np.bincount(y_test) / float(len(y_test)) * 100.0)
```

So, in order to stratify the split, we can pass the label array as an additional option to the `train_test_split` function:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=321, stratify=y)
```

Instead of using the same dataset for training and testing (this is called "resubstitution evaluation"), it is much much better to use a train/test split in order to estimate how well your trained model is doing on new data.

```python
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.prefict(X_test)

print("Accuracy:")
print(np.sum(y_pred == y_test) / float(len(y_test)))
```

We can also visualize the correct and failed predictions

```python
print('Samples correctly classified:')
correct_idx = np.where(y_pred == y_test)[0]
print(correct_idx)

print('\nSamples incorrectly classified:')
incorrect_idx = np.where(y_pred != y_test)[0]
print(incorrect_idx)
```

We can plot and visualize two dimensions

```python
colors = ["darkblue", "darkgreen", "gray"]

for n, color in enumerate(colors):
    idx = np.where(y_test == n)[0]
    plt.scatter(X_test[idx, 1], X_test[idx, 2], color=color, label="Class %s" % str(n))

plt.scatter(X_test[incorrect_idx, 1], X_test[incorrect_idx, 2], color="darkred")

plt.xlabel('sepal width [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc=3)
plt.title("Iris Classification results")
plt.show()
```

We can see that the errors occur in the area where green (class 1) and gray (class 2) overlap. This gives us insight about what features to add - any feature which helps separate class 1 and class 2 should improve classifier performance.

## <a name="Supervised Learning -- Classification">Supervised Learning -- Classification</a>

First, we will look at a two class classification problem in two dimensions. We use the synthetic data generated by the make_blobs function.

```python
from sklearn.datasets import make_blobs

X, y = make_blobs(centers=2, random_state=321)
```

As the data is two-dimensional, we can plot each sample as a point in a two-dimensional coordinate system, with the first feature being the x-axis and the second feature being the y-axis.

```python
plt.scatter(X[y == 0, 0], X[y == 0, 1],
            c='blue', s=40, label='0')
plt.scatter(X[y == 1, 0], X[y == 1, 1],
            c='red', s=40, label='1', marker='s')

plt.xlabel('first feature')
plt.ylabel('second feature')
plt.legend(loc='upper right');
```

### The scikit-learn estimator API

![sklearn2](/mldl/assets/images/2018-06-29/sklearn2.jpg)

Every algorithm is exposed in scikit-learn via an ''Estimator'' object. (All models in scikit-learn have a very consistent interface). For instance, we first import the logistic regression class. Next, we instantiate the estimator object.

```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
```

To built the model from our data, that is to learn how to classify new points, we call the `fit` function with the training data, and the corresponding training labels (the desired output for the training data point):

```python
classifier.fit(X_train, y_train)
```

(Some estimator methods such as fit return self by default. Thus, after executing the code snippet above, you will see the default parameters of this particular instance of `LogisticRegression`. Another way of retrieving the estimator's ininitialization parameters is to execute `classifier.get_params()`, which returns a parameter dictionary.)

We can then apply the model to unseen data and use the model to predict the estimated outcome using the `predict` method. We can evaluate our classifier quantitatively by measuring what fraction of predictions is correct. This is called accuracy:

```python
np.mean(prediction == y_test)
```

There is also a convenience function , `score`, that all scikit-learn classifiers have to compute this directly from the test data:

```python
classifier.score(X_test, y_test)
# It is often helpful to compare the generalization performance (on the test set) to the performance on the training set:
classifier.score(X_train, y_train)
```

LogisticRegression is a so-called linear model, that means it will create a decision that is linear in the input space. In 2d, this simply means it finds a line to separate the blue from the red:

```python
from figures import plot_2d_separator

plt.scatter(X[y == 0, 0], X[y == 0, 1],
            c='blue', s=40, label='0')
plt.scatter(X[y == 1, 0], X[y == 1, 1],
            c='red', s=40, label='1', marker='s')

plt.xlabel("first feature")
plt.ylabel("second feature")
plot_2d_separator(classifier, X)
plt.legend(loc='upper right');
```

Estimated parameters: All the estimated model parameters are attributes of the estimator object ending by an underscore. Here, these are the coefficients and the offset of the line:

```python
print(classifier.coef_)
print(classifier.intercept_)
```

## <a name="Scikit-learn's estimator interface">Scikit-learn's estimator interface</a>

Scikit-learn strives to have a uniform interface across all methods. Given a scikit-learn `estimator` object named _model_, the following methods are available (not all for each model):

* Available in **all Estimators**:
    * `model.fit()`: fit training data. For supervised learning applications, this accepts two arguments: the data `X` and the labels `y` (e.g. `model.fit(X, y)`). For unsupervised learning applications, `fit` takes only a single argument, the data `X` (e.g. `model.fit(X)`).
* Available in **supervised estimators**
    * `model.predict()`: given a trained model, predict the label of a new set of data. This method accepts one argument, the new data `X_new` (e.g. `model.predict(X_new)``), and returns the learned label for each object in the array.
    * `model.predict_proba()`: For classification problems, some estimators also provide this method, which returns the probability that a new observation has each categorical label. In this case, the label with the highest probability is returned by model.predict().
    * `model.decision_function()`: For classification problems, some estimators provide an uncertainty estimate that is not a probability. For binary classification, a decision_function >= 0 means the positive class will be predicted, while < 0 means the negative class.
    * `model.score()`: for classification or regression problems, most (all?) estimators implement a score method. Scores are between 0 and 1, with a larger score indicating a better fit. For classifiers, the score method computes the prediction accuracy. For regressors, score computes the coefficient of determination (R2) of the prediction.
    * `model.transform()`: For feature selection algorithms, this will reduce the dataset to the selected features. For some classification and regression models such as some linear models and random forests, this method reduces the dataset to the most informative features. These classification and regression models can therefore also be used as feature selection methods.
* Available in **unsupervised estimators**
    * `model.transform()`: given an unsupervised model, transform new data into the new basis. This also accepts one argument `X_new`, and returns the new representation of the data based on the unsupervised model.
    * `model.fit_transform()`: some estimators implement this method, which more efficiently performs a fit and a transform on the same input data.
    * `model.predict()`:  for clustering algorithms, the predict method will produce cluster labels for new data points. Not all clustering methods have this functionality.
    * `model.predict_proba()`: Gaussian mixture models (GMMs) provide the probability for each point to be generated by a given mixture component.
    * `model.score()`: Density models like KDE and GMMs provide the likelihood of the data under the model.

Apart from `fit`, the two most important functions are arguably `predict` which produces a target variable (a y) and `transform`, which produces a new representation of the data (an X). The following table shows for which class of models which function applies:

model.predict | model.transform
---| ---
Classification | Preprocessing
Regression | Dimensionality Reduction
Clustering | Feature Extraction
           | Feature Selection


## <a name="Supervised Learning -- Regression">Supervised Learning -- Regression</a>

In regression we are trying to predict a continuous output variable -- in contrast to the nominal variables we were predicting in the previous classification examples.

### Linear Regression

One of the simplest models again is a linear one, that simply tries to predict the data as lying on a line. One way to find such a line is LinearRegression (also known as Ordinary Least Squares (OLS) regression). The interface for `LinearRegression` is exactly the same as for the classifiers before, only that `y` now contains float values, instead of classes.

As we remember, the scikit-learn API requires us to provide the target variable (`y`) as a 1-dimensional array; scikit-learn's API expects the samples (`X`) in form a 2-dimensional array -- even though it may only consist of 1 feature. Thus, let us convert the 1-dimensional x NumPy array into an X array with 2 axes:

```python
x = np.linspace(-3, 3, 100)
rng = np.random.RandomState(321)
y = np.sin(4 * x) + x + rng.uniform(size=len(x))
X = x[:, np.newaxis]

# Again, we start by splitting our dataset into a training (75%) and a test set (25%):
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=321)

# Next, we use the learning algorithm implemented in LinearRegression to fit a regression model to the training data:
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# After fitting to the training data, we paramerterized a linear regression model with the following values.

print('Weight coefficients: ', regressor.coef_)
print('y-axis intercept: ', regressor.intercept_)
```

Plugging in the min and max values into the equation, we can plot the regression fit to our training data:

```python
min_pt = X.min() * regressor.coef_[0] + regressor.intercept_
max_pt = X.max() * regressor.coef_[0] + regressor.intercept_

plt.plot([X.min(), X.max()], [min_pt, max_pt])
plt.plot(X_train, y_train, 'o');
```

Similar to the estimators for classification in the previous notebook, we use the `predict` method to predict the target variable.

```python
y_pred_test = regressor.predict(X_test)
plt.plot(X_test, y_test, 'o', label="data")
plt.plot(X_test, y_pred_test, 'o', label="prediction")
plt.plot([X.min(), X.max()], [min_pt, max_pt], label='fit')
plt.legend(loc='best');
```

Again, scikit-learn provides an easy way to evaluate the prediction quantitatively using the `score` method. For regression tasks, this is the **R2 score**. Another popular way would be the **Mean Squared Error (MSE)**. As its name implies, the MSE is simply the average squared difference over the predicted and actual target values

### KNeighborsRegression

As for classification, we can also use a neighbor based method for regression. We can simply take the output of the nearest point, or we could average several nearest points. This method is less popular for regression than for classification, but still a good baseline.

```python
from sklearn.neighbors import KNeighborsRegressor
kneighbor_regression = KNeighborsRegressor(n_neighbors=1)
kneighbor_regression.fit(X_train, y_train)
```

```python
y_pred_test = kneighbor_regression.predict(X_test)

plt.plot(X_test, y_test, 'o', label="data", markersize=8)
plt.plot(X_test, y_pred_test, 's', label="prediction", markersize=4)
plt.legend(loc='best');
```

On the test set, we also do a better job of capturing the variation, but our estimates look much messier than before. Let us look at the R2 score:

```python
kneighbor_regression.score(X_test, y_test)
```

Much better than before! Here, the linear model was not a good fit for our problem; it was lacking in complexity and thus under-fit our data.


## <a name="Unsupervised Learning -- Transformation">Unsupervised Learning -- Transformation</a>

Many instances of unsupervised learning, such as dimensionality reduction, manifold learning, and feature extraction, find a new representation of the input data without any additional input. (In contrast to supervised learning, usnupervised algorithms don't require or consider target variables like in the previous classification and regression examples).

![sklearn3](/mldl/assets/images/2018-06-29/sklearn3.jpg)

A very basic example is the rescaling of our data, which is a requirement for many machine learning algorithms as they are not scale-invariant -- rescaling falls into the category of data pre-processing and can barely be called learning. There exist many different rescaling techniques, and in the following example, we will take a look at a particular method that is commonly called "**standardization**". Here, we will rescale the data so that each feature is centered at zero (mean = 0) with unit variance (standard deviation = 1).

```python
ary = np.array([1, 2, 3, 4, 5])
ary_standardized = (ary - ary.mean()) / ary.std()
ary_standardized
```

Since **standardization** is such a basic preprocessing procedure -- as we've seen in the code snipped above -- scikit-learn implements a `StandardScaler` class for this computation. And in later sections, we will see why and when the scikit-learn interface comes in handy over the code snippet we executed above.

Applying such a preprocessing has a very similar interface to the supervised learning algorithms we saw so far. To get some more practice with scikit-learn's "Transformer" interface, let's start by loading the iris dataset and rescale it:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

# The iris dataset is not "centered" that is it has non-zero mean and the standard deviation is different for each component:
print("mean : %s " % X_train.mean(axis=0)) # mean : [5.88660714 3.05178571 3.79642857 1.22232143]
print("standard deviation : %s " % X_train.std(axis=0)) # standard deviation : [0.86741565 0.43424445 1.79264014 0.77916047]
```

To use a preprocessing method, we first import the estimator, here `StandardScaler` and instantiate it:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
```

As with the classification and regression algorithms, we call `fit` to learn the model from the data. As this is an unsupervised model, we only pass X, not y. This simply estimates mean and standard deviation.

```python
scaler.fit(X_train)
```

Now we can rescale our data by applying the transform (not predict) method:

```python
X_train_scaled = scaler.transform(X_train)
```

`X_train_scaled` has the same number of samples and features, but the mean was subtracted and all features were scaled to have unit standard deviation.

To summarize: Via the `fit` method, the estimator is fitted to the data we provide. In this step, the estimator estimates the parameters from the data (here: mean and standard deviation). Then, if we `transform` data, these parameters are used to transform a dataset. (Please note that the transform method does not update these parameters).

It's important to note that the same transformation is applied to the training and the test set. That has the consequence that usually the mean of the test data is not zero after scaling:

```python
X_test_scaled = scaler.transform(X_test)
print("mean test data: %s" % X_test_scaled.mean(axis=0))
```

It is important for the training and test data to be transformed in exactly the same way, for the following processing steps to make sense of the data. There are several common ways to scale the data. The most common one is the `StandardScaler` we just introduced, but rescaling the data to a fix minimum an maximum value with `MinMaxScaler` (usually between 0 and 1), or using more robust statistics like median and quantile, instead of mean and standard deviation (with `RobustScaler`), are also useful.

### Principal Component Analysis

An unsupervised transformation that is somewhat more interesting is Principal Component Analysis (PCA). It is a technique to reduce the dimensionality of the data, by creating a linear projection. That is, we find new features to represent the data that are a linear combination of the old data (i.e. we rotate it). Thus, we can think of PCA as a projection of our data onto a new feature space.

The way PCA finds these new directions is by looking for the directions of maximum variance. Usually only few components that explain most of the variance in the data are kept. Here, the premise is to reduce the size (dimensionality) of a dataset while capturing most of its information. There are many reason why dimensionality reduction can be useful: It can reduce the computational cost when running learning algorithms, decrease the storage space, and may help with the so-called "curse of dimensionality," which we will discuss in greater detail later.

We create a Gaussian blob that is rotated:

```python
rnd = np.random.RandomState(5)
X_ = rnd.normal(size=(300, 2))
X_blob = np.dot(X_, rnd.normal(size=(2, 2))) + rnd.normal(size=2)
y = X_[:, 0] > 0
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y, linewidths=0, s=30)
plt.xlabel("feature 1")
plt.ylabel("feature 2");
```

As always, we instantiate our PCA model. By default all directions are kept.

```python
from sklearn.decomposition import PCA
pca = PCA()

# Then we fit the PCA model with our data. As PCA is an unsupervised algorithm, there is no output y.
pca.fit(X_blob)

# Then we can transform the data, projected on the principal components:
X_pca = pca.transform(X_blob)
```

```python
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, linewidths=0, s=30)
plt.xlabel("first principal component")
plt.ylabel("second principal component");
```

## <a name="Unsupervised Learning - Clustering">Unsupervised Learning - Clustering</a>

Clustering is the task of gathering samples into groups of similar samples according to some predefined similarity or distance (dissimilarity) measure, such as the Euclidean distance. In this section we will explore a basic clustering task on some synthetic and real-world datasets.

Let's start by creating a simple, 2-dimensional, synthetic dataset:

```python
from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=42)
X.shape
plt.scatter(X[:, 0], X[:, 1]);
```

In the scatter plot above, we can see three separate groups of data points and we would like to recover them using clustering -- think of "discovering" the class labels that we already take for granted in a classification task.

Even if the groups are obvious in the data, it is hard to find them when the data lives in a high-dimensional space, which we can't visualize in a single histogram or scatterplot.

Now we will use one of the simplest clustering algorithms, K-means. This is an iterative algorithm which searches for three cluster centers such that the distance from each point to its cluster is minimized. The standard implementation of K-means uses the Euclidean distance, which is why we want to make sure that all our variables are measured on the same scale if we are working with real-world datastets. In the previous notebook, we talked about one technique to achieve this, namely, standardization.

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
```

We can get the cluster labels either by calling `fit` and then accessing the labels_ attribute of the K means estimator, or by calling `fit_predict`. Either way, the result contains the ID of the cluster that each point is assigned to.

```python
labels = kmeans.fit_predict(X)
labels.shape # (100,)

# Let's visualize the assignments that have been found
plt.scatter(X[:, 0], X[:, 1], c=labels);

# Compared to the true labels:
plt.scatter(X[:, 0], X[:, 1], c=y);
```

Here, we are probably satisfied with the clustering results. But in general we might want to have a more quantitative evaluation. How about comparing our cluster labels with the ground truth we got when generating the blobs?

```python
from sklearn.metrics import confusion_matrix, accuracy_score

print('Accuracy score:', accuracy_score(y, labels))
print(confusion_matrix(y, labels))
```

Even though we recovered the partitioning of the data into clusters perfectly, the cluster IDs we assigned were arbitrary, and we can not hope to recover them. Therefore, we must use a different scoring metric, such as adjusted_rand_score, which is invariant to permutations of the labels:

```python
from sklearn.metrics import adjusted_rand_score

adjusted_rand_score(y, labels)
```

One of the "short-comings" of K-means is that we have to specify the number of clusters, which we often don't know apriori. For example, let's have a look what happens if we set the number of clusters to 2 in our synthetic 3-blob dataset:

```python
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels);
```

### The Elbow Method

The Elbow method is a "rule-of-thumb" approach to finding the optimal number of clusters. Here, we look at the cluster dispersion for different values of k:

```python
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')

plt.show()
```

Then, we pick the value that resembles the "pit of an elbow." As we can see, this would be k=3 in this case, which makes sense given our visual expection of the dataset previously.

Clustering comes with assumptions: A clustering algorithm finds clusters by making assumptions with samples should be grouped together. Each algorithm makes different assumptions and the quality and interpretability of your results will depend on whether the assumptions are satisfied for your goal. For K-means clustering, the model is that all clusters have equal, spherical variance.

In general, there is no guarantee that structure found by a clustering algorithm has anything to do with what you were intereste

### Some Notable Clustering Routines

The following are some well-known clustering algorithms.

* `sklearn.cluster.KMeans`: The simplest, yet effective clustering algorithm. Needs to be provided with the number of clusters in advance, and assumes that the data is normalized as input (but use a PCA model as preprocessor).
* `sklearn.cluster.MeanShift`: Can find better looking clusters than KMeans but is not scalable to high number of samples.
* `sklearn.cluster.DBSCAN`: Can detect irregularly shaped clusters based on density, i.e. sparse regions in the input space are likely to become inter-cluster boundaries. Can also detect outliers (samples that are not part of a cluster).
* `sklearn.cluster.AffinityPropagation`: Clustering algorithm based on message passing between data points.
* `sklearn.cluster.SpectralClustering`: KMeans applied to a projection of the normalized graph Laplacian: finds normalized graph cuts if the affinity matrix is interpreted as an adjacency matrix of a graph.
* `sklearn.cluster.Ward`: Ward implements hierarchical clustering based on the Ward algorithm, a variance-minimizing approach. At each step, it minimizes the sum of squared differences within all clusters (inertia criterion).

Of these, Ward, SpectralClustering, DBSCAN and Affinity propagation can also work with precomputed similarity matrices.

![sklearn4](/mldl/assets/images/2018-06-29/sklearn4.jpg)

## <a name="Feature Extraction">Feature Extraction</a>

### Numerical Features

Numerical features are pretty straightforward: each sample contains a list of floating-point numbers corresponding to the features.

```python
from sklearn.datasets import load_iris

iris = load_iris()
print(iris.data.shape) # (150, 4)
```

### Categorical Features

What if you have categorical features? For example, imagine there is data on the color of each iris: "color in [red, blue, purple]". You might be tempted to assign numbers to these features, i.e. red=1, blue=2, purple=3 but in general **this is a bad idea**. Estimators tend to operate under the assumption that numerical features lie on some continuous scale, so, for example, 1 and 2 are more alike than 1 and 3, and this is often not the case for categorical features.

In fact, the example above is a subcategory of "categorical" features, namely, "nominal" features. Nominal features don't imply an order, whereas "ordinal" features are categorical features that do imply an order. An example of ordinal features would be T-shirt sizes, e.g., XL > L > M > S. One work-around for parsing nominal features into a format that prevents the classification algorithm from asserting an order is the so-called **one-hot encoding** representation. Here, we give each category its own dimension.

Note that using many of these categorical features may result in data which is better represented as a **sparse matrix**, as we'll see with the text classification example below.

**Using the DictVectorizer to encode categorical features**: When the source data which needs to be encoded has a list of dicts where the values are either strings names for categories or numerical values, you can use the DictVectorizer class to compute the boolean expansion of the categorical features while leaving the numerical features unimpacted:

```python
from sklearn.feature_extraction import DictVectorizer

measurements = [
    {'city': 'Dubai', 'temperature': 33.},
    {'city': 'London', 'temperature': 12.},
    {'city': 'San Francisco', 'temperature': 18.},
]

vec = DictVectorizer()
dict_vec = vec.fit_transform(measurements)
dict_vec.toarray() # prints: array([[ 1.,  0.,  0., 33.],
                   #                [ 0.,  1.,  0., 12.],
                   #                [ 0.,  0.,  1., 18.]])
vec.get_feature_names() # ['city=Dubai', 'city=London', 'city=San Francisco', 'temperature']
```

### Derived Features

Another common feature type is **derived features**, where some pre-processing step is applied to the data to generate features that are somehow more informative. Derived features may be based on **feature extraction** and **dimensionality reduction** (such as PCA or manifold learning), may be linear or nonlinear combinations of features (such as in polynomial regression), or may be some more sophisticated transform of the features.

## <a name="Text Feature Extraction">Text Feature Extraction</a>

### Bag of words CountVectorizer

![sklearn5](/mldl/assets/images/2018-06-29/sklearn5.jpg)

In many tasks, like in the classical spam detection, your input data is text. Free text with variable length is very far from the fixed length numeric representation that we need to do machine learning with scikit-learn. However, there is an easy and effective way to go from text data to a numeric representation using the so-called bag-of-words model, which provides a data structure that is compatible with the machine learning aglorithms in scikit-learn.

Let's assume that each sample in your dataset is represented as one string, which could be just a sentence, an email, or a whole news article or book. To represent the sample, we first split the string into a list of tokens, which correspond to (somewhat normalized) words. A simple way to do this is to just split by whitespace, and then lowercase the word.

Then, we build a vocabulary of all tokens (lowercased words) that appear in our whole dataset. This is usually a very large vocabulary. Finally, looking at our single sample, we could show how often each word in the vocabulary appears. We represent our string by a vector, where each entry is how often a given word in the vocabulary appears in the string.

As each sample will only contain very few words, most entries will be zero, leading to a very high-dimensional but sparse representation. The method is called "bag-of-words," as the order of the words is lost entirely.

```python
from sklearn.feature_extraction.text import CountVectorizer

X = ["Some say the world will end in fire,",
     "Some say in ice."]
vectorizer = CountVectorizer()
vectorizer.fit(X)
vectorizer.vocabulary_  # returns: {'end': 0, 'fire': 1, 'ice': 2, 'in': 3, 'say': 4, 'some': 5, 'the': 6, 'will': 7, 'world': 8}
X_bag_of_words = vectorizer.transform(X)
X_bag_of_words.toarray()  # returns: array([[1, 1, 0, 1, 1, 1, 1, 1, 1],
                          #                 [0, 0, 1, 1, 1, 1, 0, 0, 0]])
vectorizer.get_feature_names()  # ['end', 'fire', 'ice', 'in', 'say', 'some', 'the', 'will', 'world']
```

### TfidfVectorizer

A useful transformation that is often applied to the bag-of-word encoding is the so-called term-frequency inverse-document-frequency (tf-idf) scaling, which is a non-linear transformation of the word counts. The tf-idf encoding rescales words that are common to have less weight.

tf-idfs are a way to represent documents as feature vectors. tf-idfs can be understood as a modification of the raw term frequencies (tf); the tf is the count of how often a particular word occurs in a given document. The concept behind the tf-idf is to downweight terms proportionally to the number of documents in which they occur. Here, the idea is that terms that occur in many different documents are likely unimportant or don't contain any useful information for Natural Language Processing tasks such as document classification.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
tfidf.fit(X)
np.set_printoptions(precision=2)
tfidf.transform(X).toarray()  # returns: array([[0.39, 0.39, 0.  , 0.28, 0.28, 0.28, 0.39, 0.39, 0.39],
                              #                 [0.  , 0.  , 0.63, 0.45, 0.45, 0.45, 0.  , 0.  , 0.  ]])
```

### Bigrams and N-Grams

In the example illustrated in the figure at the beginning of this notebook, we used the so-called 1-gram (unigram) tokenization: Each token represents a single element with regard to the splittling criterion. Entirely discarding word order is not always a good idea, as composite phrases often have specific meaning, and modifiers like "not" can invert the meaning of words.

A simple way to include some word order are n-grams, which don't only look at a single token, but at all pairs of neighboring tokens. For example, in 2-gram (bigram) tokenization, we would group words together with an overlap of one word; in 3-gram (trigram) splits we would create an overlap two words, and so forth.

Which "n" we choose for "n-gram" tokenization to obtain the optimal performance in our predictive model depends on the learning algorithm, dataset, and task. Or in other words, we have consider "n" in "n-grams" as a tuning parameters, and in later notebooks, we will see how we deal with these. Now, let's create a bag of words model of bigrams using scikit-learn's CountVectorizer:

```python
bigram_vectorizer = CountVectorizer(ngram_range=(2,2))
bigram_vectorizer.fit(X)
bigram_vectorizer.transform(X).toarray()  # returns: array([[1, 1, 0, 0, 1, 1, 1, 1, 1],
                                          #                 [0, 0, 1, 1, 0, 1, 0, 0, 0]])
bigram_vectorizer.get_feature_names()     # returns: ['end in', 'in fire', 'in ice', 'say in', 'say the', 'some say', 'the world', 'will end', 'world will']
```

Often we want to include unigrams (single tokens) AND bigrams, wich we can do by passing the following tuple as an argument to the ngram_range parameter of the CountVectorizer function:

```python
gram_vectorizer = CountVectorizer(ngram_range=(1, 2))
gram_vectorizer.fit(X)
gram_vectorizer.get_feature_names()  # returns: ['end', 'end in', 'fire', 'ice', 'in', 'in fire', 'in ice', 'say', 'say in', 'say the', 'some', 'some say', 'the', 'the world', 'will', 'will end', 'world', 'world will']
gram_vectorizer.transform(X).toarray()  # returns: array([[1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                        #                 [0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0]])
```

### Character n-grams

Sometimes it is also helpful not only to look at words, but to consider single characters instead. That is particularly useful if we have very noisy data and want to identify the language, or if we want to predict something about a single word. We can simply look at characters instead of words by setting analyzer="char". Looking at single characters is usually not very informative, but looking at longer n-grams of characters could be:

```python
char_vectorizer = CountVectorizer(ngram_range=(2, 2), analyzer="char")
char_vectorizer.fit(X)
print(char_vectorizer.get_feature_names())  # returns: [' e', ' f', ' i', ' s', ' t', ' w', 'ay', 'ce', 'd ', 'e ', 'e,', 'e.', 'en', 'fi', 'he', 'ic', 'il', 'in', 'ir', 'l ', 'ld', 'll', 'me', 'n ', 'nd', 'om', 'or', 're', 'rl', 'sa', 'so', 'th', 'wi', 'wo', 'y ']
```

## <a name="Text classification for SMS spam detection">Text classification for SMS spam detection</a>

We perform some simple preprocessing and split the data array into two parts: 1. text: A list of lists, where each sublists contains the contents of our emails. 2. y: our SPAM vs HAM labels stored in binary; a 1 represents a spam message, and a 0 represnts a ham (non-spam) message.

```python
import os

with open(os.path.join("datasets","smsspam","SMSSpamCollection")) as f:
    lines = [line.strip().split("\t") for line in f.readlines()]
text = [x[1] for x in lines]
y = [int(x[0] == "spam") for x in lines]

print(text[:5])  # prints ['Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...', 'Ok lar... Joking wif u oni...', "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's", 'U dun say so early hor... U c already then say...', "Nah I don't think he goes to usf, he lives around here though"]
print(y[:5])    # prints [0, 0, 1, 0, 0]
```

Next, we split our dataset into 2 parts, the test and training dataset:

```python
from sklearn.model_selection import train_text_split

text_train, text_test, y_train, y_test = train_test_split(text, y, test_size=0.25, random_state=321, stratify=y)
```

Now, we use the CountVectorizer to parse the text data into a bag-of-words model.

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(text_train)
X_train = vectorizer.transform(text_train)
X_test  = vectorizer.transform(text_test)
print(X_train.shape)  # (4180, 7453)
print(X_test.shape)   # (1394, 7453)
```

We can now train a classifier, for instance a logistic regression classifier, which is a fast baseline for text classification tasks:

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)  # 98.49%
```

![sklearn6](/mldl/assets/images/2018-06-29/sklearn6.jpg)


## <a name="Cross Validation">Cross Validation</a>

In the previous sections and notebooks, we split our dataset into two parts, a training set and a test set. We used the training set to fit our model, and we used the test set to evaluate its generalization performance -- how well it performs on new, unseen data.

![sklearn7](/mldl/assets/images/2018-06-29/sklearn7.jpg)

However, often (labeled) data is precious, and this approach lets us only use ~ 3/4 of our data for training. On the other hand, we will only ever try to apply our model 1/4 of our data for testing. A common way to use more of the data to build a model, but also get a more robust estimate of the generalization performance, is cross-validation. In cross-validation, the data is split repeatedly into a training and non-overlapping test-sets, with a separate model built for every pair. The test-set scores are then aggregated for a more robust estimate.

The most common way to do cross-validation is k-fold cross-validation, in which the data is first split into k (often 5 or 10) equal-sized folds, and then for each iteration, one of the k folds is used as test data, and the rest as training data:

![sklearn8](/mldl/assets/images/2018-06-29/sklearn8.jpg)

This way, each data point will be in the test-set exactly once, and we can use all but a k'th of the data for training. Let us apply this technique to evaluate the KNeighborsClassifier algorithm on the Iris dataset:

```python
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

iris = load_iris()
X, y = iris.data, iris.target

classifier = KNeighborsClassifier()
scores = cross_val_score(classifier, X, y, cv=5)
print(scores)  # [0.97, 1.  , 0.93, 0.97, 1.  ]
print(np.mean(scores))  # 0.9733
```

There are also helper objects in the cross-validation module that will generate indices for you for all kinds of different cross-validation methods, including k-fold: `from sklearn.model_selection import KFold, StrafifiedKFold, ShuffleSplit`.

By default, `cross_val_score` will use `StratifiedKFold` for classification, which ensures that the class proportions in the dataset are reflected in each fold. If you have a binary classification dataset with 90% of data point belonging to class 0, that would mean that in each fold, 90% of datapoints would belong to class 0. If you would just use `KFold` cross-validation, it is likely that you would generate a split that only contains class 0. It is generally a good idea to use `StratifiedKFold` whenever you do classification.

`StratifiedKFold` would also remove our need to shuffle iris. If you use `Kfold`, you will have to explicitly specify `shuffle=True`.

## <a name="Model Complexity and GridSearchCV">Model Complexity and GridSearchCV</a>

Most models have parameters that influence how complex a model they can learn. Remember using `KNeighborsRegressor`. If we change the number of neighbors we consider, we get a smoother and smoother prediction.

We want to find a model with parameters that fits the data fairly well, and does not suffer from either the overfit or underfit problems. What we would like is a way to quantitatively identify overfit and underfit, and optimize the hyperparameters in order to determine the best algorithm.

We trade off remembering too much about the particularities and noise of the training data vs. not modeling enough of the variability. This is a trade-off that needs to be made in basically every machine learning application and is a central concept, called bias-variance-tradeoff or "overfitting vs underfitting".

![sklearn9](/mldl/assets/images/2018-06-29/sklearn9.jpg)

Unfortunately, there is no general rule how to find the sweet spot, and so machine learning practitioners have to find the best trade-off of model-complexity and generalization by trying several hyperparameter settings. Hyperparameters are the internal knobs or tuning parameters of a machine learning algorithm (in contrast to model parameters that the algorithm learns from the training data -- for example, the weight coefficients of a linear regression model); the number of k in K-nearest neighbors is such a hyperparameter.

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegreseor
# generate toy dataset:
x = np.linspace(-3, 3, 100)
rng = np.random.RandomState(42)
y = np.sin(4 * x) + x + rng.normal(size=len(x))
X = x[:, np.newaxis]

cv = KFold(shuffle=True)
# for each parameter setting do cross-validation:
for n_neighbors in [1, 3, 5, 10, 20]:
    scores = cross_val_score(KNeighborsRegreseor(n_neighbors=n_neighbors), X, y, cv=cv)
    print("n_neighbors: %d, average score: %f" % (n_neighbors, np.mean(scores)))
```

There is a function in scikit-learn, called `validation_plot` to reproduce the cartoon figure above. It plots one parameter, such as the number of neighbors, against training and validation error (using cross-validation):

```python
from sklearn.model_selection import validation_curve
n_neighbors = [1, 3, 5, 10, 20, 50]
train_errors, test_errors = validation_curve(KNeighborsRegressor(), X, y, param_name="n_neighbors",
                                             param_range=n_neighbors, cv=cv)
plt.plot(n_neighbors, train_errors.mean(axis=1), label="train error")
plt.plot(n_neighbors, test_errors.mean(axis=1), label="test error")
plt.legend(loc="best")
```

As this is such a very common pattern, there is a built-in class for this in scikit-learn, `GridSearchCV.GridSearchCV` takes a dictionary that describes the parameters that should be tried and a model to train. The grid of parameters is defined as a dictionary, where the keys are the parameters and the values are the settings to be tested.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

param_grid = {"C": [0.001, 0.01, 0.1, 1, 10], "gamma": [0.001, 0.01, 0.1, 1]}
grid = GridSearchCV(SVR(), param_grid=param_grid, cv=cv, verbose=3)
```

One of the great things about GridSearchCV is that it is a meta-estimator. It takes an estimator like SVR above, and creates a new estimator, that behaves exactly the same - in this case, like a regressor. So we can call `fit` on it, to train it:

What `fit` does is a bit more involved then what we did above. First, it runs the same loop with cross-validation, to find the best parameter combination. Once it has the best combination, it runs fit again on all data passed to fit (without cross-validation), to built a single new model using the best parameter setting. Then, as with all models, we can use `predict` or `score`. You can inspect the best parameters found by `GridSearchCV` in the `best_params_` attribute, and the best score in the `best_score_` attribute:

```python
grid.fit(X, y)
grid.predict(X)
print(grid.best_score_)    # 0.7404309946094338
print(grid.best_params_)   # {'C': 10, 'gamma': 1}
```

There is a problem with using this score for evaluation, however. You might be making what is called a multiple hypothesis testing error. If you try very many parameter settings, some of them will work better just by chance, and the score that you obtained might not reflect how your model would perform on new unseen data. Therefore, it is good to split off a separate test-set before performing grid-search. This pattern can be seen as a training-validation-test split, and is common in machine learning:

![sklearn10](/mldl/assets/images/2018-06-29/sklearn10.jpg)

We can do this very easily by splitting of some test data using `train_test_split`, training `GridSearchCV` on the training set, and applying the score method to the test set.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1]}
cv = KFold(n_splits=10, shuffle=True)
grid = GridSearchCV(SVR(), param_grid=param_grid, cv=cv)
grid.fit(X_train, y_train)
grid.score(X_test, y_test)
```

Some practitioners go for an easier scheme, splitting the data simply into three parts, **training, **validation** and **testing**. This is a possible alternative if your training set is very large, or it is infeasible to train many models using cross-validation because training a model takes very long. You can do this with scikit-learn for example by splitting of a test-set and then applying `GridSearchCV` with `ShuffleSplit` cross-validation with a single iteration. This is much faster, but might result in worse hyperparameters and therefore worse results.

```python
from sklearn.model_selection import train_test_split, ShuffleSplit
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1]}
single_split_cv = ShuffleSplit(n_splits=1)
grid = GridSearchCV(SVR(), param_grid=param_grid, cv=single_split_cv, verbose=3)
grid.fit(X_train, y_train)
grid.score(X_test, y_test)
```

## <a name="Pipeline estimators">Pipeline estimators</a>

In this section we study how different estimators maybe be chained.

For some types of data, for instance text data, a feature extraction step must be applied to convert it to numerical features. Previously, we applied the feature extraction manually, like so:

```python
from sklearn.feature_extraction.text TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer()
vectorizer.fit(text_train)

X_train = vectorizer.transform(text_train)
X_test = vectorizer.transform(text_test)

clf = LogisticRegression()
clf.fit(X_train, y_train)

clf.score(X_test, y_test)
)
```

The situation where we learn a transformation and then apply it to the test data is very common in machine learning. Therefore scikit-learn has a shortcut for this, called `pipeline`:

```python
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression())
pipeline.fit(text_train, y_train)
pipeline.score(text_test, y_test)
```

As you can see, this makes the code much shorter and easier to handle. Behind the scenes, exactly the same as above is happening. When calling `fit` on the pipeline, it will call `fit` on each step in turn. After the first step is `fit`, it will use the `transform` method of the first step to create a new representation. This will then be fed to the `fit` of the next step, and so on. Finally, on the last step, only `fit` is called.

![sklearn11](/mldl/assets/images/2018-06-29/sklearn11.jpg)

If we call `score`, only `transform` will be called on each step - this could be the test set after all! Then, on the last step, `score` is called with the new representation. The same goes for `predict`.

Building pipelines not only simplifies the code, it is also important for model selection. Say we want to grid-search C to tune our Logistic Regression above. Let's say we do it like this:

```python
# This illustrates a common mistake. Don't use this code!
from sklearn.model_selection import GridSearchCV

vectorizer = TfidfVectorizer()
vectorizer.fit(text_train)

X_train = vectorizer.transform(text_train)
X_test = vectorizer.transform(text_test)

clf = LogisticRegression()
grid = GridSearchCV(clf, param_grid={'C': [.1, 1, 10, 100]}, cv=5)
grid.fit(X_train, y_train)
```

### What did we do wrong?

Here, we did grid-search with cross-validation on **X_train**. However, when applying `TfidfVectorizer`, it saw all of the **X_train**, not only the training folds! So it could use knowledge of the frequency of the words in the test-folds. This is called "contamination" of the test set, and leads to too optimistic estimates of generalization performance, or badly selected parameters. We can fix this with the pipeline, though:

```python
from sklearn.model_selection import GridSearchCV

pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression())
grid = GridSearchCV(pipeline, param_grid={"logisticregression__C": [.1, 1, 10, 100]}, cv=5)
grid.fit(text_train, y_train)
grid.score(text_test, t_test)
```

Note that we need to tell the pipeline where at which step we wanted to set the parameter `C`. We can do this using the special __ syntax. The name before the __ is simply the name of the class, the part after __ is the parameter we want to set with grid-search.

![sklearn12](/mldl/assets/images/2018-06-29/sklearn12.jpg)

Another benefit of using pipelines is that we can now also search over parameters of the feature extraction with `GridSearchCV`:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression())
params = {"logisticregression__C": [.1, 1, 10, 100], "tfidfvectorizer__ngram_range: [(1, 1), (1, 2), (2, 2)]"}
grid = GridSearchCV(pipeline, param_grid=params, cv=5)
grid.fit(text_train, y_train)
print(grid.best_params_)
grid.score(text_test, y_test)
```

## <a name="Performance Metrics, Model Evaluation, and Dealing with Class Imbalances">Performance Metrics, Model Evaluation, and Dealing with Class Imbalances</a>

In the previous notebook, we already went into some detail on how to evaluate a model and how to pick the best model. So far, we assumed that we were given a performance measure, a measure of the quality of the model. What measure one should use is not always obvious, though. The default scores in scikit-learn are `accuracy` for classification, which is the fraction of correctly classified samples, and `r2` for regression, which is the coefficient of determination. These are reasonable default choices in many scenarios; however, depending on our task, these are not always the definitive or recommended choices.

Let's take a look at classification in more detail, going back to the application of classifying handwritten digits. So, how about training a classifier and walking through the different ways we can evaluate it? Scikit-learn has many helpful methods in the sklearn.metrics module that can help us with this task:

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=2)

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=321, stratify=y)
classifier = LinearSVC().fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)
print("Accuracy: {}".format(classifier.score(X_test, y_test)))  # Accuracy: 0.9466666666666667
```

For multi-class problems, it is often interesting to know which of the classes are hard to predict, and which are easy, or which classes get confused. One way to get more information about misclassifications is the `confusion_matrix`, which shows for each true class, how frequent a given predicted outcome is.

```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_test_pred)
##The result:
#array([[45,  0,  0,  0,  0,  0,  0,  0,  0,  0],
#       [ 0, 40,  0,  0,  0,  0,  0,  0,  6,  0],
#       [ 0,  0, 42,  1,  0,  0,  0,  0,  0,  1],
#       [ 0,  0,  0, 43,  0,  1,  0,  0,  1,  1],
#       [ 0,  1,  0,  0, 43,  0,  0,  0,  0,  1],
#       [ 0,  0,  0,  0,  0, 43,  0,  0,  1,  2],
#       [ 0,  0,  0,  0,  0,  0, 45,  0,  0,  0],
#       [ 0,  0,  0,  0,  0,  0,  0, 44,  0,  1],
#       [ 0,  3,  0,  0,  1,  0,  0,  0, 39,  0],
#       [ 0,  1,  0,  0,  0,  1,  0,  0,  1, 42]])
```

By definition a confusion matrix $$C$$ is such that $$C_{i, j}$$ is equal to the number of observations known to be in group $$i$$ but predicted to be in group $$j$$. Thus in binary classification, the count of true negatives is $$C_{0,0}$$, false negatives is $$C_{1,0}$$, true positives is $$C_{1,1}$$ and false positives is $$C_{0,1}$$.

We can see that most entries are on the diagonal, which means that we predicted nearly all samples correctly. The off-diagonal entries show us that many eights were classified as ones, and that nines are likely to be confused with many other classes.

Another useful function is the `classification_report` which provides precision, recall, fscore and support for all classes. Precision is how many of the predictions for a class are actually that class. With TP, FP, TN, FN standing for "true positive", "false positive", "true negative" and "false negative" respectively. The values of all these values above are in the closed interval [0, 1], where 1 means a perfect score.

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_test_pred))
##The result:
#               precision    recall  f1-score   support
#
#          0       1.00      1.00      1.00        45
#          1       0.89      0.87      0.88        46
#          2       1.00      0.95      0.98        44
#          3       0.98      0.93      0.96        46
#          4       0.98      0.96      0.97        45
#          5       0.96      0.93      0.95        46
#          6       1.00      1.00      1.00        45
#          7       1.00      0.98      0.99        45
#          8       0.81      0.91      0.86        43
#          9       0.88      0.93      0.90        45
#
#avg / total       0.95      0.95      0.95       450
```

These metrics are helpful in two particular cases that come up often in practice: (a) Imbalanced classes, that is one class might be much more frequent than the other. (b) Asymmetric costs, that is one kind of error is much more "costly" than the other. `accuracy` is simply not a good way to evaluate classifiers for **imbalanced** datasets!

A much better measure for binary classification is using the so-called **ROC** (Receiver operating characteristics) curve. A roc-curve works with uncertainty outputs of a classifier, say the "decision_function" of the SVC we trained above. Instead of making a cut-off at zero and looking at classification outcomes, it looks at every possible cut-off and records how many true positive predictions there are, and how many false positive predictions there are.

For doing grid-search and cross-validation, we usually want to condense our model evaluation into a single number. A good way to do this with the *roc curve* is to use the area under the curve (AUC). We can simply use this in `cross_val_score` by specifying `scoring="roc_auc"`.


## <a name="In Depth - Linear Models">In Depth - Linear Models</a>

Linear models are useful when little data is available or for very large feature spaces as in text classification. In addition, they form a good case study for regularization.

### Linear models for regression

All linear models for regression learn a coefficient parameter `coef_` and an offset `intercept_` to make predictions using a linear combination of features. The difference between the linear models for regression is what kind of restrictions or penalties are put on `coef_` as regularization , in addition to fitting the training data well. The most standard linear model is the 'ordinary least squares regression', often simply called 'linear regression'. It doesn't put any additional restrictions on `coef_`, so when the number of features is large, it becomes ill-posed and the model overfits.

Let us generate a simple simulation, to see the behavior of these models.

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y, true_coefficient = make_regression(n_samples=200, n_feature=30, n_informative=10, noise=100, coef=True, random_state=321)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5, train_size=60)

from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)
print("R^2 on training set: %f" % linear_regression.score(X_train, y_train))
print("R^2 on test set: %f" % linear_regression.score(X_test, y_test))
```

```python
plt.figure(figsize=(10,5))
coefficient_sorting = np.argsort(true_coefficient)[::-1]
plt.plot(true_coefficient[coefficient_sorting], "o", label="true")
plt.plot(linear_regression.coef_[coefficient_sorting], "o", label="linear regression")
plt.legend()
```

```python
from sklearn.model_selection import learning_curve

def plot_learning_curve(est, X, y):
    training_set_size, train_scores, test_scores = learning_curve(est, X, y, train_sizes=np.linspace(0.1,1,20))
    estimator_name = est.__class__.__name__
    line = plt.plot(training_set_size, train_scores.mean(axis=1), "--", label="training scores " + estimator_name)
    plt.plot(training_set_size, test_scores.mean(axis=1), "-", label="test scores " + estimator_name, c=line[0].get_color())
    plt.xlabel("Training set size")
    plt.legend(loc="best")
    plt.ylim(-0.1, 1.1)

plt.figure()
plot_learning_curve(LinearRegression(), X, y)
```

#### Ridge Regression (L2 penalty)

The Ridge estimator is a simple regularization (called l2 penalty) of the ordinary LinearRegression. In particular, it has the benefit of being not computationally more expensive than the ordinary least square estimate. The amount of regularization is set via the `alpha` parameter of the Ridge. Tuning alpha is critical for performance.

```python
from sklearn.linear_model import Ridge

ridge_models = {}
training_scores = []
test_scores = []

for alpha in [100, 10, 1, .01]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    training_scores.append(ridge.score(X_train, y_train))
    test_scores.append(ridge.score(X_test, y_test))
    ridge_models[alpha] = ridge
```

```python
plt.figure()
plt.plot(training_scores, label="training score")
plt.plot(test_scores, label="test scores")
plt.xticks(range(4), [100,10,1,0.1])
plt.legend(loc="best")
```

```python
plt.figure(figsize(10,5))
plt.plot(true_coefficient[coefficient_sorting], "o", label="true", c="b")
for i, alpha in enumerate([100, 10, 1, .01]):
    plt.plot(ridge_models[alpha].coef_[coefficient_sorting], "o", label="alpha=%.2f" % alpha, c=plt.cm.summer(i/3.))
plt.legend(loc="best")
```
#### Lasso Regression (L1 penalty)

The Lasso estimator is useful to impose sparsity on the coefficient. In other words, it is to be preferred if we believe that many of the features are not relevant. This is done via the so-called l1 penalty.

```python
from sklearn.linear_model import Lasso

lasso_models = {}
training_scores = []
test_scores = []

for alpha in [30, 10, 1, .01]:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    training_scores.append(lasso.score(X_train, y_train))
    test_scores.append(lasso.score(X_test, y_test))
    lasso_models[alpha] = lasso
plt.figure()
plt.plot(training_scores, label="training scores")
plt.plot(test_scores, label="test scores")
plt.xticks(range(4), [30, 10, 1, .01])
plt.legend(loc="best")
```

Instead of picking Ridge or Lasso, you can also use **ElasticNet**, which uses both forms of regularization and provides a parameter to assign a weighting between them. ElasticNet typically performs the best amongst these models.

### Linear models for classification

All linear models for classification learn a coefficient parameter `coef_` and an offset `intercept_` to make predictions using a linear combination of features. As you can see, this is very similar to regression, only that a threshold at zero is applied.

Again, the difference among the linear models for classification depends on what kind of regularization is put on `coef_` and `intercept_`. There are also minor differences in how the fit to the training set is measured (the so-called loss function).

The two most common models for linear classification are the linear SVM as implemented in `LinearSVC` and `LogisticRegression`.

#### The influence of C in LinearSVC

In LinearSVC, the `C` parameter controls the regularization within the model. Another way of thinking of `C` is it gives you control of how the SVM will handle errors. If we set `C` to positive infinite, we will get the same result as the Hard Margin SVM. On the contrary, if we set `C` to 0, there will be no constraint anymore, and we will end up with a hyperplane not classifying anything. The rules of thumb are: small values of `C` will result in a wider margin, at the cost of some misclassifications; large values of `C` will give you the Hard Margin classifier and tolerates zero constraint violation. We need to find a value of `C` which will not make the solution be impacted by the noisy data.

Similar to the Ridge/Lasso separation, you can set the `penalty` parameter to 'l1' to enforce sparsity of the coefficients (similar to Lasso) or 'l2' to encourage smaller coefficients (similar to Ridge).

### Multi-class linear classification

```python
from sklearn.datasets import make_blobs
plt.figure()
X, y = make_blobs(random_state=321)

from sklearn.svm import LinearSVC
linear_svm = LinearSVC()
linear_svm.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y)
line = np.linspace(-15, 15)
for coef, intercept in zip(linear_svm.coef_, linear_svm.intercept_):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1])
plt.ylim(-10, 15)
plt.xlim(-10, 8)
```

Points are classified in a one-vs-rest fashion (aka one-vs-all), where we assign a test point to the class whose model has the highest confidence (in the SVM case, highest distance to the separating hyperplane) for the test point.

## <a name="In Depth - Support Vector Machines">In Depth - Support Vector Machines</a>

SVM stands for "support vector machines". They are efficient and easy to use estimators. They come in two kinds: SVCs, Support Vector Classifiers, for classification problems, and SVRs, Support Vector Regressors, for regression problems.

### Linear SVMs

The SVM module contains LinearSVC, which we already discussed briefly in the section on linear models. Using `SVC(kernel="linear")`` will also yield a linear predictor that is only different in minor technical aspects.

### Kernel SVMs

The real power of SVMs lies in using kernels, which allow for non-linear decision boundaries. A kernel defines a similarity measure between data points. The most common are:

* `linear` will give linear decision frontiers. It is the most computationally efficient approach and the one that requires the least amount of data.
* `poly` will give decision frontiers that are polynomial. The order of this polynomial is given by the `'order'` argument.
* `rbf` uses 'radial basis functions' centered at each support vector to assemble a decision frontier. The size of the RBFs ultimately controls the smoothness of the decision frontier. RBFs are the most flexible approach, but also the one that will require the largest amount of data.

The most important parameter of the SVM is the regularization parameter `C`, which bounds the influence of each individual sample:

* Low C values: many support vectors... Decision frontier = mean(class A) - mean(class B)
* High C values: small number of support vectors: Decision frontier fully driven by most discriminant samples

The other important parameters are those of the kernel. Let's look at the RBF kernel in more detail. The rbf kernel has an inverse bandwidth-parameter gamma, where large gamma mean a very localized influence for each data point, and small values mean a very global influence.


## <a name="In Depth - Decision Trees and Forests">In Depth - Decision Trees and Forests</a>

Here we'll explore a class of algorithms based on decision trees. Decision trees at their root are extremely intuitive. They encode a series of "if" and "else" choices, similar to how a person might make a decision. However, which questions to ask, and how to proceed for each answer is entirely learned from the data. The binary splitting of questions is the essence of a decision tree.

One of the main benefit of tree-based models is that they require little preprocessing of the data. They can work with variables of different types (continuous and discrete) and are invariant to scaling of the features.

Another benefit is that tree-based models are what is called "nonparametric", which means they don't have a fix set of parameters to learn. Instead, a tree model can become more and more flexible, if given more data. In other words, the number of free parameters grows with the number of samples and is not fixed, as for example in linear models.

### Decision Tree Regression

A decision tree is a simple binary classification tree that is similar to nearest neighbor classification. It can be used as follows:

```python
from figures import make_dataset
x, y = make_dataset()
X = x.reshape(-1, 1)

from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(max_depth=5)
reg.fit(X, y)
X_fit = np.linspace(-3, 3, 1000).reshape((-1, 1))
y_fit_1 = reg.predict(X_fit)
plt.plot(X_fit.ravel(), y_fit_1, color='blue', label="prediction")
plt.plot(X.ravel(), y, '.k', label="training data")
plt.legend(loc="best")
```

A single decision tree allows us to estimate the signal in a non-parametric way, but clearly has some issues. In some regions, the model shows high bias and under-fits the data. (seen in the long flat lines which don't follow the contours of the data), while in other regions the model shows high variance and over-fits the data (reflected in the narrow spikes which are influenced by noise in single points).

### Decision Tree Classification

Decision tree classification work very similarly, by assigning all points within a leaf the majority class in that leaf:

```python
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from figures import plot_2d_separator

X, y = make_blobs(centers=[[0,0], [1,1]], random_state=321, n_samples=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=321)
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)
plot_2d_separator(clf, X, fill=True)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=60, alpha=.7)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=60);
```

There are many parameter that control the complexity of a tree, but the one that might be easiest to understand is the maximum depth. This limits how finely the tree can partition the input space, or how many "if-else" questions can be asked before deciding which class a sample lies in. This parameter is important to tune for trees and tree-based models.

Decision trees are fast to train, easy to understand, and often lead to interpretable models. However, single trees often tend to overfit the training data. Therefore, in practice it is more common to combine multiple trees to produce models that generalize better. The most common methods for combining trees are random forests and gradient boosted trees.

### Random Forests

Random forests are simply many trees, built on different random subsets (drawn with replacement) of the data, and using different random subsets (drawn without replacement) of the features for each split. This makes the trees different from each other, and makes them overfit to different aspects. Then, their predictions are averaged, leading to a smoother estimate that overfits less.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

rf = RandomForestClassifier(n_estimators=200)
parameters = {"max_features": ["sqrt", "log2", 10], "max_depth": [5, 7, 9]}
clf_grid = GridSearchCV(rf, param_grid=parameters, n_jobs=-1)
clf_grid.fit(X_train, y_train)
```

### Gradient Boosting

Another Ensemble method that can be useful is Boosting: here, rather than looking at 200 (say) parallel estimators, We construct a chain of 200 estimators which iteratively refine the results of the previous estimator. The idea is that by sequentially applying very fast, simple models, we can get a total model error which is better than any of the individual pieces.

```python
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.2)
clf.fit(X_train,y_train)
```

## <a name="Feature Selection">Feature Selection</a>

Often we collected many features that might be related to a supervised prediction task, but we don't know which of them are actually predictive. To improve interpretability, and sometimes also generalization performance, we can use automatic feature selection to select a subset of the original features. There are several types of feature selection methods available, which we'll explain in order of increasing complexity.

For a given supervised model, the best feature selection strategy would be to try out each possible subset of the features, and evaluate generalization performance using this subset. However, there are exponentially many subsets of features, so this exhaustive search is generally infeasible. The strategies discussed below can be thought of as proxies for this infeasible computation.

### Univariate statistics

The simplest method to select features is using univariate statistics, that is by looking at each feature individually and running a statistical test to see whether it is related to the target.

We create a synthetic dataset that consists of the breast cancer data with an additional 50 completely random features.

```python
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
rng = np.random.RandomState(321)
noise = rng.normal(size=(len(cancer.data), 50))
# add noise features to the data
# the first 30 features are from the dataset, the next 50 are noise
X_w_noise = np.hstack([cancer.data, noise])
X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target, random_state=0, test_size=.5)
```

We have to define a threshold on the p-value of the statistical test to decide how many features to keep. There are several strategies implemented in scikit-learn, a straight-forward one being SelectPercentile, which selects a percentile of the original features (we select 50% below):

```python
from sklearn.feature_selection import SelectPercentile
# use f_classif (the default) and SelectPercentile to select 50% of features:
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
# transform training set:
X_train_selected = select.transform(X_train)
```

We can also use the test statistic directly to see how relevant each feature is. As the breast cancer dataset is a classification task, we use `f_classif`, the F-test for classification. Below we plot the p-values associated with each of the 80 features (30 original features + 50 noise features). Low p-values indicate informative features.

```python
from sklearn.feature_selection import f_classif, f_regression, chi2
F, p = f_classif(X_train, y_train)
plt.figure()
plt.plot(p, 'o')
```

Clearly most of the first 30 features have very small p-values.

Going back to the SelectPercentile transformer, we can obtain the features that are selected using the `get_support` method:

```python
mask = select.get_support()
print(mask)
# visualize the mask. black is True, white is False
plt.matshow(mask.reshape(1,-1), cmap="gray_r")
```

Nearly all of the original 30 features were recovered. We can also analize the utility of the feature selection by training a supervised model on the data. It's important to learn the feature selection only on the **training set**!

### Model-based feature selection

A somewhat more sophisticated method for feature selection is using a supervised machine learning model and selecting features based on how important they were deemed by the model. This requires the model to provide some way to rank the features by importance. This can be done for all **tree-based models** (which implement `get_feature_importances`) and all **linear models**, for which the coefficients can be used to determine how much influence a feature has on the outcome.

Any of these models can be made into a transformer that does feature selection by wrapping it with the `SelectFromModel` class:

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=321), threshold="median")
select.fit(X_train, y_train)
X_train_rf = select.transform(X_train)
print(X_train.shape)
print(X_train_rf.shape)

mask = select.get_support()
# visualize the mask. black is True, white is False
plt.matshow(mask.reshape(1, -1), cmap="gray_r")

X_test_rf = select.transform(X_test)
LogisticRegression().fit(X_train_rf, y_train).score(X_test_rf, y_test)
```

This method builds a single model (in this case a random forest) and uses the feature importances from this model. We can do a somewhat more elaborate search by training multiple models on subsets of the data. One particular strategy is recursive feature elimination:

### Recursive Feature Elimination

Recursive feature elimination builds a model on the full set of features, and similar to the method above selects a subset of features that are deemed most important by the model. However, usually only a single feature is dropped from the dataset, and a new model is built with the remaining features. The process of dropping features and model building is repeated until there are only a pre-specified number of features left:

```python
from sklearn.feature_selection import RFE
select = RFE(RandomForestClassifier(n_estimators=100, random_state=321), n_features_to_select=40)
select.fit(X_train, y_train)
# visualize the selected features:
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap="gray_r")
X_train_rfe = select.transform(X_train)
X_test_rfe = select.transform(X_test)

LogisticRegression().fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
```


## <a name="Unsupervised learning - Hierarchical and density-based clustering algorithms">Unsupervised learning - Hierarchical and density-based clustering algorithms</a>

Previously, we introduced one of the essential and widely used clustering algorithms, K-means. One of the advantages of K-means is that it is extremely easy to implement, and it is also computationally very efficient compared to other clustering algorithms. However, we've seen that one of the weaknesses of K-Means is that it only works well if the data can be grouped into a globular or spherical shape. Also, we have to assign the number of clusters, k, a priori -- this can be a problem if we have no prior knowledge about how many clusters we expect to find. In this section, we will take a look at two alternative approaches to clustering, **hierarchical clustering** and **density-based clustering**.

### Hierarchical Clustering

One nice feature of hierarchical clustering is that we can visualize the results as a dendrogram, a hierarchical tree. Using the visualization, we can then decide how "deep" we want to cluster the dataset by setting a `"depth"` threshold. Or in other words, we don't need to make a decision about the number of clusters upfront.

#### Agglomerative and divisive hierarchical clustering

Furthermore, we can distinguish between 2 main approaches to hierarchical clustering: Divisive clustering and agglomerative clustering. In agglomerative clustering, we start with a single sample from our dataset and iteratively merge it with other samples to form clusters -- we can see it as a bottom-up approach for building the clustering dendrogram.
In divisive clustering, however, we start with the whole dataset as one cluster, and we iteratively split it into smaller subclusters -- a top-down approach. In this section, we will use **agglomerative** clustering.

#### Single and complete linkage

Now, the next question is how we measure the similarity between samples. One approach is the familiar Euclidean distance metric that we already used via the K-Means algorithm.

However, that's the distance between 2 samples. Now, how do we compute the similarity between subclusters of samples in order to decide which clusters to merge when constructing the dendrogram? I.e., our goal is to iteratively merge the most similar pairs of clusters until only one big cluster remains. There are many different approaches to this, for example single and complete linkage.

In **single linkage**, we take the pair of the most similar samples (based on the Euclidean distance, for example) in each cluster, and merge the two clusters which have the most similar 2 members into one new, bigger cluster.

In **complete linkage**, we compare the pairs of the two most dissimilar members of each cluster with each other, and we merge the 2 clusters where the distance between its 2 most dissimilar members is smallest.

![sklearn13](/mldl/assets/images/2018-06-29/sklearn13.jpg)

To see the agglomerative, hierarchical clustering approach in action, let us load the familiar Iris dataset -- pretending we don't know the true class labels and want to find out how many different follower species it consists of:

```python
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
n_samples, n_features = X.shape
plt.scatter(X[:, 0], X[:, 1], c=y)
```

First, we start with some exploratory clustering, visualizing the clustering dendrogram using SciPy's `linkage` and `dendrogram` functions:

```python
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

clusters = linkage(X, metric="euclidean", method="complete")
dendr = dendrogram(clusters)
plt.ylabel("Euclidean Distance")
```

Next, let's use the `AgglomerativeClustering` estimator from scikit-learn and divide the dataset into 3 clusters.

```python
from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="complete")
prediction = ac.fit_predict(X)
print('Cluster labels: %s\n' % prediction)
plt.scatter(X[:, 0], X[:, 1], c=prediction);
```

### Density-based Clustering - DBSCAN

Another useful approach to clustering is Density-based Spatial Clustering of Applications with Noise (DBSCAN). In essence, we can think of DBSCAN as an algorithm that divides the dataset into subgroup based on dense regions of point.

In DBSCAN, we distinguish between 3 different "points":

* Core points: A core point is a point that has at least a minimum number of other points (MinPts) in its radius epsilon.
* Border points: A border point is a point that is not a core point, since it doesn't have enough MinPts in its neighborhood, but lies within the radius epsilon of a core point.
* Noise points: All other points that are neither core points nor border points.

![sklearn14](/mldl/assets/images/2018-06-29/sklearn14.jpg)

A nice feature about DBSCAN is that we don't have to specify a number of clusters upfront. However, it requires the setting of additional hyperparameters such as the value for MinPts and the radius epsilon.

```python
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=400, noise=0.1, random_state=321)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.2, min_samples=10, metric="euclidean")
prediction = df.fit_predict(X)
print("Predicted labels:\n", prediction)
plt.scatter(X[:, 0], X[:, 1], c=prediction);
```

## <a name="Unsupervised learning - Non-linear dimensionality reduction (Manifold Learning)">Unsupervised learning - Non-linear dimensionality reduction (Manifold Learning)</a>

One weakness of PCA is that it cannot detect non-linear features. A set of algorithms known as **Manifold Learning** have been developed to address this deficiency. A canonical dataset used in Manifold learning is the **S-curve**:

```python
from sklearn.datasets import make_s_curve
X, y = make_s_curve(n_samples=1000)

from mpl_toolkits.mplot3d import Axes3D
ax = plt.axes(projection="3d")
ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=y)
ax.view_init(10, -60)
```

This is a 2-dimensional dataset embedded in three dimensions, but it is embedded in such a way that PCA cannot discover the underlying data orientation:

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
```

Manifold learning algorithms, however, available in the `sklearn.manifold` submodule, are able to recover the underlying 2-dimensional manifold:

```python
from sklearn.manifold import Isomap
iso = Isomap(n_neighbors=15, n_components=2)
X_iso = iso.fit_transform(X)
plt.scatter(X_iso[:, 0], X_iso[:, 1], c=y);
```

We can apply manifold learning techniques to much higher dimensional datasets, for example the digits data that we saw before:

```python
from sklearn.datasets import load_digits
digits  = load_digits()
fig, axes = plt.subplots(2, 5, figsize(10,5), subplot_kw={'xticks':(), 'yticks': ()})
for ax, img in zip(axes.ravel(), digits.images):
    ax.imshow(img, interpolation="none", cmap="gray")
```

We can visualize the dataset using a linear technique, such as PCA. We saw this already provides some intuition about the data:

```python
pca = PCA(n_components=2)
pca.fit(digits.data)
# transform the digits data onto the first two principal components
digits_pca = pca.transform(digits.data)
colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
          "#A83683", "#4E655E", "#853541", "#3A3120","#535D8E"]
plt.figure(figsize=(10,10))
plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max() + 1)
plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max() + 1)
for i in range(len(digits.data)):
    # actually plot the digits as text instead of using scatter
    plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]), color = colors[digits.target[i]], fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("first principal component")
plt.ylabel("second principal component")
```

Using a more powerful, nonlinear techinque can provide much better visualizations, though. Here, we are using the t-SNE manifold learning method:

```python
from sklearn.manifold import TSNE
tsne = TSNE(random_state=321)
# use fit_transform instead of fit, as TSNE has no transform method:
digits_tsne = tsne.fit_transform(digits.data)
plt.figure(figsize=(10, 10))
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1)
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)
for i in range(len(digits.data)):
    # actually plot the digits as text instead of using scatter
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]),
             color = colors[digits.target[i]],
             fontdict={'weight': 'bold', 'size': 9})
```

t-SNE has a somewhat longer runtime that other manifold learning algorithms, but the result is quite striking. Keep in mind that this algorithm is purely unsupervised, and does not know about the class labels. Still it is able to separate the classes very well (though the classes four, one and nine have been split into multiple groups).


## <a name="Out-of-core Learning Large Scale Text Classification for Sentiment Analysis">Out-of-core Learning Large Scale Text Classification for Sentiment Analysis</a>

### Scalability issues

The `sklearn.feature_extraction.text.CountVectorizer` and `sklearn.feature_extraction.text.TfidfVectorizer` classes suffer from a number of scalability issues that all stem from the internal usage of the `vocabulary_` attribute (a Python dictionary) used to map the unicode string feature names to the integer feature indices.

The main scalability issues are:

* **Memory usage of the text vectorizer**: all the string representations of the features are loaded in memory
* **Parallelization problems for text feature extraction**: the vocabulary_ would be a shared state: complex synchronization and overhead
* **Impossibility to do online or out-of-core / streaming learning**: the vocabulary_ needs to be learned from the data: its size cannot be known before making one pass over the full dataset

To better understand the issue let's have a look at how the `vocabulary_` attribute work. At `fit` time the tokens of the corpus are uniquely identified by an integer index and this mapping is stored in the vocabulary:

```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)
vectorizer.fit(["The cat sat on the mat."])
vectorizer.vocabulary_  # result: {'cat': 0, 'mat': 1, 'on': 2, 'sat': 3, 'the': 4}
```

The vocabulary is used at `transform` time to build the occurrence matrix:

```python
X = vectorizer.transform(["The cat sat on the mat.", "This cat is a nice cat."]).toarray() # result: array([[1, 1, 1, 1, 2], [2, 0, 0, 0, 0]])
print(len(vectorizer.vocabulary_))  # result: 5
print(vectorizer.get_feature_names())  # result: ['cat', 'mat', 'on', 'sat', 'the']
```

Let's refit with a slightly larger corpus:

```python
vectorizer = CountVectorizer(min_df=1)
vectorizer.fit(["The cat sat on the mat.", "The quick brown fox jumps over the lazy dog."])
vectorizer.vocabulary_  # result: {'mat': 6, 'on': 7, 'brown': 0, 'fox': 3, 'cat': 1, 'the': 11, 'dog': 2, 'sat': 10, 'quick': 9, 'jumps': 4, 'lazy': 5, 'over': 8}
```

The `vocabulary_` is (logarithmically) growing with the size of the training corpus. Note that we could not have built the vocabularies in parallel on the 2 text documents as they share some words hence would require some kind of shared datastructure or synchronization barrier which is complicated to setup, especially if we want to distribute the processing on a cluster.

With this new vocabulary, the dimensionality of the output space is now larger:

```python
X = vectorizer.transform(["The cat sat on the mat.", "This cat is a nice cat."]).toarray()  # result: array([[0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 2], [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
print(len(vectorizer.vocabulary_))  # result: 12
print(vectorizer.get_feature_names())  # result: ['brown', 'cat', 'dog', 'fox', 'jumps', 'lazy', 'mat', 'on', 'over', 'quick', 'sat', 'the']
```

### The IMDb movie dataset

To illustrate the scalability issues of the vocabulary-based vectorizers, let's load a more realistic dataset for a classical text classification task: sentiment analysis on text documents. The goal is to tell apart negative from positive movie reviews from the Internet Movie Database (IMDb).

This dataset contains 50,000 movie reviews, which were split into 25,000 training samples and 25,000 test samples. The reviews are labeled as either negative (neg) or positive (pos). Moreover, positive means that a movie received > 6 stars on IMDb; negative means that a movie received < 5 stars, respectively.

```python
import os
train_path = os.path.join('datasets', 'IMDb', 'aclImdb', 'train')
test_path = os.path.join('datasets', 'IMDb', 'aclImdb', 'test')
from sklearn.datasets import load_files
train = load_files(container_path=(train_path),
                   categories=['pos', 'neg'])

test = load_files(container_path=(test_path),
                  categories=['pos', 'neg'])
train.keys()  # return: dict_keys(['target_names', 'data', 'DESCR', 'filenames', 'target'])
```

Since the movie datasets consists of 50,000 individual text files, executing the code snippet above may take ~20 sec or longer. The `load_files` function loaded the datasets into `sklearn.datasets.base.Bunch` objects, which are Python dictionaries. In particular, we are only interested in the `data` and `target` arrays.

```python
import numpy as np

for label, data in zip(('TRAINING', 'TEST'), (train, test)):
    print('\n\n%s' % label)
    print('Number of documents:', len(data['data']))
    print('\n1st document:\n', data['data'][0])
    print('\n1st label:', data['target'][0])
    print('\nClass names:', data['target_names'])
    print('Class count:',
          np.unique(data['target']), ' -> ',
          np.bincount(data['target']))
```

As we can see above the `'target'` array consists of integers 0 and 1, where 0 stands for negative and 1 stands for positive.

### The Hashing Trick

Remember the bag of word representation using a vocabulary based vectorizer:

![sklearn15](/mldl/assets/images/2018-06-29/sklearn15.jpg)

To workaround the limitations of the vocabulary-based vectorizers, one can use the hashing trick. Instead of building and storing an explicit mapping from the feature names to the feature indices in a Python dict, we can just use a hash function and a modulus operation:

![sklearn16](/mldl/assets/images/2018-06-29/sklearn16.jpg)

The `HashingVectorizer` class is an alternative to the `CountVectorizer` (or `TfidfVectorizer` class with `use_idf=False`) that internally uses the murmurhash hash function:

```python
from sklearn.feature_extraction .text import HashingVectorizer

h_vectorizer = HashingVectorizer(encoding='latin-1')
```

It shares the same "preprocessor", "tokenizer" and "analyzer" infrastructure. We can vectorize our datasets into a scipy sparse matrix exactly as we would have done with the `CountVectorizer` or `TfidfVectorizer`, except that we can directly call the `transform` method: there is no need to `fit` as `HashingVectorizer` is a stateless transformer:

```
docs_train, y_train = train['data'], train['target']
docs_valid, y_valid = test['data'][:12500], test['target'][:12500]
docs_test, y_test = test['data'][12500:], test['target'][12500:]
h_vectorizer.transform(docs_train) # returns: <25000x1048576 sparse matrix of type '<class 'numpy.float64'> with 3446628 stored elements in Compressed Sparse Row format>
```

The dimension of the output is fixed ahead of time to **n_features=2 ** 20** by default (nearly 1M features) to minimize the rate of collision on most classification problem while having reasonably sized linear models (1M weights in the coef_ attribute).

Finally, let us train a LogisticRegression classifier on the IMDb training subset:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

h_pipeline = Pipeline((("vec", HashingVectorizer(encoding="latin-1")),("clf", LogisticRegression(random_state=1))))
h_pipeline.fit(docs_train, y_train)

print('Train accuracy', h_pipeline.score(docs_train, y_train))  # Train accuracy 0.87848
print('Validation accuracy', h_pipeline.score(docs_valid, y_valid))  #  Validation accuracy 0.86208
```

### Out-of-Core learning

Out-of-Core learning is the task of training a machine learning model on a dataset that does not fit into memory or RAM. This requires the following conditions:

* a **feature extraction** layer with **fixed output dimensionality**
* knowing the list of all classes in advance (in this case we only have positive and negative tweets)
* a machine learning **algorithm that supports incremental learning** (the `partial_fit` method in scikit-learn).

In the following sections, we will set up a simple batch-training function to train an `SGDClassifier` iteratively.

```python
train_path = os.path.join('datasets', 'IMDb', 'aclImdb', 'train')
train_pos = os.path.join(train_path, 'pos')
train_neg = os.path.join(train_path, 'neg')
fnames = [os.path.join(train_pos, f) for f in os.listdir(train_pos)] +\
         [os.path.join(train_neg, f) for f in os.listdir(train_neg)]
y_train = np.zeros((len(fnames), ), dtype=int)
y_train[:12500] = 1
```

Now, we implement the `batch_train` function as follows:

```python
from sklearn.base import clone

def batch_train(clf, fnames, labels, iterations=25, batchsize=1000, random_seed=1)
    vec = HashingVectorizer(encoding='latin-1')
    idx = np.arange(labels.shape[0])
    c_clf = clone(clf)
    rng = np.random.RandomState(seed=random_seed)

    for i in range(iterations):
        rnd_idx = rng.choice(idx, size=batchsize)
        documents = []
        for i in rnd_idx:
            with open(fnames[i], 'r') as f:
                documents.append(f.read())
        X_batch = vec.transform(documents)
        batch_labels = labels[rnd_idx]
        c_clf.partial_fit(X=X_batch,
                          y=batch_labels,
                          classes=[0, 1])

    return c_clf
```

Note that we are not using `LogisticRegression` as in the previous section, but we will use a `SGDClassifier` with a logistic cost function instead. SGD stands for stochastic gradient descent, an optimization algorithm that optimizes the weight coefficients iteratively sample by sample, which allows us to feed the data to the classifier chunk by chuck.

And we train the `SGDClassifier`; using the default settings of the `batch_train` function, it will train the classifier on 25*1000=25000 documents. (Depending on your machine, this may take >2 min)

```python
from sklearn.linear_model import SGDClassifier
​
sgd = SGDClassifier(loss='log', random_state=1)
​
sgd = batch_train(clf=sgd,
                  fnames=fnames,
                  labels=y_train)
vec = HashingVectorizer(encoding='latin-1')
sgd.score(vec.transform(docs_test), y_test)
```

Using the Hashing Vectorizer makes it possible to implement streaming and parallel text classification but can also introduce some issues:

* The collisions can introduce too much noise in the data and degrade prediction quality,
* The `HashingVectorizer` does not provide "Inverse Document Frequency" reweighting (lack of a `use_idf=True` option).
* There is no easy way to inverse the mapping and find the feature names from the feature index.

The collision issues can be controlled by increasing the `n_features` parameters.

The IDF weighting might be reintroduced by appending a `TfidfTransformer` instance on the output of the vectorizer. However computing the `idf_` statistic used for the feature reweighting will require to do at least one additional pass over the training set before being able to start training the classifier: this breaks the online learning scheme.

The lack of inverse mapping (the `get_feature_names()` method of `TfidfVectorizer`) is even harder to workaround. That would require extending the `HashingVectorizer` class to add a "trace" mode to record the mapping of the most important features to provide statistical debugging information.

In the mean time to debug feature extraction issues, it is recommended to use `TfidfVectorizer(use_idf=False)` on a small-ish subset of the dataset to simulate a `HashingVectorizer()` instance that have the `get_feature_names()` method and no collision issues.

References:

* [<u>SciPy 2016 Scikit-learn Tutorial</u>](https://github.com/amueller/scipy-2016-sklearn)
* [<u>Python Numpy Tutorial</u>](http://cs231n.github.io/python-numpy-tutorial/#numpy)
