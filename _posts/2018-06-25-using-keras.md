---
layout:         post
title:          Using Keras
subtitle:
card-image:     /mldl/assets/images/cards/cat16.gif
date:           2018-06-25 09:00:00
tags:           [deep&nbsp;learning]
categories:     [deep&nbsp;learning, keras]
post-card-type: image
mathjax:        true
---

* <a href="#Getting started: 30 seconds to Keras">Getting started: 30 seconds to Keras</a>
* <a href="#Getting started with the Keras Sequential model">Getting started with the Keras Sequential model</a>
    * <a href="#Specifying the input shape">Specifying the input shape</a>
    * <a href="#Compilation">Compilation</a>
    * <a href="#Training">Training</a>
    * <a href="#Examples">Examples</a>
* <a href="#Use the Keras Functional API for Deep Learning">Use the Keras Functional API for Deep Learning</a>
    * <a href="#Three aspects of Keras functional API">Three aspects of Keras functional API</a>
    * <a href="#Standard Network Models">Standard Network Models</a>
    * <a href="#Shared Layers Model">Shared Layers Model</a>
    * <a href="#Multiple Input and Output Models">Multiple Input and Output Models</a>
    * <a href="#Best Practices">Best Practices</a>
* <a href="#Usage of Callbacks">Usage of Callbacks</a>

Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:

* Allows for easy and fast prototyping (through user friendliness, modularity, and extensibility).
* Supports both convolutional networks and recurrent networks, as well as combinations of the two.
* Runs seamlessly on CPU and GPU.

## <a name="Getting started: 30 seconds to Keras">Getting started: 30 seconds to Keras</a>

The core data structure of Keras is a model, a way to organize layers. The simplest type of model is the [`Sequential`](https://keras.io/getting-started/sequential-model-guide/) model, a linear stack of layers. For more complex architectures, you should use the [`Keras functional`](https://keras.io/getting-started/functional-api-guide/) API, which allows to build arbitrary graphs of layers.

Here is the Sequential model:

```python
from keras.models import Sequential

model = Sequential()
```

Stacking layers is as easy as `.add()`:

```python
from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
```

Once your model looks good, configure its learning process with `.compile()`:

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

If you need to, you can further configure your optimizer. A core principle of Keras is to make things reasonably simple, while allowing the user to be fully in control when they need to (the ultimate control being the easy extensibility of the source code).

```python
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
```

You can now iterate on your training data in batches:

```python
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

Alternatively, you can feed batches to your model manually:

```python
model.train_on_batch(x_batch, y_batch)
```

Evaluate your performance in one line:

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

Or generate predictions on new data:

```python
classes = model.predict(x_test, batch_size=128)
```

Building a question answering system, an image classification model, a Neural Turing Machine, or any other model is just as fast. The ideas behind deep learning are simple, so why should their implementation be painful?

For a more in-depth tutorial about Keras, you can check out:

* [<u>Getting started with the Sequential model</u>](https://keras.io/getting-started/sequential-model-guide/)
* [<u>Getting started with the functional API</u>](https://keras.io/getting-started/functional-api-guide/)

In the [<u>examples folder</u>](https://github.com/keras-team/keras/tree/master/examples) of the repository, you will find more advanced models: question-answering with memory networks, text generation with stacked LSTMs, etc.

## <a name="Getting started with the Keras Sequential model">Getting started with the Keras Sequential model</a>

The [<u>Sequential</u>](https://keras.io/models/sequential/) model is a linear stack of layers.

You can create a [<u>Sequential</u>](https://keras.io/models/sequential/) model by passing a list of layer instances to the constructor:

```python
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
```

You can also simply add layers via the .add() method:

```python
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
```

### <a name="Specifying the input shape">Specifying the input shape</a>

The model needs to know what input shape it should expect. For this reason, the first layer in a Sequential model (and only the first, because following layers can do automatic shape inference) needs to receive information about its input shape. There are several possible ways to do this:

* Pass an `input_shape` argument to the first layer. This is a shape tuple (a tuple of integers or `None` entries, where `None` indicates that any positive integer may be expected). In `input_shape`, the batch dimension is not included.
* Some 2D layers, such as `Dense`, support the specification of their input shape via the argument `input_dim`, and some 3D temporal layers support the arguments `input_dim` and `input_length`.
* If you ever need to specify a fixed batch size for your inputs (this is useful for stateful recurrent networks), you can pass a batch_size argument to a layer. If you pass both batch_size=32 and input_shape=(6, 8) to a layer, it will then expect every batch of inputs to have the batch shape (32, 6, 8).

As such, the following snippets are strictly equivalent:

```python
model = Sequential()
model.add(Dense(32, input_shape=(784,)))
```

```python
model = Sequential()
model.add(Dense(32, input_dim=784))
```

### <a name="Compilation">Compilation</a>

Before training a model, you need to configure the learning process, which is done via the `compile` method. It receives three arguments:

* An [<u>optimizer</u>](https://keras.io/optimizers/). This could be the string identifier of an existing optimizer (such as `rmsprop` or `adagrad`), or an instance of the `Optimizer` class.
* A [<u>loss function</u>](https://keras.io/losses/). This is the objective that the model will try to minimize. It can be the string identifier of an existing loss function (such as `categorical_crossentropy` or `mse`), or it can be an objective function.
* A list of [<u>metrics</u>](https://keras.io/metrics/). For any classification problem you will want to set this to metrics=['accuracy']. A metric could be the string identifier of an existing metric or a custom metric function.

```python
# For a multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# For a binary classification problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# For a mean squared error regression problem
model.compile(optimizer='rmsprop',
              loss='mse')

# For custom metrics
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```

### <a name="Training">Training</a>

Keras models are trained on Numpy arrays of input data and labels. For training a model, you will typically use the `fit` function.

```python
# For a single-input model with 2 classes (binary classification):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32)
```

```python
# For a single-input model with 10 classes (categorical classification):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=10, batch_size=32)
```
### <a name="Examples">Examples</a>

```python
###Multilayer Perceptron (MLP) for multi-class softmax classification:
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
```

```python
### VGG-like convnet:
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

# Generate dummy data
x_train = np.random.random((100, 100, 100, 3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((20, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3))) # input: 100x100x3, output: 98x98x32
model.add(Conv2D(32, (3, 3), activation='relu')) # input: 98x98x32, output: 96x96x32
model.add(MaxPooling2D(pool_size=(2, 2))) # input: 96x96x32, output: 48x48x32
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu')) # input: 48x48x32, output: 46x46x64
model.add(Conv2D(64, (3, 3), activation='relu')) # input: 46x46x64, output: 44x44x64
model.add(MaxPooling2D(pool_size=(2, 2))) # input: 44x44x64, output: 22x22x64
model.add(Dropout(0.25))

model.add(Flatten()) # input/output: 30,976
model.add(Dense(256, activation='relu')) # input: 30,976, output: 256
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax')) # input: 256, output: 10

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)
```

```python
### Sequence classification with LSTM:
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

max_features = 1024

model = Sequential()
# input_dim: Size of the vocabulary
# output_dim: Dimension of the dense embedding.
model.add(Embedding(input_dim=max_features, output_dim=256))
# LSTM layer with 128 memory units/neurons.
model.add(LSTM(128))
model.add(Dropout(0.5))
# Dense output layer with a single neuron.
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)
```

More examples:

* [<u>Sequence Classification with LSTM Recurrent Neural Networks in Python with Keras</u>](https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/)


## <a name="Use the Keras Functional API for Deep Learning">[<u>Use the Keras Functional API for Deep Learning</u>](https://machinelearningmastery.com/keras-functional-api-deep-learning/)</a>

The sequential API allows you to create models layer-by-layer for most problems. It is limited in that it does not allow you to create models that share layers or have multiple inputs or outputs.

The functional API in Keras is an alternate way of creating models that offers a lot more flexibility, including creating more complex models.

In most of the popular neural networks, we have a mini-network(a few layers arranged in a certain way like Inception module in GoogLeNet, fire module in squeezeNet) and this mini-network is repeated multiple times. Functional API allows you to call models like a function as we do for a layer. So, you can build a small model which can be repeated multiple times to build the complete network with even fewer lines of code. By calling a model on a tensor, you can reuse the model as well as weights.

In the next section, you will discover how to use the more flexible functional API in Keras to define deep learning models.

In Keras functional API, models are defined by creating instances of layers and connecting them directly to each other in pairs, then defining a Model that specifies the layers to act as the input and output to the model.

### <a name="Three aspects of Keras functional API">Three aspects of Keras functional API</a>

#### Defining Input

Unlike the Sequential model, you must create and define a standalone Input layer that specifies the shape of input data. The input layer takes a shape argument that is a tuple that indicates the dimensionality of the input data.

When input data is one-dimensional, such as for a multilayer Perceptron, the shape must explicitly leave room for the shape of the mini-batch size used when splitting the data when training the network. Therefore, the shape tuple is always defined with a hanging last dimension when the input is one-dimensional (2,), for example:

```python
from keras.layers import Input
visible = Input(shape=(2,))
```

#### Connecting Layers

The layers in the model are connected pairwise. This is done by specifying where the input comes from when defining each new layer. A bracket notation is used, such that after the layer is created, the layer from which the input to the current layer comes from is specified.

Let’s make this clear with a short example. We can create the input layer as above, then create a hidden layer as a Dense that receives input only from the input layer.

```python
from keras.layers import Input
from keras.layers import Dense
visible = Input(shape=(2,))
hidden = Dense(2)(visible)
```

Note the (visible) after the creation of the Dense layer that connects the input layer output as the input to the dense hidden layer. It is this way of connecting layers piece by piece that gives the functional API its flexibility. For example, you can see how easy it would be to start defining ad hoc graphs of layers.

#### Creating the Model

After creating all of your model layers and connecting them together, you must define the model. As with the Sequential API, the model is the thing you can summarize, fit, evaluate, and use to make predictions.

Keras provides a Model class that you can use to create a model from your created layers. It requires that you only specify the input and output layers. For example:

```python
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
visible = Input(shape=(2,))
hidden = Dense(2)(visible)
model = Model(inputs=visible, outputs=hidden)
```

Now that we know all of the key pieces of the Keras functional API, let’s work through defining a suite of different models and build up some practice with it. Each example is executable and prints the structure and creates a diagram of the graph. I recommend doing this for your own models to make it clear what exactly you have defined. My hope is that these examples provide templates for you when you want to define your own models using the functional API in the future.


### <a name="Standard Network Models">Standard Network Models</a>

When getting started with the functional API, it is a good idea to see how some standard neural network models are defined. In this section, we will look at defining a simple multilayer Perceptron, convolutional neural network, and recurrent neural network. These examples will provide a foundation for understanding the more elaborate examples later.

#### Multilayer Perceptron

In this section, we define a multilayer Perceptron model for binary classification. The model has 10 inputs, 3 hidden layers with 10, 20, and 10 neurons, and an output layer with 1 output. Rectified linear activation functions are used in each hidden layer and a sigmoid activation function is used in the output layer, for binary classification.

```python
# Multilayer Perceptron
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
visible = Input(shape=(10,))
hidden1 = Dense(10, activation='relu')(visible)
hidden2 = Dense(20, activation='relu')(hidden1)
hidden3 = Dense(10, activation='relu')(hidden2)
output = Dense(1, activation='sigmoid')(hidden3)
model = Model(inputs=visible, outputs=output)
# summarize layers
print(model.summary())
```

Running the example prints the structure of the network.

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 10)                0
_________________________________________________________________
dense_1 (Dense)              (None, 10)                110
_________________________________________________________________
dense_2 (Dense)              (None, 20)                220
_________________________________________________________________
dense_3 (Dense)              (None, 10)                210
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 551
Trainable params: 551
Non-trainable params: 0
_________________________________________________________________
```

#### Convolutional Neural Network

In this section, we will define a convolutional neural network for image classification. The model receives black and white 64×64 images as input, then has a sequence of two convolutional and pooling layers as feature extractors, followed by a fully connected layer to interpret the features and an output layer with a sigmoid activation for two-class predictions.

```python
# Convolutional Neural Network
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
visible = Input(shape=(64,64,1))
conv1 = Conv2D(32, kernel_size=4, activation='relu')(visible)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(16, kernel_size=4, activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
hidden1 = Dense(10, activation='relu')(pool2)
output = Dense(1, activation='sigmoid')(hidden1)
model = Model(inputs=visible, outputs=output)
# summarize layers
print(model.summary())
```

Running the example summarizes the model layers.


```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 64, 64, 1)         0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 61, 61, 32)        544
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 30, 30, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 27, 27, 16)        8208
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 13, 13, 16)        0
_________________________________________________________________
dense_1 (Dense)              (None, 13, 13, 10)        170
_________________________________________________________________
dense_2 (Dense)              (None, 13, 13, 1)         11
=================================================================
Total params: 8,933
Trainable params: 8,933
Non-trainable params: 0
_________________________________________________________________
```

Formula to conv net calculate parameters: ([<u>stackoverflow ref</u>](https://stackoverflow.com/questions/44608552/keras-cnn-model-parameters-calculation))

{% raw %}
$$
total\_params = (filter\_height * filter\_width * input\_image\_channels + 1) * number\_of\_filters
$$
{% endraw %}

#### Recurrent Neural Network

In this section, we will define a long short-term memory recurrent neural network for sequence classification. The model expects 100 time steps of one feature as input. The model has a single LSTM hidden layer to extract features from the sequence, followed by a fully connected layer to interpret the LSTM output, followed by an output layer for making binary predictions.

```python
# Recurrent Neural Network
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.recurrent import LSTM
visible = Input(shape=(100,1))
hidden1 = LSTM(10)(visible)
hidden2 = Dense(10, activation='relu')(hidden1)
output = Dense(1, activation='sigmoid')(hidden2)
model = Model(inputs=visible, outputs=output)
# summarize layers
print(model.summary())
```

Running the example summarizes the model layers.

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 100, 1)            0
_________________________________________________________________
lstm_1 (LSTM)                (None, 10)                480
_________________________________________________________________
dense_1 (Dense)              (None, 10)                110
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 11
=================================================================
Total params: 601
Trainable params: 601
Non-trainable params: 0
_________________________________________________________________
```

Formula to LSTM calculate parameters: ([<u>stackoverflow ref</u>](https://datascience.stackexchange.com/questions/10615/number-of-parameters-in-an-lstm-model))

{% raw %}
$$
total\_params = 4(nm+n^2+n) \\
$$
{% endraw %}

where m is input vector size and n is output vector size

### <a name="Shared Layers Model">Shared Layers Model</a>

Multiple layers can share the output from one layer. For example, there may be multiple different feature extraction layers from an input, or multiple layers used to interpret the output from a feature extraction layer. Let’s look at both of these examples.

#### Shared Input Layer

In this section, we define multiple convolutional layers with differently sized kernels to interpret an image input. The model takes black and white images with the size 64×64 pixels. There are two CNN feature extraction submodels that share this input; the first has a kernel size of 4 and the second a kernel size of 8. The outputs from these feature extraction submodels are flattened into vectors and concatenated into one long vector and passed on to a fully connected layer for interpretation before a final output layer makes a binary classification.

```python
# Shared Input Layer
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
# input layer
visible = Input(shape=(64,64,1))
# first feature extractor
conv1 = Conv2D(32, kernel_size=4, activation='relu')(visible)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
flat1 = Flatten()(pool1)
# second feature extractor
conv2 = Conv2D(16, kernel_size=8, activation='relu')(visible)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
flat2 = Flatten()(pool2)
# merge feature extractors
merge = concatenate([flat1, flat2])
# interpretation layer
hidden1 = Dense(10, activation='relu')(merge)
# prediction output
output = Dense(1, activation='sigmoid')(hidden1)
model = Model(inputs=visible, outputs=output)
# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file='shared_input_layer.png')
```

A plot of the model graph is created and saved to file.

![keras1](/mldl/assets/images/2018-06-25/keras1.jpg)

Running the example summarizes the model layers.

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_1 (InputLayer)             (None, 64, 64, 1)     0
____________________________________________________________________________________________________
conv2d_1 (Conv2D)                (None, 61, 61, 32)    544         input_1[0][0]
____________________________________________________________________________________________________
conv2d_2 (Conv2D)                (None, 57, 57, 16)    1040        input_1[0][0]
____________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)   (None, 30, 30, 32)    0           conv2d_1[0][0]
____________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)   (None, 28, 28, 16)    0           conv2d_2[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 28800)         0           max_pooling2d_1[0][0]
____________________________________________________________________________________________________
flatten_2 (Flatten)              (None, 12544)         0           max_pooling2d_2[0][0]
____________________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 41344)         0           flatten_1[0][0]
                                                                   flatten_2[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 10)            413450      concatenate_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 1)             11          dense_1[0][0]
====================================================================================================
Total params: 415,045
Trainable params: 415,045
Non-trainable params: 0
____________________________________________________________________________________________________
```

#### Shared Feature Extraction Layer

In this section, we will use two parallel submodels to interpret the output of an LSTM feature extractor for sequence classification. The input to the model is 100 time steps of 1 feature. An LSTM layer with 10 memory cells interprets this sequence. The first interpretation model is a shallow single fully connected layer, the second is a deep 3 layer model. The output of both interpretation models are concatenated into one long vector that is passed to the output layer used to make a binary prediction.

```python
# Shared Feature Extraction Layer
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.layers.merge import concatenate
# define input
visible = Input(shape=(100,1))
# feature extraction
extract1 = LSTM(10)(visible)
# first interpretation model
interp1 = Dense(10, activation='relu')(extract1)
# second interpretation model
interp11 = Dense(10, activation='relu')(extract1)
interp12 = Dense(20, activation='relu')(interp11)
interp13 = Dense(10, activation='relu')(interp12)
# merge interpretation
merge = concatenate([interp1, interp13])
# output
output = Dense(1, activation='sigmoid')(merge)
model = Model(inputs=visible, outputs=output)
# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file='shared_feature_extractor.png')
```

A plot of the model graph is created and saved to file.

![keras2](/mldl/assets/images/2018-06-25/keras2.jpg)

Running the example summarizes the model layers.

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_1 (InputLayer)             (None, 100, 1)        0
____________________________________________________________________________________________________
lstm_1 (LSTM)                    (None, 10)            480         input_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 10)            110         lstm_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 20)            220         dense_2[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 10)            110         lstm_1[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            210         dense_3[0][0]
____________________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 20)            0           dense_1[0][0]
                                                                   dense_4[0][0]
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             21          concatenate_1[0][0]
====================================================================================================
Total params: 1,151
Trainable params: 1,151
Non-trainable params: 0
____________________________________________________________________________________________________
```


### <a name="Multiple Input and Output Models">Multiple Input and Output Models</a>

The functional API can also be used to develop more complex models with multiple inputs, possibly with different modalities. It can also be used to develop models that produce multiple outputs. We will look at examples of each in this section.

#### Multiple Input Model

We will develop an image classification model that takes two versions of the image as input, each of a different size. Specifically a black and white 64×64 version and a color 32×32 version. Separate feature extraction CNN models operate on each, then the results from both models are concatenated for interpretation and ultimate prediction.

Note that in the creation of the Model() instance, that we define the two input layers as an array. Specifically:

```python
model = Model(inputs=[visible1, visible2], outputs=output)
```

The complete example is listed below.

```python
# Multiple Inputs
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
# first input model
visible1 = Input(shape=(64,64,1))
conv11 = Conv2D(32, kernel_size=4, activation='relu')(visible1)
pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)
conv12 = Conv2D(16, kernel_size=4, activation='relu')(pool11)
pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)
flat1 = Flatten()(pool12)
# second input model
visible2 = Input(shape=(32,32,3))
conv21 = Conv2D(32, kernel_size=4, activation='relu')(visible2)
pool21 = MaxPooling2D(pool_size=(2, 2))(conv21)
conv22 = Conv2D(16, kernel_size=4, activation='relu')(pool21)
pool22 = MaxPooling2D(pool_size=(2, 2))(conv22)
flat2 = Flatten()(pool22)
# merge input models
merge = concatenate([flat1, flat2])
# interpretation model
hidden1 = Dense(10, activation='relu')(merge)
hidden2 = Dense(10, activation='relu')(hidden1)
output = Dense(1, activation='sigmoid')(hidden2)
model = Model(inputs=[visible1, visible2], outputs=output)
# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file='multiple_inputs.png')
```

A plot of the model graph is created and saved to file.

![keras3](/mldl/assets/images/2018-06-25/keras3.jpg)

Running the example summarizes the model layers.

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_1 (InputLayer)             (None, 64, 64, 1)     0
____________________________________________________________________________________________________
input_2 (InputLayer)             (None, 32, 32, 3)     0
____________________________________________________________________________________________________
conv2d_1 (Conv2D)                (None, 61, 61, 32)    544         input_1[0][0]
____________________________________________________________________________________________________
conv2d_3 (Conv2D)                (None, 29, 29, 32)    1568        input_2[0][0]
____________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)   (None, 30, 30, 32)    0           conv2d_1[0][0]
____________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)   (None, 14, 14, 32)    0           conv2d_3[0][0]
____________________________________________________________________________________________________
conv2d_2 (Conv2D)                (None, 27, 27, 16)    8208        max_pooling2d_1[0][0]
____________________________________________________________________________________________________
conv2d_4 (Conv2D)                (None, 11, 11, 16)    8208        max_pooling2d_3[0][0]
____________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)   (None, 13, 13, 16)    0           conv2d_2[0][0]
____________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)   (None, 5, 5, 16)      0           conv2d_4[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2704)          0           max_pooling2d_2[0][0]
____________________________________________________________________________________________________
flatten_2 (Flatten)              (None, 400)           0           max_pooling2d_4[0][0]
____________________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 3104)          0           flatten_1[0][0]
                                                                   flatten_2[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 10)            31050       concatenate_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 10)            110         dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 1)             11          dense_2[0][0]
====================================================================================================
Total params: 49,699
Trainable params: 49,699
Non-trainable params: 0
____________________________________________________________________________________________________
```

#### Multiple Output Model

In this section, we will develop a model that makes two different types of predictions. Given an input sequence of 100 time steps of one feature, the model will both classify the sequence and output a new sequence with the same length.

An LSTM layer interprets the input sequence and returns the hidden state for each time step. The first output model creates a stacked LSTM, interprets the features, and makes a binary prediction. The second output model uses the same output layer to make a real-valued prediction for each input time step.

```python
# Multiple Outputs
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
# input layer
visible = Input(shape=(100,1))
# feature extraction
extract = LSTM(10, return_sequences=True)(visible)
# classification output
class11 = LSTM(10)(extract)
class12 = Dense(10, activation='relu')(class11)
output1 = Dense(1, activation='sigmoid')(class12)
# sequence output
output2 = TimeDistributed(Dense(1, activation='linear'))(extract)
# output
model = Model(inputs=visible, outputs=[output1, output2])
# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file='multiple_outputs.png')
```

A plot of the model graph is created and saved to file.

![keras4](/mldl/assets/images/2018-06-25/keras4.jpg)

Running the example summarizes the model layers.

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_1 (InputLayer)             (None, 100, 1)        0
____________________________________________________________________________________________________
lstm_1 (LSTM)                    (None, 100, 10)       480         input_1[0][0]
____________________________________________________________________________________________________
lstm_2 (LSTM)                    (None, 10)            840         lstm_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 10)            110         lstm_2[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 1)             11          dense_1[0][0]
____________________________________________________________________________________________________
time_distributed_1 (TimeDistribu (None, 100, 1)        11          lstm_1[0][0]
====================================================================================================
Total params: 1,452
Trainable params: 1,452
Non-trainable params: 0
____________________________________________________________________________________________________
```

### <a name="Best Practices">Best Practices</a>

In this section, I want to give you some tips to get the most out of the functional API when you are defining your own models.

* **Consistent Variable Names**. Use the same variable name for the input (visible) and output layers (output) and perhaps even the hidden layers (hidden1, hidden2). It will help to connect things together correctly.
* **Review Layer Summary**. Always print the model summary and review the layer outputs to ensure that the model was connected together as you expected.
* **Review Graph Plots**. Always create a plot of the model graph and review it to ensure that everything was put together as you intended.
* **Name the layers**. You can assign names to layers that are used when reviewing summaries and plots of the model graph. For example: Dense(1, name=’hidden1′).
* **Separate Submodels**. Consider separating out the development of submodels and combine the submodels together at the end.


## <a name="Usage of Callbacks">Usage of Callbacks</a>

A callback is a set of functions to be applied at given stages of the training procedure. You can use callbacks to get a view on internal states and statistics of the model during training. You can pass a list of callbacks (as the keyword argument `callbacks`) to the `.fit()` method of the `Sequential` or `Model` classes. The relevant methods of the callbacks will then be called at each stage of the training.

### ModelCheckpoint

Save the model after every epoch.

```python
keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
```

`filepath` can contain named formatting options, which will be filled the value of `epoch` and keys in `logs` (passed in `on_epoch_end`). For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`, then the model checkpoints will be saved with the epoch number and the validation loss in the filename.

* `monitor`: quantity to monitor.
* `save_best_only`: if `save_best_only=True`, the latest best model according to the quantity monitored will not be overwritten.
* `mode`:  one of {auto, min, max}. If `save_best_only=True`, the decision to overwrite the current save file is made based on either the maximization or the minimization of the monitored quantity. For `val_acc`, this should be `max`, for `val_loss` this should be `min`, etc. In `auto` mode, the direction is automatically inferred from the name of the monitored quantity.
* `save_weights_only`: if True, then only the model's weights will be saved (`model.save_weights(filepath)`), else the full model is saved (`model.save(filepath)`).

### EarlyStopping

Stop training when a monitored quantity has stopped improving.

```python
keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None)
```

* `monitor`: quantity to be monitored.
* `min_delta`: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.
* `patience`: number of epochs with no improvement after which training will be stopped.
* `mode`:  one of {auto, min, max}. In `min` mode, training will stop when the quantity monitored has stopped decreasing; in `max` mode it will stop when the quantity monitored has stopped increasing; in `auto` mode, the direction is automatically inferred from the name of the monitored quantity.
* `baseline`: Baseline value for the monitored quantity to reach. Training will stop if the model doesn't show improvement over the baseline.


References:

* [<u>Keras Documentation</u>](https://keras.io)
* [<u>How to Use the Keras Functional API for Deep Learning</u>](https://machinelearningmastery.com/keras-functional-api-deep-learning/)
