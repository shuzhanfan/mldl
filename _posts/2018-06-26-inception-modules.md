---
layout:         post
title:          Understanding Inception Modules
subtitle:
card-image:     /mldl/assets/images/cards/cat17.gif
date:           2018-06-26 09:00:00
tags:           [deep&nbsp;learning]
categories:     [deep&nbsp;learning]
post-card-type: image
mathjax:        true
---

If you’re up to date with the AI world, you know that Google released a model called Inception v3 with Tensorflow in 2016. The namesake of Inception v3 is the Inception modules it uses, which are basically mini models inside the bigger model.  The same Inception architecture was used in the GoogLeNet model which was a state of the art image recognition net in 2014.

In this post we’ll actually go into details about how Inception Modules work from the original paper, “Going Deep with Convolutions” .

## What are Inception modules?

As is often the case with technical creations, if we can understand the problem that led to the creation, we will more easily understand the inner workings of that creation. The inspiration of Inception architecture comes from the idea that you need to make a decision as to what type of convolution you want to make at each layer:  Do you want a 3×3? Or a 5×5?  And this can go on for a while.

So why not use all of them and let the model decide? You do this by doing each convolution in parallel and concatenating the resulting feature maps before going to the next layer.

Now let’s say the next layer is also an Inception module.  Then each of the convolution’s feature maps will be passes through the mixture of convolutions of the current layer. The idea is that you don’t need to know ahead of time if it was better to do, for example, a 3×3 then a 5×5.  Instead, just do all the convolutions and let the model pick what’s best.  Additionally, this architecture allows the model to recover both local feature via smaller convolutions and high abstracted features with larger convolutions.

## Architecture

Now that we get the basic idea, let’s look into the specific architecture that we’ll implement.  The following figure shows the architecture of a single inception module.

![inception1](/mldl/assets/images/2018-06-26/inception1.jpg)

Notice that we get the variety of convolutions that we want; specifically, we will be using 1×1, 3×3, and 5×5 convolutions along with a 3×3 max pooling.  If you’re wondering what the max pooling is doing there with all the other convolutions, we’ve got an answer: pooling is added to the Inception module for no other reason than, historically, good networks having pooling.  The larger convolutions are more computationally expensive, so the paper suggests first doing a 1×1 convolution  reducing the dimensionality of its feature map, passing the resulting feature map through a relu, and then doing the larger convolution (in this case, 5×5 or 3×3). The 1×1 convolution is key because it will be used to reduce the dimensionality of its feature map.

## What does 1x1 convolution mean ([<u>stackoverflow ref</u>](https://stats.stackexchange.com/questions/194142/what-does-1x1-convolution-mean-in-a-neural-network))

Suppose that I have a convolution layer which outputs an $$(H, W, F)$$ shaped tensor where $$H, W$$ are the spatial dimensions, and $$F$$ is the number of convolutional filters. Suppose this output is fed into a conv layer with $$F_1$$ 1x1 filters, zero padding and stride 1. Then the output of this 1x1 conv layer will have shape $$(H, W, F_1)$$.

So 1x1 conv filters can be used to change the dimensionality in the filter space. If $$F_1 > F$$ then we are increasing dimensionality, if $$F_1 < F$$ we are decreasing dimensionality, in the filter dimension.

Indeed, in the Google Inception article Going Deeper with Convolutions, they state:

One big problem with the above modules, at least in this naive form, is that even a modest number of 5x5 convolutions can be prohibitively expensive on top of a convolutional layer with a large number of filters.

This leads to the second idea of the proposed architecture: judiciously applying dimension reductions and projections wherever the computational requirements would increase too much otherwise. This is based on the success of embeddings: even low dimensional embeddings might contain a lot of information about a relatively large image patch...1x1 convolutions are used to compute reductions before the expensive 3x3 and 5x5 convolutions. Besides being used as reductions, they also include the use of rectified linear activation which makes them dual-purpose.

So in the Inception architecture, we use the 1x1 convolutional filters to reduce dimensionality in the filter dimension. As I explained above, these 1x1 conv layers can be used in general to change the filter space dimensionality (either increase or decrease) and in the Inception architecture we see how effective these 1x1 filters can be for dimensionality reduction, explicitly in the filter dimension space, not the spatial dimension space.

## Dimensionality reduction

This was the coolest part of the paper.  The authors say that you can use 1×1 convolutions to reduce the dimensionality of your input to large convolutions, thus keeping your computations reasonable.  To understand what they are talking about, let’s first see why we are in some computational trouble without the reductions.

Let’s examine the number of computations required of the first Inception module of GoogLeNet. The architecture for this model is tabulated in the following figure:

![inception2](/mldl/assets/images/2018-06-26/inception2.jpg)

In the above figure, “#3×3 reduce” and “#5×5 reduce” stands for the number of 1×1 filters in the reduction layer used before the 3×3 and 5×5 convolutions. One can see the number of 1×1 filters in the projection layer after the built-in max-pooling in the pool proj column. All these reduction/projection layers use rectified linear activation as well.

The following summarizes the computation of the number of parameters in the GoogleNet: ([<u>stackoverflow ref</u>](https://stackoverflow.com/questions/30585122/how-to-calculate-the-number-of-parameters-for-google-net))

![inception3](/mldl/assets/images/2018-06-26/inception3.jpg)

**In the Inception module, the concatenation of 1x1, 3x3, 5x5, max-pooling means the concatenation of the corresponding filters. Due to the use of the padding, all of their output have a spatial dimension of 28x28, but they have different numbers of filters. The number of 1x1, 3x3, 5x5, max-pooling conv layer filters are 64, 128, 32, 32, respectively. Hence, the output of the inception_3a has an shape of 28x28x256.**

We can tell that the net uses same padding for the convolutions inside the module, because the input and output are both 28×28.  Let’s just examine what the 5×5 convolution would be computationally if we didn’t do the dimensionality reduction. The following shows these operations.

![inception4](/mldl/assets/images/2018-06-26/inception4.jpg)

There would be $$(5)^2(28)^2(192)(32)=120,422,400$$ operations. That’s a lot of computing! You can see why people might want to do something to bring this number down.

To do this, we will ditch the naive model shown in the above figure and use the model from the following figure. For our 5×5 convolution, we put the previous layer through a 1×1 convolution that outputs a 16 28×28 feature maps (we know there are 16 from the #5×5 reduce column), then we do the 5×5 convolutions on those feature maps which outputs 32 28×28 feature maps.

![inception5](/mldl/assets/images/2018-06-26/inception5.jpg)

In this case, there would be $$(1)^2(28)^2(192)(16) + (5)^2(28)^2(16)(32) = 2,408,448+10,035,200 = 12,443,648$$ operations. Although this is a still a pretty big number, we shrunk the number of computation from the naive model by a factor of ten.

We won’t run through the calculations for the 3×3 covolutions, but they follow the same process as the 5×5 convolutions.  Hopefully, this sections cleared up why the 1×1 convolutions are necessary before large convolutions!

## Different versions of Inception network

The Inception network used a lot of tricks to push performance; both in terms of speed and accuracy. Its constant evolution lead to the creation of several versions of the network. The popular versions are as follows:

* Inception v1.
* Inception v2 and Inception v3.
* Inception v4 and Inception-ResNet.

Each version is an iterative improvement over the previous one. Understanding the upgrades can help us to build custom classifiers that are optimized both in speed and accuracy. Also, depending on your data, a lower version may actually work better. This section aims to elucidate the evolution of the inception network.

### Inception v1

This is where it all started. Let us analyze what problem it was purported to solve, and how it solved it.

#### The Premise

Salient parts in the image can have extremely large variation in size. For instance, for images with dogs, the area occupied by the dog is different in each image. Because of this huge variation in the location of the information, choosing the right kernel size for the convolution operation becomes tough. A larger kernel is preferred for information that is distributed more globally, and a smaller kernel is preferred for information that is distributed more locally. Very deep networks are prone to overfitting. It is also hard to pass gradient updates through the entire network. And naively stacking large convolution operations is computationally expensive.

#### The Solution

So why not have filters with multiple sizes operate on the same level? The network essentially would get a bit “wider” rather than “deeper”. The authors designed the inception module to reflect the idea.

The "naive" inception module performs convolution on an input, with 3 different sizes of filters (1x1, 3x3, 5x5). Additionally, max pooling is also performed. The outputs are concatenated and sent to the next inception module.

As stated before, deep neural networks are computationally expensive. To make it cheaper, the authors limit the number of input channels by adding an extra 1x1 convolution before the 3x3 and 5x5 convolutions. Though adding an extra operation may seem counterintuitive, 1x1 convolutions are far more cheaper than 5x5 convolutions, and the reduced number of input channels also help. Do note that however, the 1x1 convolution is introduced after the max pooling layer, rather than before.

Using the dimension reduced inception module, a neural network architecture was built. This was popularly known as GoogLeNet (Inception v1). The architecture is shown below:

![inception6](/mldl/assets/images/2018-06-26/inception6.jpg)

GoogLeNet has 9 such inception modules stacked linearly. It is 22 layers deep (27, including the pooling layers). It uses global average pooling at the end of the last inception module.

Needless to say, it is a pretty deep classifier. As with any very deep network, it is subject to the vanishing gradient problem.

To prevent the middle part of the network from “dying out”, the authors introduced two auxiliary classifiers (The purple boxes in the image). They essentially applied softmax to the outputs of two of the inception modules, and computed an auxiliary loss over the same labels. The total loss function is a weighted sum of the auxiliary loss and the real loss. Weight value used in the paper was 0.3 for each auxiliary loss.

The total loss used by the inception net during training is: $$total\_loss = real\_loss + 0.3 * aux\_loss\_1 + 0.3 * aux\_loss\_2$$

Needless to say, auxiliary loss is purely used for training purposes, and is ignored during inference.

### Inception v2

Inception v2 and Inception v3 were presented in the same paper. The authors proposed a number of upgrades which increased the accuracy and reduced the computational complexity. Inception v2 explores the following:

#### The Premise

* Reduce representational bottleneck. The intuition was that, neural networks perform better when convolutions didn’t alter the dimensions of the input drastically. Reducing the dimensions too much may cause loss of information, known as a “representational bottleneck”
* Using smart factorization methods, convolutions can be made more efficient in terms of computational complexity.

#### The Solution

* Factorize 5x5 convolution to two 3x3 convolution operations to improve computational speed. Although this may seem counterintuitive, a 5x5 convolution is 2.78 times more expensive than a 3x3 convolution. So stacking two 3x3 convolutions infact leads to a boost in performance. This is illustrated in the below image.

![inception7](/mldl/assets/images/2018-06-26/inception7.jpg)

* Moreover, they factorize convolutions of filter size nxn to a combination of 1xn and nx1 convolutions. For example, a 3x3 convolution is equivalent to first performing a 1x3 convolution, and then performing a 3x1 convolution on its output. They found this method to be 33% more cheaper than the single 3x3 convolution. This is illustrated in the below image.

![inception8](/mldl/assets/images/2018-06-26/inception8.jpg)

* The filter banks in the module were expanded (made wider instead of deeper) to remove the representational bottleneck. If the module was made deeper instead, there would be excessive reduction in dimensions, and hence loss of information. This is illustrated in the below image.

![inception9](/mldl/assets/images/2018-06-26/inception9.jpg)

* The above three principles were used to build three different types of inception modules and they will be used in the Inception v2 architecture.


### Inception v3

#### The Premise

* The authors noted that the auxiliary classifiers didn’t contribute much until near the end of the training process, when accuracies were nearing saturation. They argued that they function as regularizers, especially if they have BatchNorm or Dropout operations.
* Possibilities to improve on the Inception v2 without drastically changing the modules were to be investigated.

#### The Solution

Inception Net v3 incorporated all of the above upgrades stated for Inception v2, and in addition used the following:

* RMSProp Optimizer.
* Factorized 7x7 convolutions.
* BatchNorm in the Auxillary Classifiers.
* Label Smoothing (A type of regularizing component added to the loss formula that prevents the network from becoming too confident about a class. Prevents over fitting).


### Inception v4

Inception v4 and Inception-ResNet were introduced in the same paper. For clarity, let us discuss them in separate sections.

#### The Premise

Make the modules more uniform. The authors also noticed that some of the modules were more complicated than necessary. This can enable us to boost performance by adding more of these uniform modules.

#### The Solution

* The “stem” of Inception v4 was modified. The stem here, refers to the initial set of operations performed before introducing the Inception blocks.

![inception10](/mldl/assets/images/2018-06-26/inception10.jpg)

* They had three main inception modules, named A,B and C (Unlike Inception v2, these modules are infact named A,B and C). They look very similar to their Inception v2 (or v3) counterparts.

![inception11](/mldl/assets/images/2018-06-26/inception11.jpg)

* Inception v4 introduced specialized “Reduction Blocks” which are used to change the width and height of the grid. The earlier versions didn’t explicitly have reduction blocks, but the functionality was implemented.

![inception12](/mldl/assets/images/2018-06-26/inception12.jpg)

### Inception-ResNet v1 and v2

Inspired by the performance of the ResNet, a hybrid inception module was proposed. There are two sub-versions of Inception ResNet, namely v1 and v2. Before we checkout the salient features, let us look at the minor differences between these two sub-versions.

* Inception-ResNet v1 has a computational cost that is similar to that of Inception v3.
* Inception-ResNet v2 has a computational cost that is similar to that of Inception v4.
* They have different stems, as illustrated in the Inception v4 section.
* Both sub-versions have the same structure for the modules A, B, C and the reduction blocks. Only difference is the hyper-parameter settings. In this section, we’ll only focus on the structure. Refer to the paper for the exact hyper-parameter settings (The images are of Inception-Resnet v1).

#### The Premise

Introduce residual connections that add the output of the convolution operation of the inception module, to the input.

#### The Solution

* For residual addition to work, the input and output after convolution must have the same dimensions. Hence, we use 1x1 convolutions after the original convolutions, to match the depth sizes (Depth is increased after convolution).

![inception13](/mldl/assets/images/2018-06-26/inception13.jpg)

* The pooling operation inside the main inception modules were replaced in favor of the residual connections. However, you can still find those operations in the reduction blocks. Reduction block A is same as that of Inception v4.
* Networks with residual units deeper in the architecture caused the network to “die” if the number of filters exceeded 1000. Hence, to increase stability, the authors scaled the residual activations by a value around 0.1 to 0.3.
* The original paper didn’t use BatchNorm after summation to train the model on a single GPU (To fit the entire model on a single GPU).
* It was found that Inception-ResNet models were able to achieve higher accuracies at a lower epoch.
* The final network layout for both Inception v4 and Inception-ResNet are as follows:

![inception14](/mldl/assets/images/2018-06-26/inception14.jpg)

## Keras implementation

We will build a simple architecture with just one layer of inception module using keras. We will train the architecture on the popular CIFAR-10 dataset which consists of 32x32 images belonging to 10 different classes.

Keras has the functionality to directly download the dataset using the cifar10.load_data() function. Once downloaded the function loads the data ready to use.

```python
from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```

Each image is represented as 32x32 pixels each for red, blue and green channels. Each pixel has a value between 0–255. Next, we normalize the values to 0–1.

```python
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
```

In order to best model the classification model, we convert y_test and y_train to one hot representations in the form of a binary matrix.

```python
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
```

The Keras functional API allows us to define complex models. In order to create a model, let us first define an input_img tensor for a 32x32 image with 3 channels(RGB).

```python
from keras.layers import Input
input_img = Input(shape = (32, 32, 3))
```

Now, we feed the input tensor to each of the 1x1, 3x3, 5x5 filters in the inception module.

```python
from keras.layers import Conv2D, MaxPooling2D
incep_1_1by1 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
incep_3_1by1 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
incep_3_3by3 = Conv2D(64, (3,3), padding='same', activation='relu')(incep_3_1by1)
incep_5_1by1 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
incep_5_5by5 = Conv2D(64, (5,5), padding='same', activation='relu')(incep_5_1by1)
incep_pool_3by3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
incep_pool_1by1 = Conv2D(64, (1,1), padding='same', activation='relu')(incep_pool_3by3)
```

The padding is kept same so that the output shape of the Conv2D operation is same as the input shape. Thus we can easily concatenate these filters to form the output of our inception module.

```python
output = keras.layers.concatenate([incep_1_1by1, incep_3_3by3, incep_5_5by5, incep_pool_1by1], axis = 3)
```

Concatenate operation assumes that the dimensions of the items are the same, except for the concatenation axis.

We flatten the output to a one dimensional collection of neurons which is then used to create a fully connected neural network as a final classifier

```python
from keras.layers import Flatten, Dense
output = Flatten()(output)
out    = Dense(10, activation='softmax')(output)
```

Thus we obtain a fully connected neural network with final layer having 10 neurons one corresponding to each class.

We can now create the model

```python
from keras.models import Model
model = Model(inputs = input_img, outputs = out)
print model.summary()
```

References:

* [<u>Inception modules: explained and implemented</u>](https://hacktilldawn.com/2016/09/25/inception-modules-explained-and-implemented/)
* [<u>A Simple Guide to the Versions of the Inception Network</u>](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202)
* [<u>Understanding and Coding Inception Module in Keras</u>](https://becominghuman.ai/understanding-and-coding-inception-module-in-keras-eb56e9056b4b)
* [<u>Going deeper with convolutions</u>](https://arxiv.org/pdf/1409.4842.pdf)
