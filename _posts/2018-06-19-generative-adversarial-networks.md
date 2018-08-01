---
layout:         post
title:          Generative Adversarial Networks
subtitle:
card-image:     /mldl/assets/images/cards/cat13.gif
date:           2018-06-19 09:00:00
tags:           [deep&nbsp;learning]
categories:     [deep&nbsp;learning]
post-card-type: image
mathjax:        true
---

## Introduction

Neural Networks have made great progress. They now recognize images and voice at levels comparable to humans. They are also able to understand natural language with a good accuracy.

But, even then, the talk of automating human tasks with machines looks a bit far fetched. After all, we do much more than just recognizing image / voice or understanding what people around us are saying – don’t we?

Let us see a few examples where we need human creativity (at least as of now):

* Train an artificial author which can write an article and explain data science concepts to a community in a very simplistic manner by learning from past articles
* You are not able to buy a painting from a famous painter which might be too expensive. Can you create an artificial painter which can paint like any famous artist by learning from his / her past collections?

Do you think, these tasks can be accomplished by machines? Well – the answer might surprise you.

These are definitely difficult to automate tasks, but Generative Adversarial Networks (GANs) have started making some of these tasks possible.

If you feel intimidated by the name GAN – don’t worry! You will feel comfortable with them by end of this article. In this article, I will introduce you to the concept of GANs and explain how they work along with the challenges. I will also let you know of some cool things people have done using GAN and give you links to some of the important resources for getting deeper into these techniques.

## An analogy to GANs

But what is a GAN? Let us take an analogy to explain the concept:

A real analogy can be considered as a relation between forger and an investigator. The task of a forger is to create fraudulent imitations of original paintings by famous artists. If this created piece can pass as the original one, the forger gets a lot of money in exchange of the piece.

On the other hand, an art investigator’s task is to catch these forgers who create the fraudulent pieces. How does he do it? He knows what are the properties which sets the original artist apart and what kind of painting he should have created. He evaluates this knowledge with the piece in hand to check if it is real or not.

This contest of forger vs investigator goes on, which ultimately makes world class investigators (and unfortunately world class forger); a battle between good and evil.

## How do GANs work?

We got a high level overview of GANs. Now, we will go on to understand their nitty-gritty of these things.

### The GAN framework

The basic idea of GANs is to set up a game between two players. One of them is called the **generator**. The generator creates samples that are intended to come from the same distribution as the training data. The other player is the **discriminator**. The discriminator examines samples to determine whether they are real or fake. The discriminator learns using traditional supervised learning techniques, dividing inputs into two classes (real or fake). The generator is trained to fool the discriminator. We can think of the generator as being like a counterfeiter, trying to make fake money, and the discriminator as being like police, trying to allow legitimate money and catch counterfeit money. To succeed in this game, the counterfeiter must learn to make money that is indistinguishable from genuine money, and the generator network must learn to create samples that are drawn from the same distribution as the training data. The process is illustrated in the following figure:

![gan1](/mldl/assets/images/2018-06-19/gan1.jpg)

Figure 1: The GAN framework pits two adversaries against each other in a game. Each player is represented by a differentiable function controlled by a set of parameters. Typically these functions are implemented as deep neural networks. The game plays out in two scenarios. In one scenario, training examples $$x$$ are randomly sampled from the training set and used as input for the first player, the discriminator, represented by the function $$D$$.  The goal of the discriminator is to output the probability that its input is real rather than fake, under the assumption that half of the inputs it is ever shown are real and half are fake. In this first scenario, the goal of the discriminator is for $$D(x)$$ to be near 1. In the second scenario, inputs $$z$$ to the generator are randomly sampled from the model’s prior over the latent variables. The discriminator then receives input $$G(z)$$, a fake sample created by the generator. In this scenario, both players participate. The discriminator strives to make $$D(G(z))$$ approach 0 while the generator  strives to make the same quantity approach 1. If both models have sufficient capacity, then the Nash equilibrium of this game corresponds to the $$G(z)$$ being drawn from the same distribution as the training data, and $$D(x)=\frac{1}{2}$$ for all $$x$$.

Formally, GANs are a structured probabilistic model containing latent variables $$z$$ and observed variables $$x$$.

The two players in the game are represented by two functions, each of which is differentiable both with respect to its inputs and with respect to its parameters. The discriminator is a function $$D$$ that takes $$x$$ as input and uses $$\theta^{(D)}$$ as parameters. The generator is defined by a function $$G$$ that takes $$z$$ as input and used $$\theta^{(G)}$$ as parameters.

Both players have cost functions that are defined in terms of both players’ parameters. The discriminator wishes to minimize $$J^{(D)}(\theta^{(D)}, \theta^{(G)})$$ and must do so while controlling only $$\theta^{(D)}$$. The generator wishes to minimize $$J^{(G)}(\theta^{(D)}, \theta^{(G)})$$ and must do so while controlling on $$\theta^{(G)}$$. Because each player’s cost depends on the other player’s parameters, but each player cannot control the other player’s parameters, this scenario is most straightforward to describe as a game rather than as an optimization problem. The solution to an optimization problem is a (local) minimum, a point in parameter space where all neighboring points have greater or equal cost. The solution to a game is a Nash equilibrium. Here, we use the terminology of local differential Nash equilibria. In this context, a Nash equilibrium is a tuple $$(\theta^{(D)}, \theta^{(G)})$$ that is a local minimum of $$J^{(D)}$$ with respect to $$\theta^{(D)}$$ and a local minimum of $$J^{(G)}$$ with respect to $$\theta^{(G)}$$.

### Another discussion of the GAN framework

As we saw, there are two main components of a GAN – Generator Neural Network and Discriminator Neural Network.

![gan2](/mldl/assets/images/2018-06-19/gan2.jpg)

The Generator Network takes an random input and tries to generate a sample of data. In the above image, we can see that generator $$G(z)$$ takes an input $$z$$ from $$p_z(z)$$, where $$z$$ is a sample from probability distribution $$p_z(z)$$. It then generates a data which is then fed into a discriminator network $$D(x)$$. The task of Discriminator Network is to take input either from the real data or from the generator and try to predict whether the input is real or generated. It takes an input $$x$$ from $$p_{data}(x)$$ where $$p_{data}(x)$$ is our real data distribution. $$D(x)$$ then solves a binary classification problem using sigmoid function giving output in the range 0 to 1.

Let us define the notations we will be using to formalize our GAN,
* $$p_{data}(x)$$ -- the distribution of real data
* $$X$$ -- sample from $$p_{data}(x)$$
* $$p_z(z)$$ -- distribution of generator
* $$Z$$ -- sample from $$p_z(z)$$
* $$G(z)$$ -- Generator Network
* $$D(x)$$ -- Discriminator Network

Now the training of GAN is done (as we saw above) as a fight between generator and discriminator. This can be represented mathematically as

{% raw %}
$$
    \min_G \max_D V(D,G) \\
    V(D,G) = \mathbb{E}_{x\sim p_{data}(x)}[log(D(x))] + \mathbb{E}_{z\sim p_{z}(z)}[log(1-D(G(z)))]
$$
{% endraw %}

In our function $$V(D, G)$$ the first term is entropy that the data from real distribution ($$p_{data}(x)$$) passes through the discriminator (aka best case scenario). The discriminator tries to maximize this to 1. The second term is entropy that the data from random input ($$p_z(z)$$) passes through the generator, which then generates a fake sample which is then passed through the discriminator to identify the fakeness (aka worst case scenario). In this term, discriminator tries to maximize it to 0 (i.e. the log probability that the data from generated is fake is equal to 0). **So overall, the discriminator is trying to maximize our function $$V$$**.

On the other hand, **the task of generator is exactly opposite, i.e. it tries to minimize the function V** so that the differentiation between real and fake data is bare minimum. This, in other words is a cat and mouse game between generator and discriminator!

Note: This method of training a GAN is taken from game theory called the minimax game.

### Parts of training GAN

So broadly a training phase has two main subparts and they are done sequentially.

**Pass 1**: Train discriminator and freeze generator (freezing means setting training as false. The network does only forward pass and no backpropagation is applied)

![gan3](/mldl/assets/images/2018-06-19/gan3.jpg)

**Pass 2**: Train generator and freeze discriminator

![gan4](/mldl/assets/images/2018-06-19/gan4.jpg)

### Steps to train a GAN

1. **Define the problem**. Do you want to generate fake images or fake text. Here you should completely define the problem and collect data for it.
2. **Define architecture of GAN**. Define how your GAN should look like. Should both your generator and discriminator be multi layer perceptrons, or convolutional neural networks? This step will depend on what problem you are trying to solve.
3. **Train Discriminator on real data for n epochs**. Get the data you want to generate fake on and train the discriminator to correctly predict them as real. Here value n can be any natural number between 1 and infinity.
4. **Generate fake inputs for generator and train discriminator on fake data**. Get generated data and let the discriminator correctly predict them as fake.
5. **Train generator with the output of discriminator**. Now when the discriminator is trained, you can get its predictions and use it as an objective for training the generator. Train the generator to fool the discriminator.
6. **Repeat step 3 to step 5 for a few epochs**. Repeat step 3 to step 5 for a few epochs.
7. **Check the fake data manually if it seems legit. If it seems appropriate, stop training, else go to step 3**. This is a bit of a manual task, as hand evaluating the data is the best way to check the fakeness. When this step is over, you can evaluate whether the GAN is performing well enough.

A pseudocode of GAN training can be thought out as follows

![gan5](/mldl/assets/images/2018-06-19/gan5.jpg)

Note: This is the first implementation of GAN that was published in the paper. Numerous improvements/updates in the pseudocode can be seen in the recent papers such as adding batch normalization in the generator and discrimination network, training generator k times etc.

Now just take a breath and look at what kind of implications this technique could have. If hypothetically you had a fully functional generator, you can duplicate almost anything. To give you examples, you can generate fake news; create books and novels with unimaginable stories; on call support and much more. You can have artificial intelligence as close to reality; a true artificial  intelligence! That’s the dream!!

### The DCGAN architecture

Most GANs today are at least loosely based on the DCGAN architecture.  DCGAN stands for “deep, convolution GAN.” Though GANs were both deep and convolutional prior to DCGANs, the name DCGAN is useful to refer to this specific style of architecture. Some of the key insights of the DCGAN architecture were to:

* Use batch normalization layers in most layers of both the discriminator and the generator, with the two minibatches for the discriminator normalized separately. The last layer of the generator and first layer of the discriminator are not batch normalized, so that the model can learn the correct mean and scale of the data distribution.
* The overall network structure is mostly borrowed from the all-convolutional net (Springenberg et al., 2015). This architecture contains neither pooling nor “unpooling” layers. When the generator needs to increase the spatial dimension of the representation it uses transposed convolution with a stride greater than 1.
* The use of the Adam optimizer rather than SGD with momentum.

## Challenges with GANs

You may ask, if we know what could these beautiful creatures (monsters?) do; why haven’t something happened? This is because we have barely scratched the surface. There’s so many roadblocks into building a “good enough” GAN and we haven’t cleared many of them yet. There’s a whole area of research out there just to find “how to train a GAN”.

The most important roadblock while training a GAN is stability. If you start to train a GAN, and the discriminator part is much powerful that its generator counterpart, the generator would fail to train effectively. This will in turn affect training of your GAN. On the other hand, if the discriminator is too lenient; it would let literally any image be generated. And this will mean that your GAN is useless.

Another way to glance at stability of GAN is to look as a holistic convergence problem. Both generator and discriminator are fighting against each other to get one step ahead of the other. Also, **they are dependent on each other for efficient training. If one of them fails, the whole system fails. So you have to make sure they don’t explode**.

There are other problems too, which I will list down here.

* **Problems with Counting**: GANs fail to differentiate how many of a particular object should occur at a location.
* **Problems with Perspective**: GANs fail to adapt to 3D objects. It doesn’t understand perspective, i.e.difference between frontview and backview.
* **Problems with Global Structures**: Same as the problem with perspective, GANs do not understand a holistic structure.

A substantial research is being done to take care of these problems. Newer types of models are proposed which give more accurate results than previous techniques, such as DCGAN, WassersteinGAN etc.

## Applications of GAN

We saw an overview of how these things work and got to know the challenges of training them. We will now see the cutting edge research that has been done using GANs

* **Predicting the next frame in a video**: You train a GAN on video sequences and let it predict what would occur next.
* **Increasing Resolution of an image**: Generate a high resolution photo from a comparatively low resolution.
* **Interactive Image Generation**: Draw simple strokes and let the GAN draw an impressive picture for you!
* **Image to Image Translation**: Generate an image from another image.
* **Text to Image Generation**: Just say to your GAN what you want to see and get a realistic photo of the target.

References:

* [<u>Ian Goodfellow's GANs tutorial on NIPS 2016</u>](https://arxiv.org/pdf/1701.00160.pdf)
* [<u>Ian Goodfellow's original paper on Generative Adversarial Nets</u>](https://arxiv.org/pdf/1406.2661.pdf)
* [<u>Introductory guide to Generative Adversarial Networks (GANs) and their promise!</u>](https://www.analyticsvidhya.com/blog/2017/06/introductory-generative-adversarial-networks-gans/)
* [<u>Understanding Generative Adversarial Networks</u>](https://danieltakeshi.github.io/2017/03/05/understanding-generative-adversarial-networks/)
