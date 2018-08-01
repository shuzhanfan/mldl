---
layout:         post
title:          Understanding Maximum Likelihood Estimation
subtitle:
card-image:     /mldl/assets/images/cards/cat14.gif
date:           2018-06-20 09:00:00
tags:           [machine&nbsp;learning]
categories:     [machine&nbsp;learning, algorithms]
post-card-type: image
mathjax:        true
---

Often in machine learning we use a model to describe the process that results in the data that are observed. Each model contains its own set of parameters that ultimately defines what the model looks like. So parameters define a blueprint for the model. It is only when specific values are chosen for the parameters that we get an instantiation for the model that describes a given phenomenon.

## Intuitive explanation of maximum likelihood estimation

Maximum Likelihood Estimation (MSE) is a method of estimating the parameters of a statistical model given some data. The parameter values are found such that they maximise the likelihood that the process described by the model produced the data that were actually observed.

Let’s suppose we have observed 10 data points from some process. For example, each data point could represent the length of time in seconds that it takes a student to answer a specific exam question.

We first have to decide which model we think best describes the process of generating the data. This part is very important. At the very least, we should have a good idea about which model to use. This usually comes from having some domain expertise but we wont discuss this here.

For these data we’ll assume that the data generation process can be adequately described by a Gaussian (normal) distribution. Recall that the Gaussian distribution has 2 parameters. The mean, $$\mu$$, and the standard deviation, $$\sigma$$. Different values of these parameters result in different curves. We want to know **which curve was most likely responsible for creating the data points that we observed**?  Maximum likelihood estimation is a method that will find the values of $$/mu$$ and $$\sigma$$ that result in the curve that best fits the data.

## Calculating the Maximum Likelihood Estimates

Now that we have an intuitive understanding of what maximum likelihood estimation is we can move on to learning how to calculate the parameter values. The values that we find are called the maximum likelihood estimates (MLE).

Suppose we have x samples from independent and identically distributed observations, coming from an unknown probability density function, $$f(x\vert\theta)$$, where $$\theta$$ is unknown.

### Properties

#### Joint Probability Density Function

The probability density of observing data points $$x_1, x_2,...,x_n$$ that are generated from a distribution $$f(x\vert \theta)$$ is:

{% raw %}
$$
    f(x_1,x_2,...x_n\vert \theta) = f(x_1\vert \theta)f(x_2\vert \theta)...f(x_n\vert \theta)
$$
{% endraw %}

#### Likelihood

The probability density of the data given the parameters is equal to the likelihood of the parameters given the data.

{% raw %}
$$
    \mathcal{L}(\theta; x_1,x_2,...,x_n) = f(x_1,x_2,...x_n\vert \theta) = \prod_{i=1}^n{f(x_i\vert \theta)}
$$
{% endraw %}

#### Log-Likelihood

The above expression for the total probability is actually quite a pain to differentiate, so it is almost always simplified by taking the natural logarithm of the expression. This is absolutely fine because the natural logarithm is a monotonically increasing function.

{% raw %}
$$
    log \space \mathcal{L}(\theta; x_1,x_2,...,x_n) = \sum_{i=1}^n{log \space f(x_i\vert \theta)}
$$
{% endraw %}

The properties above are building blocks to get an estimation for our parameter value $$\theta$$. By fixing our x sample data as parameters, and $$\theta$$ as a free variable, we are able to find the maximum likelihood by getting the partial derivative respect to $$\theta$$.

### Calculation

Suppose we have a dataset and we assume that they have been generated from a process that is adequately described by a Gaussian distribution.

The Gaussian Distribution probability density function is:

{% raw %}
$$
    f(x\vert \mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}exp(-\frac{(x-\mu)^2}{2\sigma^2})
$$
{% endraw %}

And the joint density function / Likelihood is:

{% raw %}
$$
    f(x_1,x_2,...x_n\vert \mu,\sigma^2) = \prod_{i=1}^n{f(x_i\vert \mu,\sigma^2)} = (\frac{1}{2\pi\sigma^2})^{\frac{n}{2}}exp(-\frac{\sum_{i=1}^n{(x_i-\mu)^2}}{2\sigma^2})
$$
{% endraw %}

The Log-Likelihood is then:

{% raw %}
$$
    log \space \mathcal{L}(\mu,\sigma^2; x_1,x_2,...,x_n) = -\frac{n}{2}log(2\pi\sigma^2) - \frac{\sum_{i=1}^n{(x_i-\mu)^2}}{2\sigma^2}
$$
{% endraw %}

In order to get the MLE values for our parameters, we take derivatives of the function, set the derivative function to zero and then rearrange the equation to make the parameter of interest the subject of the equation.

{% raw %}
$$
    0 = \frac{\delta}{\delta \mu}log \space \mathcal{L}(\mu,\sigma^2) = 0 - \frac{-2\sum_{i=1}^n{(x_i-\mu)}}{2\sigma^2} = \frac{\sum_{i=1}^n{(x_i-\mu)}}{\sigma^2}
$$
{% endraw %}

We can get the MLE of the mean $$\mu$$ is:

{% raw %}
$$
    \hat{\mu} = \sum_{i=1}^n{\frac{x_i}{n}}
$$
{% endraw %}

Likewise, we can get the MLE of the deviation $$\sigma$$ is:

{% raw %}
$$
    \hat{\sigma^2} = \frac{1}{n}\sum_{i=1}^n{(x_i-\mu)^2}
$$
{% endraw %}

As seen with the Normal Distribution case, the parameters relies heavily on our x, “fixed parameters.”

## How could we use this information in building classifier or regression model?

![mle](/mldl/assets/images/mle.jpg)

Our problem shows data with three classes of red, blue, and green. Our independent variables are x and y and we assume these distributions to be Gaussian. With notes above, given our x and y data we can estimate parameters for each classes.

Let’s try to predict for this new data. How would we use the above estimated parameters to classify this new data: $$x = (-3.88, -3.04)$$

{% raw %}
$$
    p(X\vert w_{blue}) = p(X_x\vert w_{blue}) \space p(X_y\vert w_{blue}) \\
    p(X_x\vert w_{blue}) = \frac{1}{1.04\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{-3.88-(-3.98)}{1.04})^2} = 0.37 \\
    p(X_y\vert w_{blue}) = \frac{1}{1.03\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{-3.04-(-2.89)}{1.03})^2} = 0.38 \\
    p(X\vert w_{blue}) = 0.37 \times 0.38 = 0.14
$$
{% endraw %}

Likewise, we can get:
{% raw %}
$$
    p(X\vert w_{red}) = 0.05 \times 2.18e^{-19} = 1.28e^{-20} \\
    p(X\vert w_{green}) = 2.65e^{-18} \times 0.12 = 3.22e^{-19}
$$
{% endraw %}

This is done through calculating three different parameter value sets of red, blue, and green. Probability for the above case being blue is the highest.

References:

* [<u>Probability concepts explained: Maximum likelihood estimation</u>](https://towardsdatascience.com/probability-concepts-explained-maximum-likelihood-estimation-c7b4342fdbb1)
* [<u>Maximum Likelihood Estimation</u>](https://medium.com/@kangeugine/maximum-likelihood-estimation-71c5d0f82f4d)
