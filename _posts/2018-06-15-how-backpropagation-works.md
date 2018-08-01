---
layout:         post
title:          How the backpropagation algorithm works
subtitle:
card-image:     /mldl/assets/images/cards/cat12.gif
date:           2018-06-15 09:00:00
tags:           [machine&nbsp;learning]
categories:     [machine&nbsp;learning, algorithms]
post-card-type: image
mathjax:        true
---

We know how neural networks can learn their weights and biases using the gradient descent algorithm. There was, however, a gap in our explanation: we didn't discuss how to compute the gradient of the cost function. That's quite a gap! In this chapter I'll explain a fast algorithm for computing such gradients, an algorithm known as backpropagation.

At the heart of backpropagation is an expression for the partial derivative $$\delta C / \delta w$$ of the cost function $$C$$ with respect to $$w$$ (or bias $$b$$) in the network. The expression tells us how quickly the cost changes when we change the weights and biases. And while the expression is somewhat complex, it also has a beauty to it, with each element having a natural, intuitive interpretation. And so backpropagation isn't just a fast algorithm for learning. It actually gives us detailed insights into how changing the weights and biases changes the overall behaviour of the network. That's well worth studying in detail.


## Notations

Let's begin with a notation which lets us refer to weights in the network in an unambiguous way. We'll use $$w_{jk}^{l}$$ to denote the weight for the connection from the $$k^{th}$$ neuron in the $$(l-1)^{th}$$ layer to the $$j^{th}$$ neuron in the $$l^{th}$$ layer.

This notation is cumbersome at first, and it does take some work to master. But with a little effort you'll find the notation becomes easy and natural. We use a similar notation for the network's biases and activations. Explicitly, we use $$b_j^l$$ for the bias of the $$j^{th}$$ neuron in the $$l^{th}$$ layer. And we use $$a_j^l$$ for the activation of the $$j^{th}$$ neuron in the $$l^{th}$$ layer.

With these notations, the activation $$a_j^l$$ of the $$j^{th}$$ neuron in the $$l^{th}$$ layer is related to the activations in the $$(l-1)^{th}$$ layer by the equation:
{% raw %}
$$
    a_j^l = \sigma (\sum_k{w_{jk}^j a_k^{l-1}} + b_j^l)
$$
{% endraw %}

where the sum is over all neurons $$k$$ in the $$(l-1)^{th}$$ layer. To rewrite this expression in a matrix form we define a weight matrix $$w^l$$ for each layer, $$l$$. The entries of the weight matrix $$w^l$$ are just the weights connecting to the $$l^{th}$$ layer of neurons, that is, the entry in the $$j^{th}$$ row and $$k^{th}$$ column is $$w_{jk}^l$$. Similarly, for each layer $$l$$ we define a bias vector, $$b^l$$ and an activation vector $$a^l$$:
{% raw %}
$$
    a^l = \sigma (w^l a^{l-1} + b^l)
$$
{% endraw %}

This expression gives us a much more global way of thinking about how the activations in one layer relate to activations in the previous layer: we just apply the weight matrix to the activations, then add the bias vector, and finally apply the $$\sigma$$ function.

When using this equation to compute $$a^l$$, we compute the intermediate quantity $$z^l = w^l a^{l-1} + b^l$$ along the way. This quantity turns out to be useful enough to be worth naming: we call $$z^l$$ the weighted input to the neurons in layer $$l$$. And we can get $$a^l = \sigma(z^l)$$.

## The four fundamental equations behind backpropagation

Backpropagation is about understanding how changing the weights and biases in a network changes the cost function. Ultimately, this means computing the partial derivatives $$\delta C / \delta w_{jk}^l$$ and $$\delta C / \delta b_j^l$$. But to compute those, we first introduce an intermediate quantity, $$\sigma_j^l$$, which we call the error in the $$j^{th}$$ neuron in the $$l^{th}$$ layer. Backpropagation will give us a procedure to compute the error $$\sigma_j^l$$, and then will relate $$\sigma_j^l$$ to $$\delta C / \delta w_{jk}^l$$ and $$\delta C / \delta b_j^l$$.

To understand how the error is defined, imagine there is a demon in our neural network. The demon sits at the $$j^{th}$$ neuron in layer $$l$$. As the input to the neuron comes in, the demon messes with the neuron's operation. It adds a little change $$\Delta z_j^l$$ to the neuron's weighted input, so that instead of outputting $$\sigma (z_j^l)$$, the neuron instead outputs $$\sigma (z_j^l+\Delta z_j^l)$$. This change propagates through later layers in the network, finally causing the overall cost to change by an amount $$\frac{\delta C}{\delta z_j^l} \Delta z_j^l$$.

Now, this demon is a good demon, and is trying to help you improve the cost, i.e., they're trying to find a $$\Delta z_j^l$$ which makes the cost smaller. Suppose $$\frac{\delta C}{\delta z_j^l}$$ has a large value (either positive or negative). Then the demon can lower the cost quite a bit by choosing $$\Delta z_j^l$$ to have the opposite sign to $$\frac{\delta C}{\delta z_j^l}$$. By contrast, if $$\frac{\delta C}{\delta z_j^l}$$ is close to zero, then the demon can't improve the cost much at all by perturbing the weighted input $$z_j^l$$. So far as the demon can tell, the neuron is already pretty near optimal. And so there's a heuristic sense in which $$\frac{\delta C}{\delta z_j^l}$$ is a measure of the error in the neuron.

Motivated by this story, we define the error $$\delta_j^l$$ of neuron $$j$$ in layer $$l$$ by:
{% raw %}
$$
    \delta_j^l = \frac{\delta C}{\delta z_j^l}
$$
{% endraw %}

As per our usual conventions, we use $$\delta^l$$ to denote the vector of errors associated with layer $$l$$. Backpropagation will give us a way of computing $$\delta^l$$ for every layer, and then relating those errors to the quantities of real interest, $$\delta C / \delta w_{jk}^l$$ and $$\delta C / \delta b_j^l$$.

You might wonder why the demon is changing the weighted input $$z_j^l$$. Surely it'd be more natural to imagine the demon changing the output activation $$a_j^l$$, with the result that we'd be using $$\delta C / \delta a_j^l$$ as our measure of error. In fact, if you do this things work out quite similarly to the discussion below. But it turns out to make the presentation of backpropagation a little more algebraically complicated. So we'll stick with $$\delta_j^l = \frac{\delta C}{\delta z_j^l}$$ as our measure of error.

**Plan of attack**: Backpropagation is based around four fundamental equations. Together, those equations give us a way of computing both the error $$\delta^l$$ nd the gradient of the cost function. I state the four equations below. Be warned, though: you shouldn't expect to instantaneously assimilate the equations. Such an expectation will lead to disappointment. In fact, the backpropagation equations are so rich that understanding them well requires considerable time and patience as you gradually delve deeper into the equations. The good news is that such patience is repaid many times over. And so the discussion in this section is merely a beginning, helping you on the way to a thorough understanding of the equations.

**An equation for the error $$\delta^L$$ in the output layer**: The components of $$\delta^L$$ are given by:
{% raw %}
$$
    \delta_j^L = \frac{\delta C}{\delta a_j^L} \sigma'(z_j^L) \space \space \space \space \space \space \space \space \space \space (BP1)
$$
{% endraw %}

This is a very natural expression. The first term on the right, $$\frac{\delta C}{\delta a_j^L}$$, just measures how fast the cost is changing as a function of the $$j^{th}$$ output activation. If, for example, $$C$$ doesn't depend much on a particular output neuron, $$j$$, then $$\delta_j^L$$ will be small, which is what we'd expect. The second term on the right, $$\sigma'(z_j^L)$$, measures how fast the activation function $$\sigma$$ is changing at $$z_j^L$$.

Notice that everything in (BP1) is easily computed. In particular, we compute $$z_j^L$$ while computing the behaviour of the network, and it's only a small additional overhead to compute $$\sigma'(z_j^L)$$. The exact form of $$\frac{\delta C}{\delta a_j^L}$$ will, of course, depend on the form of the cost function. However, provided the cost function is known there should be little trouble computing $$\frac{\delta C}{\delta a_j^L}$$. For example, if we're using the quadratic cost function then $$C=\frac{1}{2} \sum_j{(y_j-a_j^L)^2}$$ and so $$\frac{\delta C}{\delta a_j^L} = (a_j^L-y_j)$$, which obviously is easily computable.

Equation (BP1) is a componentwise expression for $$\delta^L$$. It's a perfectly good expression, but not the matrix-based form we want for backpropagation. However, it's easy to rewrite the equation in a matrix-based form, as:

{% raw %}
$$
    \delta^L = \nabla_a C \odot \sigma'(z_j^L) \space \space \space \space \space \space \space \space \space \space (BP1a)
$$
{% endraw %}

**An equation for the error $$\delta^l$$ in terms of the error $$\delta^{l+1}$$ in the next layer**: In particular

{% raw %}
$$
    \delta^l = ((w^{l+1})^T \delta^{l+1}) \odot \sigma'(z_j^l) \space \space \space \space \space \space \space \space \space \space (BP2)
$$
{% endraw %}

where $$(w^{l+1})^T$$ is the transpose of the weight matrix $$w^{l+1}$$ for the $$(l+1)^{th}$$ layer. This equation appears complicated, but each element has a nice interpretation. Suppose we know the error $$\delta^{l+1}$$ at the $$(l+1)^{th}$$ layer. When we apply the transpose weight matrix, $$(w^{l+1})^T$$, we can think intuitively of this as moving the error backward through the network, giving us some sort of measure of the error at the output of the $$l^{th}$$ layer. We then take the Hadamard product $$\odot \sigma'(z_j^l)$$. This moves the error backward through the activation function in layer $$l$$, giving us the error $$\delta^l$$ in the weighted input to layer $$l$$.

By combining (BP2) with (BP1) we can compute the error $$\delta^l$$ for any layer in the network. We start by using (BP1) to compute $$\delta^L$$, then apply Equation (BP2) to compute $$\delta^{L-1}$$, then Equation (BP2) again to compute $$\delta^{L-2}$$, and so on, all the way back through the network.

**An equation for the rate of change of the cost with respect to any bias in the network**

{% raw %}
$$
    \frac{\delta C}{\delta b_j^l} = \delta_j^l \space \space \space \space \space \space \space \space \space \space (BP3)
$$
{% endraw %}

That is, the error $$\delta_j^l$$ is exactly equal to the rate of change $$\frac{\delta C}{\delta b_j^l}$$. This is great news, since (BP1) and (BP2) have already told us how to compute $$\delta_j^l$$. We can rewrite (BP3) in shorthand as:

{% raw %}
$$
    \frac{\delta C}{\delta b} = \delta \space \space \space \space \space \space \space \space \space \space
$$
{% endraw %}

where it is understood is that $$\delta$$ is being evaluated at the same neuron as the bias $$b$$.

**An equation for the rate of change of the cost with respect to any weight in the network**: In particular:

{% raw %}
$$
    \frac{\delta C}{\delta w_{jk}^l} = a_k^{l-1} \delta_j^l \space \space \space \space \space \space \space \space \space \space (BP4)
$$
{% endraw %}

This tells us how to compute the partial derivatives $$\frac{\delta C}{\delta w_{jk}^l} = a_k^{l-1}$$ in terms of the quantities $$\delta^l$$ and $$a^{l-1}$$, which we already know how to compute. The equation can be rewritten in a less index-heavy notation as:

{% raw %}
$$
    \frac{\delta C}{\delta w} = a_{in} \delta_{out} \space \space \space \space \space \space \space \space \space \space
$$
{% endraw %}

where it's understood that $$a_{in}$$ is the activation of the neuron input to the weight $$w$$, and $$\delta_{out}$$ is the error of the neuron output from the weight $$w$$.

A nice consequence of the above Equation is that when the activation $$a_{in}$$ is small, the gradient term $$\frac{\delta C}{\delta w}$$ will also tend to be small. In this case, we'll say the weight learns slowly, meaning that it's not changing much during gradient descent. In other words, one consequence of (BP4) is that weights output from low-activation neurons learn slowly.

There are other insights along these lines which can be obtained from (BP1)-(BP4). Let's start by looking at the output layer. Consider the term $$\sigma'(z_j^L)$$ in (BP1). Recall from the graph of the sigmoid function that the $$\sigma$$ function becomes very flat when $$\sigma(z_j^L)$$ is approximately 0 or 1. When this occurs we will have $$\sigma'(z_j^L) \approx 0$$. And so the lesson is that a weight in the final layer will learn slowly if the output neuron is either low activation ($$\approx 0$$) or high activation ($$\approx 1$$). In this case it's common to say the output neuron has saturated and, as a result, the weight has stopped learning (or is learning slowly). Similar remarks hold also for the biases of output neuron.

We can obtain similar insights for earlier layers. In particular, note the $$\sigma'(z^l)$$ term in (BP2). This means that $$\delta_j^l$$ is likely to get small if the neuron is near saturation. And this, in turn, means that any weights input to a saturated neuron will learn slowly.

Summing up, we've learnt that a weight will learn slowly if either the input neuron is low-activation, or if the output neuron has saturated, i.e., is either high- or low-activation.

None of these observations is too greatly surprising. Still, they help improve our mental model of what's going on as a neural network learns. Furthermore, we can turn this type of reasoning around. The four fundamental equations turn out to hold for any activation function, not just the standard sigmoid function (that's because, as we'll see in a moment, the proofs don't use any special properties of $$\sigma$$). And so we can use these equations to design activation functions which have particular desired learning properties. As an example to give you the idea, suppose we were to choose a (non-sigmoid) activation function $$\sigma$$ so that $$\sigma'$$ is always positive, and never gets close to zero. That would prevent the slow-down of learning that occurs when ordinary sigmoid neurons saturate. Later in the book we'll see examples where this kind of modification is made to the activation function. Keeping the four equations (BP1)-(BP4) in mind can help explain why such modifications are tried, and what impact they can have.

![bp1](/mldl/assets/images/bp1.jpg)

## Proof of the four fundamental equations

We'll now prove the four fundamental equations (BP1)-(BP4). All four are consequences of the chain rule from multivariable calculus.

Let's begin with Equation (BP1), which gives an expression for the output error, $$\delta^L$$. To prove this equation, recall that by definition:

{% raw %}
$$
    \delta_j^L = \frac{\delta C}{\delta z_j^L}
$$
{% endraw %}

Applying the chain rule, we can re-express the partial derivative above in terms of partial derivatives with respect to the output activations,

{% raw %}
$$
    \delta_j^L = \sum_k \frac{\delta C}{\delta a_k^L} \frac{\delta a_k^L}{\delta z_j^L}
$$
{% endraw %}

where the sum is over all neurons $$k$$ in the output layer. Of course, the output activation $$$$ $$a_k^L$$ of the $$k^{th}$$ neuron depends only on the weighted input $$z_j^L$$ for the $$j^{th}$$ neuron when $$k=j$$. And so $$\frac{\delta a_k^L}{\delta z_j^L}$$ vanishes when $$k \neq j$$. As a result we can simplify the previous equation to:

{% raw %}
$$
    \delta_j^L = \frac{\delta C}{\delta a_j^L} \frac{\delta a_j^L}{\delta z_j^L}
$$
{% endraw %}

Recalling that $$a_j^L=\sigma (z_j^L)$$ the second term on the right can be written as $$\sigma' (z_j^L)$$, and the equation becomes

{% raw %}
$$
    \delta_j^L = \frac{\delta C}{\delta a_j^L} \sigma' (z_j^L)
$$
{% endraw %}

which is just **(BP1)**, in component form.

Next, we'll prove (BP2), which gives an equation for the error $$\delta^l$$ in terms of the error in the next layer, $$\delta^{l+1}$$. To do this, we want to rewrite $$\delta_j^l = \frac{\delta C}{\delta z_j^l}$$ in terms of $$\delta_k^{l+1} = \frac{\delta C}{\delta z_k^{l+1}}$$. We can do this using the chain rule,

{% raw %}
$$
    \delta_j^l = \frac{\delta C}{\delta z_j^l} \\
               = \sum_k{\frac{\delta C}{\delta z_k^{l+1}} \frac{\delta z_k^{l+1}}{\delta z_j^l}} \\
               = \sum_k{\frac{\delta z_k^{l+1}}{\delta z_j^l} \delta_k^{l+1}}
$$
{% endraw %}

where in the last line we have interchanged the two terms on the right-hand side, and substituted the definition of $$\delta_k^{l+1}$$. To evaluate the first term on the last line, note that:

{% raw %}
$$
    z_k^{l+1} = \sum_j{w_{kj}^{l+1}a_j^l} + b_k^{l+1} = \sum_j{w_{kj}^{l+1}\sigma(z_j^l)} + b_k^{l+1}
$$
{% endraw %}

Differentiating, we obtain:

{% raw %}
$$
    \frac{\delta z_k^{l+1}}{\delta z_j^l} = w_{kj}^{l+1} \sigma'(z_j^l)
$$
{% endraw %}

Substituting back into the above equation we obtain:

{% raw %}
$$
    \delta_j^l = \sum_k{w_{kj}^{l+1}\delta_k^{l+1}\sigma'(z_j^l)}
$$
{% endraw %}

This is just **(BP2)** written in component form.

Then, we'll prove (BP3) and (BP4) using the chain rule.

{% raw %}
$$
    \frac{\delta C}{\delta b_j^l} = \sum_k{\frac{\delta C}{\delta z_k^l} \frac{\delta z_k^l}{\delta b_j^l}} = \sum_k{\frac{\delta z_k^l}{\delta b_j^l}\delta_k^l}
$$
{% endraw %}

and

{% raw %}
$$
    \frac{\delta C}{\delta w_{jk}^l} = \sum_k{\frac{\delta C}{\delta z_k^l} \frac{\delta z_k^l}{\delta w_{jk}^l}} = \sum_k{\frac{\delta z_k^l}{\delta w_{jk}^l}\delta_k^l}
$$
{% endraw %}

To evaluate the first terms, note that:

{% raw %}
$$
    z_k^{l} = \sum_j{w_{kj}^{l}a_j^{l-1}} + b_k^{l} = \sum_j{w_{kj}^{l}\sigma(z_j^{l-1})} + b_k^{l}
$$
{% endraw %}

Differentiating, we obtain:

{% raw %}
$$
    \frac{\delta z_k^{l}}{\delta b_j^l} = 1
$$
{% endraw %}

and

{% raw %}
$$
    \frac{\delta z_k^{l}}{\delta w_{jk}^l} = \sigma (z_k^{l-1}) = a_k^{l-1}
$$
{% endraw %}

Substituting back into the above equations we obtain:

{% raw %}
$$
    \frac{\delta C}{\delta b_j^l} = \delta_j^l
$$
{% endraw %}

and

{% raw %}
$$
    \frac{\delta C}{\delta w_{jk}^l} = a_k^{l-1}\sigma_j^l
$$
{% endraw %}

These are the **(BP3)** and **(BP4)**.

That completes the proof of the four fundamental equations of backpropagation. The proof may seem complicated. But it's really just the outcome of carefully applying the chain rule. A little less succinctly, we can think of backpropagation as a way of computing the gradient of the cost function by systematically applying the chain rule from multi-variable calculus. That's all there really is to backpropagation - the rest is details.

## The backpropagation algorithm

The backpropagation equations provide us with a way of computing the gradient of the cost function. Let's explicitly write this out in the form of an algorithm:

1. **Input**: $$x$$: Set the corresponding activation $$a^1$$ for the input layer.
2. **Feedforward**: For each $$l=2,3...,L$$ compute $$z^l = w^la^{l-1}+b^l$$ and $$a^l=\sigma(z^l)$$
3. **Output error**: $$\delta^L$$: Compute the vector $$\delta^L = \nabla_a C \odot \sigma'(z^L)$$
4. **Backpropagate the error**: For each $$l=L-1, L-2,...,2$$ compute $$\delta^l=((w^{l+1})^T\delta^{l+1}) \odot \sigma'(z^l)$$
5. **Output**: The gradient of the cost function is given by $$\frac{\delta C}{\delta w_{jk}^l}=a_k^{l-1}\delta_j^l$$ and $$\frac{\delta C}{\delta b_{j}^l}=\delta_j^l$$.

Examining the algorithm you can see why it's called backpropagation. We compute the error vectors $$\delta^l$$ backward, starting from the final layer. It may seem peculiar that we're going through the network backward. But if you think about the proof of backpropagation, the backward movement is a consequence of the fact that the cost is a function of outputs from the network. To understand how the cost varies with earlier weights and biases we need to repeatedly apply the chain rule, working backward through the layers to obtain usable expressions.

As I've described it above, the backpropagation algorithm computes the gradient of the cost function for a single training example, $$C=C_x$$. In practice, it's common to combine backpropagation with a learning algorithm such as stochastic gradient descent, in which we compute the gradient for many training examples. In particular, given a mini-batch of $$m$$ training examples, the following algorithm applies a gradient descent learning step based on that mini-batch:

1.**Input a set of training examples**

2.**For each training example**

$$x$$: Set the corresponding input activation $$a^{x,1}$$, and perform the following steps:

a. **Feedforward**: For each $$l=2,3,...,L$$ compute $$z^{x,l} = w^la^{x,l-1}+b^l$$ and $$a^{x,l}=\sigma(z^{x,l})$$

b. **Output error**: Compute the vector $$\delta^{x,L} = \nabla_a C_x \odot \sigma'(z^{x,L})$$

c. **Backpropagate the error**: For each $$l=L-1, L-2,...,2$$ compute $$\delta^{x,l}=((w^{l+1})^T\delta^{x,l+1}) \odot \sigma'(z^{x,l})$$

3.**Gradient descent**: For each $$l=L, L-1, L-2,...,2$$ update the weights according to the rule $$w^l->w^l - \frac{\eta}{m} \sum_x{\delta^{x,l}(a^{x,l-1})^T}$$, and the biases according to the rule $$b^l-> b^l - \frac{\eta}{m} \sum_x{\delta^{x,l}}$$.

Of course, to implement stochastic gradient descent in practice you also need an outer loop generating mini-batches of training examples, and an outer loop stepping through multiple epochs of training. I've omitted those for simplicity.

References:
* [<u>How the backpropagation algorithm works</u>](http://neuralnetworksanddeeplearning.com/chap2.html)
