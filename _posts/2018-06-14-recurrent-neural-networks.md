---
layout:         post
title:          Recurrent Neural Networks
subtitle:
card-image:     /mldl/assets/images/cards/cat11.gif
date:           2018-06-14 09:00:00
tags:           [deep&nbsp;learning]
categories:     [deep&nbsp;learning]
post-card-type: image
mathjax:        true
---

## What are RNNs?

The idea behind RNNs is to make use of sequential information. In a traditional neural network we assume that all inputs (and outputs) are independent of each other. But for many tasks that’s a very bad idea. If you want to predict the next word in a sentence you better know which words came before it. RNNs are called recurrent because they perform the same task for every element of a sequence, with the output being depended on the previous computations. Another way to think about RNNs is that they have a “memory” which captures information about what has been calculated so far. In theory RNNs can make use of information in arbitrarily long sequences, but in practice they are limited to looking back only a few steps (more on this later). Here is what a typical RNN looks like:

![rnn1](/mldl/assets/images/2018-06-14/rnn1.jpg)

The above diagram shows a RNN being unrolled (or unfolded) into a full network. By unrolling we simply mean that we write out the network for the complete sequence. For example, if the sequence we care about is a sentence of 5 words, the network would be unrolled into a 5-layer neural network, one layer for each word. The formulas that govern the computation happening in a RNN are as follows:

* $$x_t$$ is the input at time step $$t$$. For example, $$x_1$$ could be a one-hot vector corresponding to the second word of a sentence.
* $$s_t$$ is the hidden state at time step $$t$$. It’s the “memory” of the network. $$s_t$$ is calculated based on the previous hidden state and the input at the current step: $$s_t = f(Ux_t + Ws_{t-1})$$. The function $$f$$ usually is a nonlinearity such as tanh or ReLU. $$s_{-1}$$, which is required to calculate the first hidden state, is typically initialized to all zeroes.
* $$o_t$$ is the output at step $$t$$. For example, if we wanted to predict the next word in a sentence it would be a vector of probabilities across our vocabulary: $$o_t = softmax(Vs_t)$$.

There are a few things to note here:

* You can think of the hidden state $$s_t$$ as the memory of the network. $$s_t$$ captures information about what happened in all the previous time steps. The output at step $$o_t$$ is calculated solely based on the memory at time $$t$$. As briefly mentioned above, it’s a bit more complicated in practice because $$s_t$$ typically can’t capture information from too many time steps ago.
* Unlike a traditional deep neural network, which uses different parameters at each layer, a RNN shares the same parameters (U, V, W above) across all steps. This reflects the fact that we are performing the same task at each step, just with different inputs. This greatly reduces the total number of parameters we need to learn.
* The above diagram has outputs at each time step, but depending on the task this may not be necessary. For example, when predicting the sentiment of a sentence we may only care about the final output, not the sentiment after each word. Similarly, we may not need inputs at each time step. The main feature of an RNN is its hidden state, which captures some information about a sequence.

## Training RNNs: Backpropagation Through Time (BPTT)

Let’s quickly recap the basic equations of our RNN. Note that there’s a slight change in notation from $$o$$ to $$\hat{y}$$. That’s only to stay consistent with some of the literature out there that I am referencing.

{% raw %}
$$
    s_t = tanh(Ux_t + Ws_{t-1}) \\
    \hat{y_t} = softmax(Vs_t)
$$
{% endraw %}

We also defined our loss, or error, to be the cross entropy loss, given by:

{% raw %}
$$
    E(y,\hat{y}) = - \sum_t{y_tlog\hat{y_t}}
$$
{% endraw %}

Here, $$y_t$$ is the correct word at time step $$t$$, and $$\hat{y_t}$$ is our prediction. We typically treat the full sequence (sentence) as one training example, so the total error is just the sum of the errors at each time step (word).

Remember that our goal is to calculate the gradients of the error with respect to our parameters $$U$$, $$V$$ and $$W$$ and then learn good parameters using Stochastic Gradient Descent. Just like we sum up the errors, we also sum up the gradients at each time step for one training example, e.g. $$\frac{\delta E}{\delta W} = \sum_t{\frac{\delta E_t}{\delta W}}$$.

To calculate these gradients we use the chain rule of differentiation. That’s the backpropagation algorithm when applied backwards starting from the error. For the rest of this post we’ll use $$E_3$$ as an example, just to have concrete numbers to work with.

{% raw %}
$$
    \frac{\delta E_3}{\delta V} = \frac{\delta E_3}{\delta \hat{y_3}} \frac{\delta \hat{y_3}}{\delta V} = \frac{\delta E_3}{\delta \hat{y_3}} \frac{\delta \hat{y_3}}{\delta z_3} \frac{\delta z_3}{\delta V}
$$
{% endraw %}

In the above, we know:

{% raw %}
$$
    E_3 = - \sum_t^3{y_tlog\hat{y_t}} \\
    \hat{y_3} = softmax(Vs_3) \\
    z_3 = Vs_3
$$
{% endraw %}

Taking derivatives, we could get:

{% raw %}
$$
    \frac{\delta E_3}{\delta z_3} = \hat{y_3} - y_3 \\
    \frac{\delta z_3}{\delta V} = s_3
$$
{% endraw %}

Thus, we get:

{% raw %}
$$
    \frac{\delta E_3}{\delta V} = (\hat{y_3} - y_3) \odot s_3
$$
{% endraw %}

The point I’m trying to get across is that $$\frac{\delta E_3}{\delta V}$$ only depends on the values at the current time step, $$\hat{y_3}, y_3, s_3$$. If you have these, calculating the gradient for $$V$$ is a simple matrix multiplication.

But the story is different for $$\frac{\delta E_3}{\delta W}$$. To see why, we write out the chain rule, just as above:

{% raw %}
$$
    \frac{\delta E_3}{\delta W} = \frac{\delta E_3}{\delta \hat{y_3}} \frac{\delta \hat{y_3}}{\delta s_3} \frac{\delta s_3}{\delta W}
$$
{% endraw %}

Now, note that $$s_3=tanh(Ux_3 + Ws_2)$$ depends on $$s_2$$, which depends on $$W$$ and $$s_1$$, and so on. So if we take the derivative with respect to $$W$$ we can’t simply treat $$s_2$$ as a constant! We need to apply the chain rule again and what we really have is this:

{% raw %}
$$
    \frac{\delta E_3}{\delta W} = \sum_{k=0}^{3}\frac{\delta E_3}{\delta \hat{y_3}} \frac{\delta \hat{y_3}}{\delta s_3} \frac{\delta s_3}{\delta s_k} \frac{\delta s_k}{\delta W}
$$
{% endraw %}

We sum up the contributions of each time step to the gradient. In other words, because $$W$$ is used in every step up to the output we care about, we need to backpropagate gradients from $$t=3$$ through the network all the way to $$t=0$$.

Note that this is exactly the same as the standard backpropagation algorithm. The key difference is that we sum up the gradients for $$W$$ at each time step. In a traditional NN we don’t share parameters across layers, so we don’t need to sum anything. But in my opinion BPTT is just a fancy name for standard backpropagation on an unrolled RNN.

### The Vanishing Gradient Problem

I mentioned that RNNs have difficulties learning long-range dependencies – interactions between words that are several steps apart. That’s problematic because the meaning of an English sentence is often determined by words that aren’t very close: “The man who wore a wig on his head went inside”. The sentence is really about a man going inside, not about the wig. But it’s unlikely that a plain RNN would be able capture such information. To understand why, let’s take a closer look at the gradient we calculated above:

{% raw %}
$$
    \frac{\delta E_3}{\delta W} = \sum_{k=0}^{3}\frac{\delta E_3}{\delta \hat{y_3}} \frac{\delta \hat{y_3}}{\delta s_3} \frac{\delta s_3}{\delta s_k} \frac{\delta s_k}{\delta W}
$$
{% endraw %}

Note that $$\frac{\delta s_3}{\delta s_k}$$ is a chain rule in itself! For example, $$\frac{\delta s_3}{\delta s_1} = \frac{\delta s_3}{\delta s_2} \frac{\delta s_2}{\delta s_1}$$. Also note that because we are taking the derivative of a vector function with respect to a vector, the result is a matrix (called the Jacobian matrix) whose elements are all the pointwise derivatives. We can rewrite the above gradient:

{% raw %}
$$
    \frac{\delta E_3}{\delta W} = \sum_{k=0}^{3}\frac{\delta E_3}{\delta \hat{y_3}} \frac{\delta \hat{y_3}}{\delta s_3} (\prod_{j=k+1}^3 \frac{\delta s_j}{\delta s_{j-1}}) \frac{\delta s_k}{\delta W}
$$
{% endraw %}

It turns out that the 2-norm, which you can think of it as an absolute value, of the above Jacobian matrix has an upper bound of 1. This makes intuitive sense because our tanh (or sigmoid) activation function maps all values into a range between -1 and 1, and the derivative is bounded by 1 (1/4 in the case of sigmoid) as well.

We know that tanh and sigmoid functions have derivatives of 0 at both ends. They approach a flat line. When this happens we say the corresponding neurons are saturated. They have a zero gradient and drive other gradients in previous layers towards 0. Thus, with small values in the matrix and multiple matrix multiplications ($$t-k$$ in particular) the gradient values are shrinking exponentially fast, eventually vanishing completely after a few time steps. Gradient contributions from “far away” steps become zero, and the state at those steps doesn’t contribute to what you are learning: You end up not learning long-range dependencies. Vanishing gradients aren’t exclusive to RNNs. They also happen in deep Feedforward Neural Networks. It’s just that RNNs tend to be very deep (as deep as the sentence length in our case), which makes the problem a lot more common.

Fortunately, there are a few ways to combat the vanishing gradient problem. Proper initialization of the $$W$$ matrix can reduce the effect of vanishing gradients. So can regularization. A more preferred solution is to use ReLU instead of tanh or sigmoid activation functions. The ReLU derivative is a constant of either 0 or 1, so it isn’t as likely to suffer from vanishing gradients. An even more popular solution is to use Long Short-Term Memory (LSTM) or Gated Recurrent Unit (GRU) architectures. LSTMs were first proposed in 1997 and are the perhaps most widely used models in NLP today. GRUs, first proposed in 2014, are simplified versions of LSTMs. Both of these RNN architectures were explicitly designed to deal with vanishing gradients and efficiently learn long-range dependencies. We'll talk about in the next section.

## RNN Extensions

Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. They were introduced by Hochreiter & Schmidhuber (1997), and were refined and popularized by many people in following work. They work tremendously well on a large variety of problems, and are now widely used.

LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!

### LSTM

#### LSTM Networks

Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. They were introduced by Hochreiter & Schmidhuber (1997), and were refined and popularized by many people in following work. They work tremendously well on a large variety of problems, and are now widely used.

LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!

All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer.

![rnn2](/mldl/assets/images/2018-06-14/rnn2.jpg)

LSTMs also have this chain like structure, but the repeating module has a different structure. Instead of having a single neural network layer, there are four, interacting in a very special way.

![lstm1](/mldl/assets/images/2018-06-14/lstm1.jpg)

In the above diagram, each line carries an entire vector, from the output of one node to the inputs of others. The pink circles represent pointwise operations, like vector addition, while the yellow boxes are learned neural network layers. Lines merging denote concatenation, while a line forking denote its content being copied and the copies going to different locations.

#### The Core Idea Behind LSTMs

The key to LSTMs is the cell state, the horizontal line running through the top of the diagram. The cell state is kind of like a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions. It’s very easy for information to just flow along it unchanged.

The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates. Gates are a way to optionally let information through. They are composed out of a sigmoid neural net layer and a pointwise multiplication operation.

The sigmoid layer outputs numbers between zero and one, describing how much of each component should be let through. A value of zero means “let nothing through,” while a value of one means “let everything through!”. An LSTM has three of these gates, to protect and control the cell state.

#### Step-by-Step LSTM Walk Through

The first step in our LSTM is to decide what information we’re going to throw away from the cell state. This decision is made by a sigmoid layer called the “forget gate layer.” It looks at $$h_{t-1}$$ and $$x_t$$, and outputs a number between 0 and 1 for each number in the cell state $$C{t-1}$$. A 1 represents “completely keep this” while a 0 represents “completely get rid of this.”

Let’s go back to our example of a language model trying to predict the next word based on all the previous ones. In such a problem, the cell state might include the gender of the present subject, so that the correct pronouns can be used. When we see a new subject, we want to forget the gender of the old subject.

![lstm2](/mldl/assets/images/2018-06-14/lstm2.jpg)

The next step is to decide what new information we’re going to store in the cell state. This has two parts. First, a sigmoid layer called the “input gate layer” decides which values we’ll update. Next, a tanh layer creates a vector of new candidate values, $$\tilde{C_t}$$, that could be added to the state. In the next step, we’ll combine these two to create an update to the state.

In the example of our language model, we’d want to add the gender of the new subject to the cell state, to replace the old one we’re forgetting.

![lstm3](/mldl/assets/images/2018-06-14/lstm3.jpg)

It’s now time to update the old cell state, $$C_{t-1}$$, into the new cell state $$C_t$$. The previous steps already decided what to do, we just need to actually do it.

We multiply the old state by $$f_t$$, forgetting the things we decided to forget earlier. Then we add $$i_t * \tilde{C_t}$$. This is the new candidate values, scaled by how much we decided to update each state value.

In the case of the language model, this is where we’d actually drop the information about the old subject’s gender and add the new information, as we decided in the previous steps.

![lstm4](/mldl/assets/images/2018-06-14/lstm4.jpg)

Finally, we need to decide what we’re going to output. This output will be based on our cell state, but will be a filtered version. First, we run a sigmoid layer which decides what parts of the cell state we’re going to output. Then, we put the cell state through tanh (to push the values to be between -1 and 1) and multiply it by the output of the sigmoid gate, so that we only output the parts we decided to.

For the language model example, since it just saw a subject, it might want to output information relevant to a verb, in case that’s what is coming next. For example, it might output whether the subject is singular or plural, so that we know what form a verb should be conjugated into if that’s what follows next.

![lstm5](/mldl/assets/images/2018-06-14/lstm5.jpg)

#### Variants on Long Short Term Memory

What I’ve described so far is a pretty normal LSTM. But not all LSTMs are the same as the above. In fact, it seems like almost every paper involving LSTMs uses a slightly different version. The differences are minor, but it’s worth mentioning some of them.

One popular LSTM variant, introduced by Gers & Schmidhuber (2000), is adding “peephole connections.” This means that we let the gate layers look at the cell state.

![lstm6](/mldl/assets/images/2018-06-14/lstm6.jpg)

The above diagram adds peepholes to all the gates, but many papers will give some peepholes and not others.

Another variation is to use coupled forget and input gates. Instead of separately deciding what to forget and what we should add new information to, we make those decisions together. We only forget when we’re going to input something in its place. We only input new values to the state when we forget something older.

![lstm7](/mldl/assets/images/2018-06-14/lstm7.jpg)

### GRU

The idea behind a GRU layer is quite similar to that of a LSTM layer, as are the equations.

{% raw %}
$$
    z_t = \sigma(x_tU^z + h_{t-1}W^z) \\
    r_t = \sigma(x_tU^r + h_{t-1}W^r) \\
    \tilde{h_t} = tanh(x_tU^h + (h_{t-1} r_t)W^h) \\
    h_t = (1-z_t) h_{t-1} + z_t \tilde{h_t}
$$
{% endraw %}

A GRU has two gates, a reset gate $$r$$, and an update gate $$z$$. Intuitively, the reset gate determines how to combine the new input with the previous memory, and the update gate defines how much of the previous memory to keep around. If we set the reset to all 1’s and  update gate to all 0’s we again arrive at our plain RNN model. The basic idea of using a gating mechanism to learn long-term dependencies is the same as in a LSTM, but there are a few key differences:

* A GRU has two gates, an LSTM has three gates.
* GRUs don’t possess an internal memory ($$c_t$$) that is different from the exposed hidden state. They don’t have the output gate that is present in LSTMs.
* The input and forget gates are coupled by an update gate $$z$$ and the reset gate $$r$$ is applied directly to the previous hidden state. Thus, the responsibility of the reset gate in a LSTM is really split up into both $$r$$ and $$z$$.
* We don’t apply a second nonlinearity when computing the output.

![gru](/mldl/assets/images/2018-06-14/gru.jpg)

### Bidirectional RNNs

Bidirectional RNNs combine an RNN that moves forward through time beginning from the start of the sequence with another RNN that moves backward through time beginning from the end of the sequence and concatenate the resulting outputs (both cell outputs and final hidden states).

The below image illustrates the typical bidirectional RNN, where $$h(t)$$ and $$g(t)$$ standing for the (hidden) state of the sub-RNN that moves forward and backward through time, respectively. This allows the output units $$o(t)$$ to compute a representation that depends on both the past and the future but is most sensitive to the input values around time $$t$$.

![birnn](/mldl/assets/images/2018-06-14/birnn.jpg)

### Deep (Bidirectional) RNNs

Deep (Bidirectional) RNNs are similar to Bidirectional RNNs, only that we now have multiple layers per time step. In practice this gives us a higher learning capacity (but we also need a lot of training data).

References:

* [<u>Recurrent Neural Networks Tutorial, Part 1 – Introduction to RNNs</u>](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)
* [<u>Derivation of RNN gradients</u>](https://github.com/go2carter/nn-learn/blob/master/grad-deriv-tex/rnn-grad-deriv.pdf)
* [<u>Understanding LSTM Networks</u>](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* [<u>Standford CS224n course notes</u>](https://github.com/stanfordnlp/cs224n-winter17-notes)
