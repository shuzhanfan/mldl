---
layout:         post
title:          Reinforcement Learning
subtitle:
card-image:     /mldl/assets/images/cards/cat15.gif
date:           2018-06-22 09:00:00
tags:           [deep&nbsp;learning]
categories:     [deep&nbsp;learning]
post-card-type: image
mathjax:        true
---

## Why Study Reinforcement Learning

Reinforcement Learning is one of the fields I’m most excited about. Over the past few years amazing results like learning to play Atari Games from raw pixels and Mastering the Game of Go have gotten a lot of attention, but RL is also widely used in Robotics, Image Processing and Natural Language Processing.

Combining Reinforcement Learning and Deep Learning techniques works extremely well. Both fields heavily influence each other. On the Reinforcement Learning side Deep Neural Networks are used as function approximators to learn good representations, e.g. to process Atari game images or to understand the board state of Go. In the other direction, RL techniques are making their way into supervised problems usually tackled by Deep Learning. For example, RL techniques are used to implement attention mechanisms in image processing, or to optimize long-term rewards in conversational interfaces and neural translation systems. Finally, as Reinforcement Learning is concerned with making optimal decisions it has some extremely interesting parallels to human Psychology and Neuroscience (and many other fields).

With lots of open problems and opportunities for fundamental research I think we’ll be seeing multiple Reinforcement Learning breakthroughs in the coming years. And what could be more fun than teaching machines to play Starcraft and Doom?

## How to Study Reinforcement Learning

There are many excellent Reinforcement Learning resources out there. Two I recommend the most are:

* [<u>David Silver’s Reinforcement Learning Course</u>](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
* [<u>Richard Sutton’s & Andrew Barto’s Reinforcement Learning: An Introduction (2nd Edition) book</u>](http://incompleteideas.net/sutton/book/bookdraft2017june.pdf)

The latter is still work in progress but it’s ~80% complete. The course is based on the book so the two work quite well together. In fact, these two cover almost everything you need to know to understand most of the recent research papers. The prerequisites are basic Math and some knowledge of Machine Learning.

That covers the theory. But what about practical resources? What about actually implementing the algorithms that are covered in the book/course? That’s where this post and the [<u>Github repository</u>](https://github.com/dennybritz/reinforcement-learning) comes in. I’ve tried to implement most of the standard Reinforcement Algorithms using Python, OpenAI Gym and Tensorflow. I separated them into chapters (with brief summaries) and exercises and solutions so that you can use them to supplement the theoretical material above. All of this is in the Github repository.

Some of the more time-intensive algorithms are still work in progress, so feel free to contribute. I’ll update this post as I implement them.

## Table of Contents

### **[<u>Introduction to RL problems, OpenAI gym</u>](https://github.com/dennybritz/reinforcement-learning/tree/master/Introduction/)**

### **[<u>MDPs and Bellman Equations</u>]()**


### **Dynamic Programming: Model-Based RL, Policy Iteration and Value Iteration**
### **Monte Carlo Model-Free Prediction & Control**
### **Temporal Difference Model-Free Prediction & Control**
### **Function Approximation**
### **Deep Q Learning (WIP)**
### **Policy Gradient Methods (WIP)**
### **Learning and Planning (WIP)**
### **Exploration and Exploitation (WIP)**


References:

* [<u>Learning Reinforcement Learning (with Code, Exercises and Solutions)</u>](http://www.wildml.com/2016/10/learning-reinforcement-learning/)
* [<u>David Silver’s Reinforcement Learning Course</u>](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
* [<u>Richard Sutton’s & Andrew Barto’s Reinforcement Learning: An Introduction (2nd Edition) book</u>](http://incompleteideas.net/sutton/book/bookdraft2017june.pdf)
* [<u></u>]()
* [<u></u>]()
