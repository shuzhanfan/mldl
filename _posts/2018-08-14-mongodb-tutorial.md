---
layout:         post
title:          MongoDB Tutorial
subtitle:
card-image:     /mldl/assets/images/cards/cat13.gif
date:           2018-08-06 09:00:00
tags:           [python]
categories:     [python]
post-card-type: image
mathjax:        true
---

## MongoDB and Deep Learning

### Part 1: The history of AI

Deep learning and Artificial Intelligence (AI) have moved well beyond science fiction into the cutting edge of internet and enterprise computing. Access to more computational power in the cloud, advancement of sophisticated algorithms, and the availability of funding are unlocking new possibilities unimaginable just five years ago. But it’s the availability of new, rich data sources that is making deep learning real.

In this 4-part blog series, we are going to explore deep learning, and the role database selection plays in successfully applying deep learning to business problems:

* In part 1 we will look at the history of AI, and why it is taking off now
* In part 2, we will discuss the differences between AI, Machine Learning, and Deep Learning
* In part 3, we’ll dive deeper into deep learning and evaluate key considerations when selecting a database for new projects.
* We’ll wrap up in part 4 with a discussion on why MongoDB is being used for deep learning, and provide examples of where it is being used.

#### The History of Artificial Intelligence

We are living in an era where artificial intelligence (AI) has started to scratch the surface of its true potential. Not only does AI create the possibility of disrupting industries and transforming the workplace, but it can also address some of society’s biggest challenges. Autonomous vehicles may save tens of thousands of lives, and increase mobility for the elderly and the disabled. Precision medicine may unlock tailored individual treatment that extends life. Smart buildings may help reduce carbon emissions and save energy. These are just a few of the potential benefits that AI promises, and is starting to deliver upon.

By 2018, Gartner estimates that machines will author 20% of all business content, and an expected 6 billion IoT-connected devices will be generating a deluge of data. AI will be essential to make sense of it all. No longer is AI confined to science fiction movies; artificial intelligence and machine learning are finding real world applicability and adoption.

Artificial intelligence has been a dream for many ever since Alan Turing wrote his seminal 1950 paper “Computing Machinery and Intelligence”. In Turing’s paper, he asked the fundamental question, “Can Machines Think?” and contemplated the concept of whether computers could communicate like humans. The birth of the AI field really started in the summer of 1956, when a group of researchers came together at Dartmouth College to initiate a series of research projects aimed at programming computers to behave like humans. It was at Dartmouth where the term “artificial intelligence” was first coined, and concepts from the conference crystallized to form a legitimate interdisciplinary research area.

Over the next decade, progress in AI experienced boom and bust cycles as advances with new algorithms were constrained by the limitations of contemporary technologies. In 1968, the science fiction film 2001: A Space Odyssey helped AI leave an indelible impression in mainstream consciousness when a sentient computer – HAL 9000 – uttered the famous line, “I’m sorry Dave, I’m afraid I can’t do that.” In the late 1970s, Star Wars further cemented AI in mainstream culture when a duo of artificially intelligent robots (C-3PO and R2-D2) helped save the galaxy.

But it wasn’t until the late 1990s that AI began to transition from science fiction lore into real world applicability. Beginning in 1997 with IBM’s Deep Blue chess program beating then current world champion Garry Kasparov, the late 1990s ushered in a new era of AI in which progress started to accelerate. Researchers began to focus on sub-problems of AI and harness it to solve real world applications such as image recognition and speech. Instead of trying to structure logical rules determined by the knowledge of experts, researchers started to work on how algorithms could learn the logical rules themselves. This trend helped to shift research focus into Artificial Neural Networks (ANNs). First conceptualized in the 1940s, ANNs were invented to “loosely” mimic how the human brain learns. ANNs experienced a resurgence in popularity in 1986 when the concept of backpropagation gradient descent was improved. The backpropagation method reduced the huge number of permutations needed in an ANN, and thus was a more efficient way to reduce AI training time.

Even with advances in new algorithms, neural networks still suffered from limitations with technology that had plagued their adoption over the previous decades. It wasn’t until the mid 2000s that another wave of progress in AI started to take form. In 2006, Geoffrey Hinton of the University of Toronto made a modification to ANNs, which he called deep learning (deep neural networks). Hinton added multiple layers to ANNs and mathematically optimized the results from each layer so that learning accumulated faster up the stack of layers. In 2012, Andrew Ng of Stanford University took deep learning a step further when he built a crude implementation of deep neural networks using Graphical Processing Units (GPUs). Since GPUs have a massively parallel architecture that consist of thousands of cores designed to handle multiple tasks simultaneously, Ng found that a cluster of GPUs could train a deep learning model much faster than general purpose CPUs. Rather than take weeks to generate a model with traditional CPUs, he was able to perform the same task in a day with GPUs.

Essentially, this convergence – advances in software algorithms combined with highly performant hardware – had been brewing for decades, and would usher in the rapid progress AI is currently experiencing.

#### Why Is AI Taking Off Now?

There are four main factors driving the adoption of AI today:

**More Data.** AI needs a huge amount of data to learn, and the digitization of society is providing the available raw material to fuel its advances. Big data from sources such as Internet of Things (IoT) sensors, social and mobile computing, science and academia, healthcare, and many more new applications generate data that can be used to train AI models. Not surprisingly, the companies investing most in AI – Amazon, Apple, Baidu, Google, Microsoft, Facebook – are the ones with the most data.

**Cheaper Computation.** In the past, even as AI algorithms improved, hardware remained a constraining factor. Recent advances in hardware and new computational models, particularly around GPUs, have accelerated the adoption of AI. GPUs gained popularity in the AI community for their ability to handle a high degree of parallel operations and perform matrix multiplications in an efficient manner – both are necessary for the iterative nature of deep learning algorithms. Subsequently, CPUs have also made advances for AI applications. Recently, Intel added new deep learning instructions to its Xeon and Xeon Phi processors to allow for better parallelization and more efficient matrix computation. This is coupled with improved tools and software frameworks from it’s software development libraries. With the adoption of AI, hardware vendors now also have the chip demand to justify and amortize the large capital costs required to develop, design, and manufacture products exclusively tailored for AI. These advancements result in better hardware designs, performance, and power usage profiles.

**More Sophisticated Algorithms.** Higher performance and less expensive compute also enable researchers to develop and train more advanced algorithms because they aren’t limited by the hardware constraints of the past. As a result, deep learning is now solving specific problems (e.g., speech recognition, image classification, handwriting recognition, fraud detection) with astonishing accuracy, and more advanced algorithms continue to advance the state of the art in AI.

**Broader Investment.** Over the past decades, AI research and development was primarily limited to universities and research institutions. Lack of funding combined with the sheer difficulty of the problems associated with AI resulted in minimal progress. Today, AI investment is no longer confined to university laboratories, but is pervasive in many areas – government, venture capital-backed startups, internet giants, and large enterprises across every industry sector.


### Part 2: Differences between AI, Machine Learning, and Deep Learning

In many contexts, artificial intelligence, machine learning and deep learning are used interchangeably, but in reality, machine and deep learning is a subset of AI. We can think of AI as the branch of computer science focused on building machines capable of intelligent behaviour, while machine and deep learning is the practice of using algorithms to sift through data, learn from the data, and make predictions or take autonomous actions. Therefore, instead of programming specific constraints for an algorithm to follow, the algorithm is trained using large amounts of data to give it the ability to independently learn, reason, and perform a specific task.

![ai1](/mldl/assets/images/2018-08-14/ai1.jpg)

So what’s the difference between machine learning and deep learning? Before defining deep learning – which we’ll do in part 3, let’s dig deeper into machine learning.

#### Machine Learning: Supervised vs. Unsupervised

There are two main classes of machine learning approaches: supervised learning and unsupervised learning.

**Supervised Learning**. Currently, supervised learning is the most common type of machine learning algorithm. With supervised learning, the algorithm takes input data manually labeled by developers and analysts, using it to train the model and generate predictions. Supervised learning can be delineated into two groups: regression and classification problems.

**Unsupervised Learning**. In our supervised learning example, labeled datasets (benign or malignant classifications) help the algorithm determine what the correct answer is. With unsupervised learning, we give the algorithm an unlabeled dataset and depend on the algorithm to uncover structures and patterns in the data.


### Part 3: Deep Learning

#### What is Deep Learning?

Deep learning is a subset of machine learning that has attracted worldwide attention for its recent success solving particularly hard and large-scale problems in areas such as speech recognition, natural language processing, and image classification. Deep learning is a refinement of ANNs, which, as discussed earlier, “loosely” emulate how the human brain learns and solves problems.

Before diving into how deep learning works, it’s important to first understand how ANNs work. ANNs are made up of an interconnected group of neurons, similar to the network of neurons in the brain.

At a simplistic level, a neuron in a neural network is a unit that receives a number of inputs (xi), performs a computation on the inputs, and then sends the output to other nodes or neurons in the network. Weights (wj), or parameters, represent the strength of the input connection and can be either positive or negative. The inputs are multiplied by the associated weights (x1w1, x2w2,..) and the neuron adds the output from all inputs. The final step is that a neuron performs a computation, or activation function. The activation function (sigmoid function is popular) allows an ANN to model complex nonlinear patterns that simpler models may not represent correctly.

The first layer is called the input layer and is where features (x1, x2, x3) are input. The second layer is called the hidden layer. Any layer that is not an input or output layer is a hidden layer. Deep learning was originally coined because of the multiple levels of hidden layers. Networks typically contain more than 3 hidden layers, and in some cases more than 1,200 hidden layers.

What is the benefit of multiple hidden layers? Certain patterns may need deeper investigation that can be surfaced with the additional hidden layers. Image classification is an area where deep learning can achieve high performance on very hard visual recognition tasks – even exceeding human performance in certain areas.

When a picture is input into a deep learning network, it is first decomposed into image pixels. The algorithm will then look for patterns of shapes at certain locations in the image. The first hidden layer might try to uncover specific facial patterns: eyes, mouth, nose, ears. Adding an additional hidden layer deconstructs the facial patterns into more granular attributes. For example, a “mouth” could be further deconstructed into “teeth”, “lips”, “gums”, etc. Adding additional hidden layers can devolve these patterns even further to recognize the subtlest nuances. The end result is that a deep learning network can break down a very complicated problem into a set of simple questions. The hidden layers are essentially a hierarchical grouping of different variables that provide a better defined relationship. Currently, most deep learning algorithms are supervised; thus, deep learning models are trained against a known truth.

#### How Does Training Work?

The purpose of training a deep learning model is to reduce the cost function, which is the discrepancy between the expected output and the real output. The connections between the nodes will have specific weights that need to be refined to minimize the discrepancy. By modifying the weights, we can minimize the cost function to its global minimum, which means we’ve reduced the error in our model to its lowest value. The reason deep learning is so computationally intensive is that it requires finding the correct set of weights within millions or billions of connections. This is where constant iteration is required as new sequences of weights are tested repeatedly to find the point where the cost function is at its global minimum.

A common technique in deep learning is to use backpropagation gradient descent. Gradient descent emerged as an efficient mathematical optimization that works effectively with a large number of dimensions (or features) without having to perform brute force dimensionality analysis. Gradient descent works by computing a gradient (or slope) in the direction of the function global minimum based on the weights. During training, weights are first randomly assigned, and an error is calculated. Based on the error, gradient descent will then modify the weights, backpropagate the updated weights through the multiple layers, and retrain the model such that the cost function moves towards the global minimum. This continues iteratively until the cost function reaches the global minimum. There may be instances where gradient descent resolves itself at a local minimum instead of global minimum. Methods to mitigate this issue is to use a convex function or generate more randomness for the parameters.

#### Database Considerations with Deep Learning

Non-relational databases have played an integral role in the recent advancement of the technology enabling machine learning and deep learning. The ability to collect and store large volumes of structured and unstructured data has provided deep learning with the raw material needed to improve predictions. When building a deep learning application, there are certain considerations to keep in mind when selecting a database for management of underlying data.

**Flexible Data Model.** In deep learning there are three stages where data needs to be persisted – input data, training data, and results data. Deep learning is a dynamic process that typically involves significant experimentation. For example, it is not uncommon for frequent modifications to occur during the experimentation process – tuning hyperparameters, adding unstructured input data, modifying the results output – as new information and insights are uncovered. Therefore, it is important to choose a database that is built on a flexible data model, avoiding the need to perform costly schema migrations whenever data structures need to change.

**Scale.** One of the biggest challenges of deep learning is the time required to train a model. Deep learning models can take weeks to train – as algorithms such as gradient descent may require many iterations of matrix operations involving billions of parameters. In order to reduce training times, deep learning frameworks try to parallelize the training workload across fleets of distributed commodity servers.

There are two main ways to parallelize training: data parallelism and model parallelism.

* **Data parallelism**. Splitting the data across many nodes for processing and storage, enabled by distributed systems such as Apache Spark, MongoDB, and Apache Hadoop
* **Model parallelism**. Splitting the model and its associated layers across many nodes, enabled by software libraries and frameworks such as Tensorflow, Caffe, and Theano. Splitting provides parallelism, but does incur a performance cost in coordinating outputs between different nodes

n addition to the model’s training phase, another big challenge of deep learning is that the input datasets are continuously growing, which increases the number of parameters to train against. Not only does this mean that the input dataset may exceed available server memory, but it also means that the matrices involved in gradient descent can exceed the node’s memory as well. Thus, scaling out, rather than scaling up, is important as this enables the workload and associated dataset to be distributed across multiple nodes, allowing computations to be performed in parallel.

**Fault Tolerance.** Many deep learning algorithms use checkpointing as a way to recover training data in the event of failure. However, frequent checkpointing requires significant system overhead. An alternative is to leverage the use of multiple data replicas hosted on separate nodes. These replicas provide redundancy and data availability without the need to consume resources on the primary node of the system.

**Consistency.** For most deep learning algorithms it is recommended to use a strong data consistency model. With strong consistency each node in a distributed database cluster is operating on the latest copy of the data, though some algorithms, such as Stochastic Gradient Descent (SGD), can tolerate a certain degree of inconsistency. Strong consistency will provide the most accurate results, but in certain situations where faster training time is valued over accuracy, eventual consistency is acceptable. To optimize for accuracy and performance, the database should offer tunable consistency.

### Part 4： Why MongoDB for Deep Learning?

If you haven’t read part 3, it’s worth visiting that post to learn more about the key considerations when selecting a database for new deep learning projects. As the following section demonstrates, developers and data scientists can harness MongoDB as a flexible, scalable, and performant distributed database to meet the rigors of AI application development.

#### Flexible Data Model

MongoDB's document data model makes it easy for developers and data scientists to store and combine data of any structure within the database, without giving up sophisticated validation rules to govern data quality. The schema can be dynamically modified without application or database downtime that results from costly schema modifications or redesign incurred by relational database systems.

This data model flexibility is especially valuable to deep learning, which involves constant experimentation to uncover new insights and predictions:

* Input datasets can comprise rapidly changing structured and unstructured data ingested from clickstreams, log files, social media and IoT sensor streams, CSV, text, images, video, and more. Many of these datasets do not map well into the rigid row and column formats of relational databases.
* The training process often involves adding new hidden layers, feature labels, hyperparameters, and input data, requiring frequent modifications to the underlying data model.

A database supporting a wide variety of input datasets, with the ability to seamlessly modify parameters for model training, is therefore essential.

#### Rich Programming and Query Model

MongoDB offers both native drivers and certified connectors for developers and data scientists building deep learning models with data from MongoDB. The PyMongo driver is the recommended way to work with MongoDB from Python, implementing an idiomatic API that makes development natural for Python programmers. The community developed MongoDB Client for R is also available for R programmers.

The MongoDB query language and rich secondary indexes enable developers to build applications that can query and analyze the data in multiple ways. Data can be accessed by single keys, ranges, text search, graph, and geospatial queries through to complex aggregations and MapReduce jobs, returning responses in milliseconds.

To parallelize data processing across a distributed database cluster, MongoDB provides the aggregation pipeline and MapReduce. The MongoDB aggregation pipeline is modeled on the concept of data processing pipelines. Documents enter a multi-stage pipeline that transforms the documents into an aggregated result using native operations executed within MongoDB. The most basic pipeline stages provide filters that operate like queries, and document transformations that modify the form of the output document. Other pipeline operations provide tools for grouping and sorting documents by specific fields as well as tools for aggregating the contents of arrays, including arrays of documents. In addition, pipeline stages can use operators for tasks such as calculating the average or standard deviations across collections of documents, and manipulating strings. MongoDB also provides native MapReduce operations within the database, using custom JavaScript functions to perform the map and reduce stages.

![ai2](/mldl/assets/images/2018-08-14/ai2.jpg)

The MongoDB Connector for Apache Spark can take advantage of MongoDB’s aggregation pipeline and secondary indexes to extract, filter, and process only the range of data it needs – for example, analyzing all customers located in a specific geography. This is very different from simple NoSQL datastores that do not support either secondary indexes or in-database aggregations. In these cases, Spark would need to extract all data based on a simple primary key, even if only a subset of that data is required for the Spark process. This means more processing overhead, more hardware, and longer time-to-insight for data scientists and engineers. To maximize performance across large, distributed data sets, the MongoDB Connector for Apache Spark can co-locate Resilient Distributed Datasets (RDDs) with the source MongoDB node, thereby minimizing data movement across the cluster and reducing latency.

#### Performance, Scalability & Redundancy

Model training time can be reduced by building the deep learning platform on top of a performant and scalable database layer. MongoDB offers a number of innovations to maximize throughput and minimize latency of deep learning workloads:

* WiredTiger is the default storage engine for MongoDB, developed by the architects of Berkeley DB, the most widely deployed embedded data management software in the world. WiredTiger scales on modern, multi-core architectures. Using a variety of programming techniques such as hazard pointers, lock-free algorithms, fast latching and message passing, WiredTiger maximizes computational work per CPU core and clock cycle. To minimize on-disk overhead and I/O, WiredTiger uses compact file formats and storage compression.
* For the most latency-sensitive deep learning applications, MongoDB can be configured with the In-Memory storage engine. Based on WiredTiger, this storage engine gives users the benefits of in-memory computing, without trading away the rich query flexibility, real-time analytics, and scalable capacity offered by conventional disk-based databases.
* To parallelize model training and scale input datasets beyond a single node, MongoDB uses a technique called sharding, which distributes processing and data across clusters of commodity hardware. MongoDB sharding is fully elastic, automatically rebalancing data across the cluster as the input dataset grows, or as nodes are added and removed.
* Within a MongoDB cluster, data from each shard is automatically distributed to multiple replicas hosted on separate nodes. MongoDB replica sets provide redundancy to recover training data in the event of a failure, reducing the overhead of checkpointing.


## MongoDB Atlas M0

Scroll to the bottom of the page and select an administration user for your database. This will be the database specific user you'll use to shell to your cluster or connect your application. You can select a username of your choice and then enter a password, we've also provided you a way to generate random passwords. Configuring a username and password is a default for all MongoDB Atlas Clusters ensuring your data is protected.

We'll be provided with a three node cluster with 512 megabytes of storage at absolutely no cost. Our new cluster will be created in the AWS us-east-1 region, this is the default for all M0's along with using MongoDB 3.4 with WiredTiger. This cannot be modified. Automated Backups are also not included, so consider doing database dumps on your M0 to preserve any data before terminating your cluster.  For a full list of all the limitations on the M0, please visit our documentation.

You'll see we can now launch our cluster, so let's click on CONFIRM & DEPLOY. The process to create your cluster will take about 5 - 10 minutes. You can use this time to configure your whitelist or maybe get a cup of coffee. Regardless, when you're done you'll be able to start building your application at no cost to you!

Now that your M0 is launched, what's next? Make sure you set up your IP Whitelist, configure additional database users and then start building your app. We made this as easy as possible by ensuring Atlas will handle the creation of all the database security, users and infrastructure at no cost. Some people say there's no such thing as a free lunch, well there's a free MongoDB Atlas cluster. That credit card of yours can stay where it is until you're ready to go to upgrade to a production ready cluster!


## MongoDB Aggregation

Aggregation operations process data records and return computed results. Aggregation operations group values from multiple documents together, and can perform a variety of operations on the grouped data to return a single result. MongoDB provides three ways to perform aggregation: the aggregation pipeline, the map-reduce function, and single purpose aggregation methods.

### Aggregation Pipeline

MongoDB’s aggregation framework is modeled on the concept of data processing pipelines. Documents enter a multi-stage pipeline that transforms the documents into an aggregated result.

The most basic pipeline stages provide filters that operate like queries and document transformations that modify the form of the output document. Other pipeline operations provide tools for grouping and sorting documents by specific field or fields as well as tools for aggregating the contents of arrays, including arrays of documents. In addition, pipeline stages can use operators for tasks such as calculating the average or concatenating a string.

The pipeline provides efficient data aggregation using native operations within MongoDB, and is the preferred method for data aggregation in MongoDB. The aggregation pipeline can use indexes to improve its performance during some of its stages. In addition, the aggregation pipeline has an internal optimization phase.

![ai3](/mldl/assets/images/2018-08-14/ai3.jpg)

Some common stages are:

Stage  | Description
-------| -----------
$match | Filters the document stream to allow only matching documents to pass unmodified into the next pipeline stage.
$group | Groups input documents by a specified identifier expression and applies the accumulator expression(s), if specified, to each group.
$sort  | Reorders the document stream by a specified sort key. Only the order changes; the documents remain unmodified.
$count | Returns a count of the number of documents at this stage of the aggregation pipeline.

`$group` groups documents by some specified expression and outputs to the next stage a document for each distinct grouping. The output documents contain an `_id` field which contains the distinct group by key. The output documents can also contain computed fields that hold the values of some accumulator expression grouped by the `$group`’s `_id` field. `$group` does not order its output documents.

The `$group` stage has the following prototype form:

```
{ $group: { _id: <expression>, <field1>: { <accumulator1> : <expression1> }, ... } }
```

The `_id` field is mandatory; however, you can specify an `_id` value of `null` to calculate accumulated values for all the input documents as a whole. The remaining computed fields are optional and computed using the `<accumulator>` operators. The `_id` and the `<accumulator>` expressions can accept any valid expression.

Some common `<accumulator>` operators are: `$avg`, `$max`, `$min`, `$sum`.

**Expressions** can include field paths and system variables, literals, expression objects, and expression operators. Expressions can be nested. Aggregation expressions use field path to access fields in the input documents. To specify a field path, use a string that prefixes with a dollar sign `$` the field name or the dotted field name, if the field is in embedded document. For example, `$user` to specify the field path for the `user` field or `$user.name` to specify the field path to `user.name` field.


## MongoDB Indexing

Indexes are very important in any database, and with MongoDB it's no different. With the use of Indexes, performing queries in MongoDB becomes more efficient.

If you had a collection with thousands of documents with no indexes, and then you query to find certain documents, then in such case MongoDB would need to scan the entire collection to find the documents. But if you had indexes, MongoDB would use these indexes to limit the number of documents that had to be searched in the collection.

Indexes are special data sets which store a partial part of the collection's data. Since the data is partial, it becomes easier to read this data. This partial set stores the value of a specific field or a set of fields ordered by the value of the field.

MongoDB creates a unique index on the `_id` field during the creation of a collection. The `_id` index prevents clients from inserting two documents with the same value for the `_id` field. You cannot drop this index on the `_id` field.


## MongoDB Connector for Apache Spark ([<u>documentation</u>](https://docs.mongodb.com/spark-connector/current/configuration/))

The MongoDB Connector for Apache Spark exposes all of Spark’s libraries, including Scala, Java, Python and R. MongoDB data is materialized as DataFrames and Datasets for analysis with machine learning, graph, streaming, and SQL APIs.

![mongodb1](/mldl/assets/images/2018-08-14/mongodb1.jpg)

The MongoDB Connector for Apache Spark can take advantage of MongoDB’s **aggregation pipeline** and **rich secondary indexes** to extract, filter, and process only the range of data it needs – for example, analyzing all customers located in a specific geography. This is very different from simple NoSQL datastores that do not offer secondary indexes or in-database aggregations. In these cases, Spark would need to extract all data based on a simple primary key, even if only a subset of that data is required for the Spark process. This means more processing overhead, more hardware, and longer time-to-insight for data scientists and engineers.

To maximize performance across large, distributed data sets, the MongoDB Connector for Apache Spark can co-locate Resilient Distributed Datasets (RDDs) with the source MongoDB node, thereby minimizing data movement across the cluster and reducing latency.

### Configuration

Various configuration options are available for the MongoDB Spark Connector.













References:

* [<u>Deep Learning and the Artificial Intelligence Revolution</u>](https://www.mongodb.com/blog/post/deep-learning-and-the-artificial-intelligence-revolution-part-1)
* [<u><MongoDB Atlas M0</u>](https://www.linkedin.com/pulse/introducing-m0-free-tier-mongodb-atlas-jay-gordon/)
