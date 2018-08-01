---
layout:         post
title:          Machine learning clustering algorithms
subtitle:
card-image:     /mldl/assets/images/cards/cat23.gif
date:           2018-07-12 09:00:00
tags:           [machine&nbsp;learning]
categories:     [machine&nbsp;learning, algorithms]
post-card-type: image
mathjax:        true
---

* <a href="#Introduction">Introduction</a>
* <a href="#Overview">Overview</a>
* <a href="#Types of Clustering">Types of Clustering</a>
* <a href="#Types of clustering algorithms">Types of clustering algorithms</a>
* <a href="#K-Means Clustering">K-Means Clustering</a>
* <a href="#Mean-Shift Clustering">Mean-Shift Clustering</a>
* <a href="#DBSCAN">DBSCAN</a>
* <a href="#Agglomerative Hierarchical Clustering">Agglomerative Hierarchical Clustering</a>
* <a href="#EM-GMM">EM-GMM</a>

## <a name="Introduction">Introduction</a>

Have you come across a situation when a Chief Marketing Officer of a company tells you – “Help me understand our customers better so that we can market our products to them in a better manner!”

I did and the analyst in me was completely clueless what to do! I was used to getting specific problems, where there is an outcome to be predicted for various set of conditions. But I had no clue, what to do in this case. If the person would have asked me to calculate Life Time Value (LTV) or propensity of Cross-sell, I wouldn’t have blinked. But this question looked very broad to me!

This is usually the first reaction when you come across a unsupervised learning problem for the first time! You are not looking for specific insights for a phenomena, but what you are looking for are structures with in data with out them being tied down to a specific outcome.

The method of identifying similar groups of data in a data set is called clustering. Entities in each group are comparatively more similar to entities of that group than those of the other groups. In this article, I will be taking you through the types of clustering, different clustering algorithms and a comparison among some most commonly used clustering methods.

## <a name="Overview">Overview</a>

Clustering is the task of dividing the population or data points into a number of groups such that data points in the same groups are more similar to other data points in the same group than those in other groups. In simple words, the aim is to segregate groups with similar traits and assign them into clusters.

Let’s understand this with an example. Suppose, you are the head of a rental store and wish to understand preferences of your costumers to scale up your business. Is it possible for you to look at details of each costumer and devise a unique business strategy for each one of them? Definitely not. But, what you can do is to cluster all of your costumers into say 10 groups based on their purchasing habits and use a separate strategy for costumers in each of these 10 groups. And this is what we call clustering.

Now, that we understand what is clustering. Let’s take a look at the types of clustering.

## <a name="Types of Clustering">Types of Clustering</a>

Broadly speaking, clustering can be divided into two subgroups :

* **Hard Clustering**: In hard clustering, each data point either belongs to a cluster completely or not. For example, in the above example each customer is put into one group out of the 10 groups.
* **Soft Clustering**: In soft clustering, instead of putting each data point into a separate cluster, a probability or likelihood of that data point to be in those clusters is assigned. For example, from the above scenario each costumer is assigned a probability to be in either of 10 clusters of the retail store.

## <a name="Types of clustering algorithms">Types of clustering algorithms</a>

Since the task of clustering is subjective, the means that can be used for achieving this goal are plenty. Every methodology follows a different set of rules for defining the ‘similarity’ among data points. In fact, there are more than 100 clustering algorithms known. But few of the algorithms are used popularly, let’s look at them in detail:

* **Connectivity models**: As the name suggests, these models are based on the notion that the data points closer in data space exhibit more similarity to each other than the data points lying farther away. These models can follow two approaches. In the first approach, they start with classifying all data points into separate clusters & then aggregating them as the distance decreases. In the second approach, all data points are classified as a single cluster and then partitioned as the distance increases. Also, the choice of distance function is subjective. These models are very easy to interpret but lacks scalability for handling big datasets. Examples of these models are **hierarchical clustering algorithm** and its variants.
* **Centroid models**: These are iterative clustering algorithms in which the notion of similarity is derived by the closeness of a data point to the centroid of the clusters. **K-Means clustering algorithm** is a popular algorithm that falls into this category. In these models, the no. of clusters required at the end have to be mentioned beforehand, which makes it important to have prior knowledge of the dataset. These models run iteratively to find the local optima.
* **Distribution models**: These clustering models are based on the notion of how probable is it that all data points in the cluster belong to the same distribution (For example: Normal, Gaussian). These models often suffer from overfitting. A popular example of these models is **Expectation-maximization algorithm **which uses multivariate normal distributions.
* **Density Models**: These models search the data space for areas of varied density of data points in the data space. It isolates various different density regions and assign the data points within these regions in the same cluster. Popular examples of density models are **DBSCAN** and **Mean-Shift clustering**.

Now let's talk about five of the most popular clustering algorithms.

## <a name="K-Means Clustering">K-Means Clustering</a>

K-Means is probably the most well know clustering algorithm. It’s taught in a lot of introductory data science and machine learning classes. It’s easy to understand and implement in code! This algorithm works in these 4 steps :

1. Specify the desired number of clusters K and randomly initialize their respective center points. To figure out the number of classes to use, it’s good to take a quick look at the data and try to identify any distinct groupings.
2. Each data point is classified by computing the distance between that point and each group center, and then classifying the point to be in the group whose center is closest to it.
3. Based on these classified points, we recompute the group center by taking the mean of all the vectors in the group.
4. Repeat these steps for a set number of iterations or until the group centers don’t change much between iterations.

K-Means has the advantage that it’s pretty fast, as all we’re really doing is computing the distances between points and group centers; very few computations! It thus has a linear complexity O(n).

On the other hand, K-Means has a couple of disadvantages. Firstly, you have to select how many groups/classes there are. This isn’t always trivial and ideally with a clustering algorithm we’d want it to figure those out for us because the point of it is to gain some insight from the data. K-means also starts with a random choice of cluster centers and therefore it may yield different clustering results on different runs of the algorithm. Thus, the results may not be repeatable and lack consistency. Other cluster methods are more consistent.

K-Medians is another clustering algorithm related to K-Means, except instead of recomputing the group center points using the mean we use the median vector of the group. This method is less sensitive to outliers (because of using the Median) but is much slower for larger datasets as sorting is required on each iteration when computing the Median vector.

## <a name="Mean-Shift Clustering">Mean-Shift Clustering</a>

Mean shift clustering is a sliding-window-based algorithm that attempts to find dense areas of data points. It is a centroid-based algorithm meaning that the goal is to locate the center points of each group/class, which works by updating candidates for center points to be the mean of the points within the sliding-window. These candidate windows are then filtered in a post-processing stage to eliminate near-duplicates, forming the final set of center points and their corresponding groups.

### Meanshift by example ([<u>ref</u>](http://www.chioka.in/meanshift-algorithm-for-the-rest-of-us-python/))

Here is a diagram that shows what happens step-by-step in Meanshift.

![meanshift](/mldl/assets/images/2018-07-12/meanshift.jpg)

The blue datapoints are the initial datapoints and red are the positions of those datapoints at each iteration. Description for each step:

Iteration:

1. Initial state. The red and blue datapoints overlap completely in the first iteration before the Meanshift algorithm starts.
2. End of iteration 1. All the red datapoints move closer to clusters. Looks like there will be 4 clusters.
3. End of iteration 2. The clusters of upper right and lower left seems to have reached convergence just using two iterations. The center and lower right clusters looks like they are merging, since the two centroids are very close.
4. End of iteration 3. No change in the upper right and lower left centroids. The other two centroids’ have pulled each other together as the datapoints affect each clusters. This is a signature of Meanshift, the number of clusters are not pre-determined.
5. End of iteration 4. All the clusters should have converged.
6. End of iteration 5. All the clusters indeed have no movement. The algorithm stops here since no change is detected for all red datapoints.

![meanshift1](/mldl/assets/images/2018-07-12/meanshift1.jpg)

Meanshift found 3 clusters here, which is circled out above. The original data is actually generated from 4 clusters of data, but Meanshift thinks 3 can represent the set of data better, and it’s not too bad.

The MeanShift clustering algorithm can be described as the following 4 steps:

1. To explain mean-shift we will consider a set of points in two-dimensional space. We begin with a circular sliding window centered at a point C (randomly selected) and having radius r as the kernel. Mean shift is a hill climbing algorithm which involves shifting this kernel iteratively to a higher density region on each step until convergence.
2. At every iteration the sliding window is shifted towards regions of higher density by shifting the center point to the mean of the points within the window (hence the name). The density within the sliding window is proportional to the number of points inside it. Naturally, by shifting to the mean of the points in the window it will gradually move towards areas of higher point density.
3. We continue shifting the sliding window according to the mean until there is no direction at which a shift can accommodate more points inside the kernel.
4. This process of steps 1 to 3 is done with many sliding windows until all points lie within a window. When multiple sliding windows overlap the window containing the most points is preserved. The data points are then clustered according to the sliding window in which they reside.

In contrast to K-means clustering there is no need to select the number of clusters as mean-shift automatically discovers this. That’s a massive advantage. The fact that the cluster centers converge towards the points of maximum density is also quite desirable as it is quite intuitive to understand and fits well in a naturally data-driven sense. The drawback is that the selection of the window size/radius “r” can be non-trivial.

## <a name="DBSCAN">Density-Based Spatial Clustering of Applications with Noise (DBSCAN)</a>

DBSCAN is a density based clustering algorithm similar to mean-shift, but with a couple of notable advantages. Check out the fancy graphic below and let’s get started!

![dbscan](/mldl/assets/images/2018-07-12/dbscan.gif)

1. DBSCAN begins with an arbitrary starting data point that has not been visited. The neighborhood of this point is extracted using a distance epsilon $$\epsilon$$ (All points which are within the $$\epsilon$$ distance are neighborhood points).
2. If there are a sufficient number of points (according to minPoints) within this neighborhood then the clustering process starts and the current data point becomes the first point in the new cluster. Otherwise, the point will be labeled as noise (later this noisy point might become the part of the cluster). In both cases that point is marked as “visited”.
3. For this first point in the new cluster, the points within its $$\epsilon$$ distance neighborhood also become part of the same cluster. This procedure of making all points in the $$\epsilon$$ neighborhood belong to the same cluster is then repeated for all of the new points that have been just added to the cluster group.
4. This process of steps 2 and 3 is repeated until all points in the cluster are determined i.e all points within the $$\epsilon$$ neighborhood of the cluster have been visited and labelled.
5. Once we’re done with the current cluster, a new unvisited point is retrieved and processed, leading to the discovery of a further cluster or noise. This process repeats until all points are marked as visited. Since at the end of this all points have been visited, each point well have been marked as either belonging to a cluster or being noise.

DBSCAN poses some great advantages over other clustering algorithms. Firstly, it does not require a pe-set number of clusters at all. It also identifies outliers as noises unlike mean-shift which simply throws them into a cluster even if the data point is very different. Additionally, it is able to find arbitrarily sized and arbitrarily shaped clusters quite well.

The main drawback of DBSCAN is that it doesn’t perform as well as others when the clusters are of varying density. This is because the setting of the distance threshold $$\epsilon$$ and minPoints for identifying the neighborhood points will vary from cluster to cluster when the density varies. This drawback also occurs with very high-dimensional data since again the distance threshold $$\epsilon$$ becomes challenging to estimate.

## <a name="Agglomerative Hierarchical Clustering">Agglomerative Hierarchical Clustering</a>

Hierarchical clustering algorithms actually fall into 2 categories: top-down or bottom-up. Bottom-up algorithms treat each data point as a single cluster at the outset and then successively merge (or agglomerate) pairs of clusters until all clusters have been merged into a single cluster that contains all data points. Bottom-up hierarchical clustering is therefore called hierarchical agglomerative clustering or HAC. This hierarchy of clusters is represented as a tree (or dendrogram). The root of the tree is the unique cluster that gathers all the samples, the leaves being the clusters with only one sample.

1. We begin by treating each data point as a single cluster i.e if there are X data points in our dataset then we have X clusters. We then select a distance metric that measures the distance between two clusters. As an example we will use average linkage which defines the distance between two clusters to be the average distance between data points in the first cluster and data points in the second cluster.
2. On each iteration we combine two clusters into one. The two clusters to be combined are selected as those with the smallest average linkage. I.e according to our selected distance metric, these two clusters have the smallest distance between each other and therefore are the most similar and should be combined.
3. Step 2 is repeated until we reach the root of the tree i.e we only have one cluster which contains all data points. In this way we can select how many clusters we want in the end, simply by choosing when to stop combining the clusters i.e when we stop building the tree!

Hierarchical clustering does not require us to specify the number of clusters and we can even select which number of clusters looks best since we are building a tree. Additionally, the algorithm is not sensitive to the choice of distance metric; all of them tend to work equally well whereas with other clustering algorithms, the choice of distance metric is critical. A particularly good use case of hierarchical clustering methods is when the underlying data has a hierarchical structure and you want to recover the hierarchy; other clustering algorithms can’t do this. These advantages of hierarchical clustering come at the cost of lower efficiency, as it has a time complexity of $$O(n^3)$$, unlike the linear complexity of K-Means and GMM.

## <a name="EM-GMM">Expectation–Maximization (EM) Clustering using Gaussian Mixture Models (GMM)</a>

One of the major drawbacks of K-Means is its naive use of the mean value for the cluster center. We can see why this isn’t the best way of doing things by looking at the image below. On the left hand side it looks quite obvious to the human eye that there are two circular clusters with different radius’ centered at the same mean. K-Means can’t handle this because the mean values of the clusters are a very close together. K-Means also fails in cases where the clusters are not circular, again as a result of using the mean as cluster center.

![gmm](/mldl/assets/images/2018-07-12/gmm.jpg)

Gaussian Mixture Models (GMMs) give us more flexibility than K-Means. With GMMs we assume that the data points are Gaussian distributed; this is a less restrictive assumption than saying they are circular by using the mean. That way, we have two parameters to describe the shape of the clusters: the mean and the standard deviation! Taking an example in two dimensions, this means that the clusters can take any kind of elliptical shape (since we have standard deviation in both the x and y directions). Thus, each Gaussian distribution is assigned to a single cluster.

In order to find the parameters of the Gaussian for each cluster (e.g the mean and standard deviation) we will use an optimization algorithm called Expectation–Maximization (EM). Take a look at the graphic below as an illustration of the Gaussians being fitted to the clusters. Then we can proceed on to the process of Expectation–Maximization clustering using GMMs.

![gmm1](/mldl/assets/images/2018-07-12/gmm.gif)

1. We begin by selecting the number of clusters (like K-Means does) and randomly initializing the Gaussian distribution parameters for each cluster. One can try to provide a good guesstimate for the initial parameters by taking a quick look at the data too. Though note, as can be seen in the graphic above, this isn’t 100% necessary as the Gaussians start out as very poor but are quickly optimized.
2. Given these Gaussian distributions for each cluster, compute the probability that each data point belongs to a particular cluster. The closer a point is to the Gaussian’s center, the more likely it belongs to that cluster. This should make intuitive sense since with a Gaussian distribution we are assuming that most of the data lies closer to the center of the cluster.
3. Based on these probabilities, we compute a new set of parameters for the Gaussian distributions such that we maximize the probabilities of data points within the clusters. We compute these new parameters using a weighted sum of the data point positions, where the weights are the probabilities of the data point belonging in that particular cluster. To explain this in a visual manner we can take a look at the graphic above, in particular the yellow cluster as an example. The distribution starts off randomly on the first iteration, but we can see that most of the yellow points are to the right of that distribution. When we compute a sum weighted by the probabilities, even though there are some points near the center, most of them are on the right. Thus naturally the distribution’s mean is shifted closer to those set of points. We can also see that most of the points are “top-right to bottom-left”. Therefore the standard deviation changes to create an ellipse that is more fitted to these points, in order to maximize the sum weighted by the probabilities.
4. Steps 2 and 3 are repeated iteratively until convergence, where the distributions don’t change much from iteration to iteration.

There are really 2 key advantages to using GMMs. Firstly GMMs are a lot more flexible in terms of cluster covariance than K-Means; due to the standard deviation parameter, the clusters can take on any ellipse shape, rather than being restricted to circles. K-Means is actually a special case of GMM in which each cluster’s covariance along all dimensions approaches 0. Secondly, since GMMs use probabilities, they can have multiple clusters per data point. So if a data point is in the middle of two overlapping clusters, we can simply define its class by saying it belongs X-percent to class 1 and Y-percent to class 2. I.e GMMs support mixed membership.


References:

* [<u>An Introduction to Clustering and different methods of clustering</u>](https://www.analyticsvidhya.com/blog/2016/11/an-introduction-to-clustering-and-different-methods-of-clustering/)
* [<u></u>]()
