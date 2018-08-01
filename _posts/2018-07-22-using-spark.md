---
layout:         post
title:          Using Spark
subtitle:
card-image:     /assets/images/cards/cat5.gif
date:           2018-07-22 09:00:00
tags:           [deep&nbsp;learning]
categories:     [deep&nbsp;learning, spark]
post-card-type: image
mathjax:        true
---

* <a href="#Introduction">Introduction</a>
* <a href="#Setting up Python with Spark">Setting up Python with Spark</a>
* <a href="#Databricks setup">Databricks setup</a>
* <a href="#Spark basics">Spark basics</a>
* <a href="#Spark DataFrame Basics">Spark DataFrame Basics</a>
* <a href="#MLlib">MLlib</a>
* <a href="#Linear Regression">Linear Regression</a>
* <a href="#Logistic Regression">Logistic Regression</a>
* <a href="#Decision trees and random forests">Decision trees and random forests</a>
* <a href="#K-means clustering">K-means clustering</a>
* <a href="#NLP">NLP</a>
* <a href="#Spark Streaming">Spark Streaming</a>

## <a name="Introduction">Introduction</a>

Apache Spark is a fast and general-purpose cluster computing system. It provides high-level APIs in Java, Scala, Python and R, and an optimized engine that supports general execution graphs. It also supports a rich set of higher-level tools including **Spark SQL** for SQL and structured data processing, **MLlib** for machine learning, **GraphX** for graph processing, and **Spark Streaming**.

Note that, before Spark 2.0, the main programming interface of Spark was the Resilient Distributed Dataset (RDD). After Spark 2.0, RDDs are replaced by Dataset, which is strongly-typed like an RDD, but with richer optimizations under the hood. The RDD interface is still supported, and you can get a more complete reference at the [RDD programming guide](https://spark.apache.org/docs/latest/rdd-programming-guide.html). However, we highly recommend you to switch to use Dataset, which has better performance than RDD. See the [SQL programming guide](https://spark.apache.org/docs/latest/sql-programming-guide.html) to get more information about Dataset.

## <a name="Setting up Python with Spark">Setting up Python with Spark</a>

To install Spark (PySpark) on Mac, first go to [Apache Spark official site](https://spark.apache.org/downloads.html) to select and download a Spark release. Then, using `pip` to install the package `findspark`.

Now open Jupyter notebook and configure the PySpark application:

```python
import findspark
findspark.init("/Users/shuzhan/spark-2.2.1-bin-hadoop2.7/")

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("test").getOrCreate()
```

## <a name="Databricks setup">Databricks setup</a>

[<u>A Gentle Introduction to Apache Spark on Databricks</u>](https://community.cloud.databricks.com/?o=1889204078568059#notebook/2642820168385511/command/2642820168385512)

## <a name="Spark basics">Spark basics</a>

### Spark operations

Spark allows two distinct kinds of operations by the user: **transformations** and **actions**.

**Transformations** create a new dataset from an existing one. For example, `map` is a transformation that passes each dataset element through a function and returns a new dataset representing the results.

All transformations in Spark are **lazy**, in that they do not compute their results right away. Instead, they just remember the transformations applied to some base dataset (e.g. a file). The transformations are only computed when an action requires a result to be returned to the driver program. This design enables Spark to run more efficiently. For example, we can realize that a dataset created through `map` will be used in a `reduce` and return only the result of the `reduce` to the driver, rather than the larger mapped dataset.

**Actions** return a value to the driver program after running a computation on the dataset. For example, `reduce` is an action that aggregates all the elements of the dataset using some function and returns the final result to the driver program.

Actions are commands that are computed by Spark right at the time of their execution. They consist of running all of the previous transformations in order to get back an actual result. An action is composed of one or more jobs which consists of tasks that will be executed by the workers in parallel where possible.

Now we've seen that Spark consists of actions and transformations. Let's talk about why that's the case. The reason for this is that it gives a simple way to optimize the entire pipeline of computations as opposed to the individual pieces. This makes it exceptionally fast for certain types of computation because it can perform all relevant computations at once. Technically speaking, Spark `pipelines` this computation. This means that certain computations can all be performed at once (like a `map` and a `filter`) rather than having to do one operation for all pieces of data then the following operation. Apache Spark can also keep results in memory as opposed to other frameworks that immediately write to disk after each task.

Here are some simple examples of transformations and actions. Remember, these are not all the transformations and actions - this is just a short sample of them.

Transformations (lazy) | Actions
---| --- |
map   | reduce(func)
filter   | collect
union   | count
groupByKey   | first
sortByKey   | take(n)
distinct   | save

### Apache Spark architecture

Apache Spark allows you to treat many machines as one machine and this is done via a master-worker type architecture where there is a `driver` or master node in the cluster, accompanied by worker nodes, and a cluster manager.

A spark cluster has a single master node and any number of worker nodes. The driver (master) and the executors (workers) run their individual Java processes and users can run them on the same horizontal spark cluster or on separate machines i.e. in a vertical spark cluster or in mixed machine configuration.

![spark1](/assets/images/2018-07-22/spark1.jpg)

**Role of driver in Spark architecture**. Spark driver is the master node of a Spark application. It is the central point and the entry point of the Spark Shell (Scala, Python, and R). The driver program runs the main() function of the application and is the place where the Spark Context is created. Spark Driver contains various components – DAGScheduler, TaskScheduler, BackendScheduler and BlockManager responsible for the translation of spark user code into actual spark jobs executed on the cluster.

* The driver program that runs on the master node of the spark cluster schedules the job execution and negotiates with the cluster manager.
* It translates the RDD’s into the execution graph and splits the graph into multiple stages.
* Driver stores the metadata about all the Resilient Distributed Databases and their partitions.
* Cockpits of jobs and tasks execution -Driver program converts a user application into smaller execution units known as tasks. Tasks are then executed by the executors i.e. the worker processes which run individual tasks.
* Driver exposes the information about the running spark application through a Web UI at port 4040.

**Role of executor in Spark architecture**. Executor is a distributed agent responsible for the execution of tasks. Every spark application has its own executor process. Executors usually run for the entire lifetime of a Spark application and this phenomenon is known as “Static Allocation of Executors”. However, users can also opt for dynamic allocations of executors wherein they can add or remove spark executors dynamically to match with the overall workload.

* Executor performs all the data processing.
* Reads from and Writes data to external sources.
* Executor stores the computation results data in-memory, cache or on hard disk drives.
* Interacts with the storage systems.

**Role of cluster manager in Spark architecture**. Cluster manager is an external service responsible for acquiring resources on the spark cluster and allocating them to a spark job. There are 3 different types of cluster managers a Spark application can leverage for the allocation and deallocation of various physical resources such as memory for client spark jobs, CPU memory, etc. -- Hadoop YARN, Apache Mesos or the simple standalone spark cluster manager. Either of them can be launched on-premise or in the cloud for a spark application to run.

Choosing a cluster manager for any spark application depends on the goals of the application because all cluster managers provide different set of scheduling capabilities. To get started with apache spark, the standalone cluster manager is the easiest one to use when developing a new spark application.

### Understanding the run time architecture of a Spark application

**What happens when a Spark job is submitted?** When a client submits a spark user application code, the driver implicitly converts the code containing transformations and actions into a logical directed acyclic graph (DAG). At this stage, the driver program also performs certain optimizations like pipelining transformations and then it converts the logical DAG into physical execution plan with set of stages. After creating the physical execution plan, it creates small physical execution units referred to as tasks under each stage. Then tasks are bundled to be sent to the Spark Cluster.

The driver program then talks to the cluster manager and negotiates for resources. The cluster manager then launches executors on the worker nodes on behalf of the driver. At this point the driver sends tasks to the cluster manager based on data placement. Before executors begin execution, they register themselves with the driver program so that the driver has holistic view of all the executors. Now executors start executing the various tasks assigned by the driver program. At any point of time when the spark application is running, the driver program will monitor the set of executors that run. Driver program in the spark architecture also schedules future tasks based on data placement by tracking the location of cached data. When driver program's main() method exits or when it call the stop() method of the Spark Context, it will terminate all the executors and release the resources from the cluster manager.

The structure of a Spark program at higher level is - RDD's are created from the input data and new RDD's are derived from the existing RDD's using different transformations, after which an action is performed on the data. In any spark program, the DAG operations are created by default and whenever the driver runs the Spark DAG will be converted into a physical execution plan.

## <a name="Spark DataFrame Basics">Spark DataFrame Basics</a>

**Spark SQL** is a Spark module for structured data processing. Unlike the basic **Spark RDD API**, the interfaces provided by **Spark SQL** provide Spark with more information about the structure of both the data and the computation being performed. Internally, **Spark SQL** uses this extra information to perform extra optimizations. There are several ways to interact with Spark SQL including **SQL** and the **Dataset API**. When computing a result the same execution engine is used, independent of which API/language you are using to express the computation. This unification means that developers can easily switch back and forth between different APIs based on which provides the most natural way to express a given transformation.

**SQL**. One use of **Spark SQL** is to execute **SQL queries**. Spark SQL can also be used to read data from an existing Hive installation. When running SQL from within another programming language the results will be returned as a **Dataset/DataFrame**.

**Datasets and DataFrames**. A **Dataset** is a distributed collection of data. Dataset is a new interface added in Spark 1.6 that provides the benefits of RDDs (strong typing, ability to use powerful lambda functions) with the benefits of Spark SQL’s optimized execution engine. A Dataset can be constructed from JVM objects and then manipulated using functional transformations (map, flatMap, filter, etc.). The Dataset API is available in Scala and Java. **Python** does **not** have the support for the Dataset API. But due to Python’s dynamic nature, many of the benefits of the Dataset API are already available.

A **DataFrame** is a Dataset organized into named columns. It is conceptually equivalent to a table in a relational database or a dataframe in R/Python, but with richer optimizations under the hood. DataFrames can be constructed from a wide array of sources such as: structured data files, tables in Hive, external databases, or existing RDDs. The DataFrame API is **available** in Scala, Java, Python, and R. In Scala and Java, a DataFrame is represented by a Dataset of Rows.

Spark DataFrames are able to input and output data from a wide variety of sources. We can then use these DataFrames to apply various transformations on the data. At the end of the transformation calls, we can either show or collect the results to display or for some final processing.

Spark DataFrames are the workhouse and main way of working with Spark and Python post Spark 2.0. DataFrames act as powerful versions of tables, with rows and columns, easily handling large datasets. The shift to DataFrames provides many advantages:

* A much simpler syntax
* Ability to use SQL directly in the dataframe
* Operations are automatically distributed across RDDs

If you've used R or even the pandas library with Python you are probably already familiar with the concept of DataFrames. Spark DataFrame expand on a lot of these concepts, allowing you to transfer that knowledge easily by understanding the simple syntax of Spark DataFrames. Remember that the main advantage to using Spark DataFrames vs those other programs is that Spark can handle data across many RDDs, huge data sets that would never fit on a single computer. That comes at a slight cost of some "peculiar" syntax choices, but after this course you will feel very comfortable with all those topics!

Let's get started exploring the code:

### Starting Point: SparkSession

The entry point into all functionality in Spark is the `SparkSession` class. To create a basic `SparkSession`, just use `SparkSession.builder`:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("test").getOrCreate()
```

### Creating DataFrames

With a `SparkSession`, applications can create DataFrames from an existing RDD, from a Hive table, or from Spark data sources.

```python
df = spark.read.json("people.json")
# Displays the content of the DataFrame to stdout
df.show()
# +----+-------+
# | age|   name|
# +----+-------+
# |null|Michael|
# |  30|   Andy|
# |  19| Justin|
# +----+-------+
```

### Showing the data

In Python it’s possible to access a DataFrame’s columns either by attribute (`df.age`) or by indexing (`df['age']`). While the former is convenient for interactive data exploration, users are highly **encouraged to use the latter form**, which is future proof and won’t break with column names that are also attributes on the DataFrame class.

```python
# Print the schema in a tree format
df.printSchema()
# root
# |-- age: long (nullable = true)
# |-- name: string (nullable = true)
df.columns
# ['age', 'name']
df.describe()
# DataFrame[summary: string, age: string, name: string]
```

### Create data schema

Some data types make it easier to infer schema (like tabular formats such as csv which we will show later). However you often have to set the schema yourself if you aren't dealing with a `.read` method that doesn't have `inferSchema()` built-in. Spark has all the tools you need for this, it just requires a very specific structure:

We need to create the list of `StructField`:

* :param name: string, name of the field.
* :param dataType: :class:`DataType` of the field.
* :param nullable: boolean, whether the field can be null (None) or not.

```python
from pyspark.sql.types import StructField, StringType, IntegerType, StructType

data_schema = [StructField("age", IntegerType(), True), StructField("name", StringType(), True)]
final_struc = StructType(fields=data_schema)
df = spark.read.json("people.json", schema=final_struc)
df.printSchema()
# root
# |-- age: integer (nullable = true)
# |-- name: string (nullable = true)
```

### Grabbing the data

```python
df["age"]
# Column<age>
df.select("age")
# DataFrame[age: int]
df.select("age").show()
# +----+
#| age|
#+----+
#|null|
#|  30|
#|  19|
#+----+

## Returns list of Row objects
df.head(2)
# [Row(age=None, name=u'Michael'), Row(age=30, name=u'Andy')]

## Select multiple columns
df.select(["age", "name"])
# DataFrame[age: int, name: string]
```

### Creating new columns

```python
## Adding a new column with a simple copy
df.withColumn("newage", df["age"]).show()
#+----+-------+------+
#| age|   name|newage|
#+----+-------+------+
#|null|Michael|  null|
#|  30|   Andy|    30|
#|  19| Justin|    19|
#+----+-------+------+

## Simple rename
df.withColumnRenamed("age", "supernewage").show()
# DataFrame[supernewage: int, name: string]

## More complicated operations to create new columns
df.withColumn("doubleage", df["age"]*2).show()
#+----+-------+------+
#| age|   name|double|
#+----+-------+------+
#|null|Michael|  null|
#|  30|   Andy|    60|
#|  19| Justin|    38|
#+----+-------+------+
```

### Using SQL

To use SQL queries directly with the dataframe, you will need to register it to a temporary view:

```python
## Register the DataFrame as a SQL temporary view
df.createOrReplaceTempView("people")
sql_results = spark.sql("Select * From people")
sql_results.show()
#+----+-------+
#| age|   name|
#+----+-------+
#|null|Michael|
#|  30|   Andy|
#|  19| Justin|
#+----+-------+
spark.sql("Select * From people Where age=30").show()
#+---+----+
#|age|name|
#+---+----+
#| 30|Andy|
#+---+----+
```

### Filtering data

```python
df = spark.read.csv("appl_stock.csv", inferSchema=True, header=True)
df.printSchema()
#root
# |-- Date: timestamp (nullable = true)
# |-- Open: double (nullable = true)
# |-- High: double (nullable = true)
# |-- Low: double (nullable = true)
# |-- Close: double (nullable = true)
# |-- Volume: integer (nullable = true)
# |-- Adj Close: double (nullable = true)
```

A large part of working with DataFrames is the ability to quickly filter out data based on conditions. Spark DataFrames are built on top of the Spark SQL platform, which means that is you already know SQL, you can quickly and easily grab that data using SQL commands, or using the DataFrame methods.

```python
# Using SQL
df.filter("Close<500").show()

# Using SQL with .select()
df.filter("Close<500").select("Open").show()

# Using SQL with .select()
df.filter("Close<500").select(["open", "Close"]).show()
```

Using normal python comparison operators is another way to do this, they will look very similar to SQL operators, except you need to make sure you are calling the entire column within the dataframe, using the format: `df["column name"]`

```python
df.filter(df["Close"]<200).show()

# Will produce an error, make sure to read the error!
df.filter(df["Close"]<200 and df["Open"]>200).show()

# You should use "&" for "and". And make sure to add in the parenthesis separating the statements!
df.filter((df["Close"]<200) & (df["Open"]>200)).show()

# You should use "|" for "or". And make sure to add in the parenthesis separating the statements!
df.filter((df["Close"]<200) | (df["Open"]>200)).show()

# You should use "~" for "not". And make sure to add in the parenthesis separating the statements!
df.filter((df["Close"]<200) & ~(df["Open"]<200)).show()

df.filter(df["Low"] == 197.16).show()

# Collecting results as Python objects
result = df.filter(df["Low"] == 197.16).collect()

result
#[Row(Date=datetime.datetime(2010, 1, 22, 0, 0), Open=206.78000600000001, High=207.499996, Low=197.16, Close=197.75, Volume=220441900, Adj Close=25.620401)]

row = result[0]
# Rows can be called to turn into dictionaries
row.asDict()
#{'Adj Close': 25.620401,
# 'Close': 197.75,
# 'Date': datetime.datetime(2010, 1, 22, 0, 0),
# 'High': 207.499996,
# 'Low': 197.16,
# 'Open': 206.78000600000001,
# 'Volume': 220441900}

for item in result[0]:
    print item
#2010-01-22 00:00:00
#206.780006
#207.499996
#197.16
#197.75
#220441900
#25.620401
```

### Dates and Timestamps

```python
df = spark.read.csv("appl_stock.csv", header=True, inferSchema=True)
df.printSchema()
#root
# |-- Date: timestamp (nullable = true)
# |-- Open: double (nullable = true)
# |-- High: double (nullable = true)
# |-- Low: double (nullable = true)
# |-- Close: double (nullable = true)
# |-- Volume: integer (nullable = true)
# |-- Adj Close: double (nullable = true)

from pyspark.sql.functions import format_number, dayofmonth, hour, dayofyear, month, year, weekofyear, date_format

df.select(dayofmonth(df["Date"])).show()

df.select(hour(df["Date"])).show()

df.select(dayofyear(df["Date"])).show()

df.select(month(df["Date"])).show()

df.select(year(df["Date"])).show()

df.withColumn("Year", year(df["Date"])).show()

newdf = df.withColumn("Year", year(df["Date"]))
newdf.groupBy("Year").mean()[["avg(Year)", "avg(Close)"]].show()

# Still not quite presentable! Let's use the .alias method as well as round() to clean this up!

result = newdf.groupBy("Year").mean()[['avg(Year)','avg(Close)']]
result = result.withColumnRenamed("avg(Year)", "Year")
result = result.select("Year", format_number("avg(Close)",2).alias("Mean Close")).show()
```

### GroupBy and Aggregate Functions

Let's learn how to use **GroupBy** and **Aggregate** methods on a DataFrame. GroupBy allows you to group rows together based off some column value, for example, you could group together sales data by the day the sale occured, or group repeast customer data based off the name of the customer. Once you've performed the GroupBy operation you can use an aggregate function off that data. An aggregate function aggregates multiple rows of data into a single output, such as taking the sum of inputs, or counting the number of inputs.

```python
df = spark.read.csv("sales_info.csv", inferSchema=True, header=True)
df.printSchema()
#root
# |-- Company: string (nullable = true)
# |-- Person: string (nullable = true)
# |-- Sales: double (nullable = true)

df.groupBy("Company").mean().show()
#+-------+-----------------+
#|Company|       avg(Sales)|
#+-------+-----------------+
#|   APPL|            370.0|
#|   GOOG|            220.0|
#|     FB|            610.0|
#|   MSFT|322.3333333333333|
#+-------+-----------------+

df.groupBy("Company").count().show()

df.groupBy("Company").max().show()

df.groupBy("Company").sum().show()
```

Not all methods need a `groupBy` call, instead you can just call the generalized `.agg()` method, that will call the aggregate across all rows in the dataframe column specified. It can take in arguments as a single column, or create multiple aggregate calls all at once using dictionary notation.

```python
# Max sales across everything
df.agg({"Sales": "max"}).show()

# Could have done this on the groupBy object as well:
grouped = df.groupBy("Company")
grouped.agg({"Sales":'max'}).show()
```

There are a variety of **functions** you can import from `pyspark.sql.functions`.

```python
from pyspark.sql.functions import countDistinct, avg, stddev

df.select(countDistinct("Sales")).show()
#+---------------------+
#|count(DISTINCT Sales)|
#+---------------------+
#|                   11|
#+---------------------+

# Often you will want to change the name, use the .alias() method for this:
df.select(countDistinct("Sales").alias("Distinct Sales")).show()
#+--------------+
#|Distinct Sales|
#+--------------+
#|            11|
#+--------------+

df.select(avg("Sales")).show()

df.select(stddev("Sales")).show()
#+------------------+
#|stddev_samp(Sales)|
#+------------------+
#|250.08742410799007|
#+------------------+

# That is a lot of precision for digits! Let's use the format_number to fix that!
from pyspark.sql.functions import format_number

sales_std = df.select(stddev("Sales").alias("std"))
sales_std.select(format_number("std", 2)).show()
#+---------------------+
#|format_number(std, 2)|
#+---------------------+
#|               250.09|
#+---------------------+
```

You can easily sort with the `orderBy` method:

```python
# orderBy ascending
df.orderBy("Sales").show()

# Descending call off the column itself.
df.orderBy(df["Sales"].desc()).show()
```

### Missing Data

Often data sources are incomplete, which means you will have missing data, you have 3 basic options for filling in missing data (you will personally have to make the decision for what is the right approach:

* Just keep the missing data points.
* Drop the missing data points (including the entire row)
* Fill them in with some other value.

Let's cover examples of each of these methods!

**Drop the missing data**. You can use the `.na` functions for missing data. The `df.na.drop` command has the following parameters:

* param how: 'any' or 'all'. If 'any', drop a row if it contains any nulls. If 'all', drop a row only if all its values are null.
*  param thresh: int, default None. If specified, drop rows that have less than `thresh` non-null values. This overwrites the `how` parameter.
* param subset: optional list of column names to consider.

```python
df = spark.read.csv("ContainsNull.csv",header=True,inferSchema=True)
df.show()
#+----+-----+-----+
#|  Id| Name|Sales|
#+----+-----+-----+
#|emp1| John| null|
#|emp2| null| null|
#|emp3| null|345.0|
#|emp4|Cindy|456.0|
#+----+-----+-----+

# Drop any row that contains missing data
df.na.drop().show()
#+----+-----+-----+
#|  Id| Name|Sales|
#+----+-----+-----+
#|emp4|Cindy|456.0|
#+----+-----+-----+

# Has to have at least 2 NON-null values
df.na.drop(thresh=2).show()
#+----+-----+-----+
#|  Id| Name|Sales|
#+----+-----+-----+
#|emp1| John| null|
#|emp3| null|345.0|
#|emp4|Cindy|456.0|
#+----+-----+-----+

df.na.drop(subset=["Sales"]).show()
#+----+-----+-----+
#|  Id| Name|Sales|
#+----+-----+-----+
#|emp3| null|345.0|
#|emp4|Cindy|456.0|
#+----+-----+-----+

df.na.drop(how="any").show()
#+----+-----+-----+
#|  Id| Name|Sales|
#+----+-----+-----+
#|emp4|Cindy|456.0|
#+----+-----+-----+

df.na.drop(how="all").show()
#+----+-----+-----+
#|  Id| Name|Sales|
#+----+-----+-----+
#|emp1| John| null|
#|emp2| null| null|
#|emp3| null|345.0|
#|emp4|Cindy|456.0|
#+----+-----+-----+
```

**Fill the missing values**. We can also fill the missing values with new values. If you have multiple nulls across multiple data types, Spark is actually smart enough to match up the data types. For example:

```python
df.na.fill("NEW VALUE").show()
#+----+---------+-----+
#|  Id|     Name|Sales|
#+----+---------+-----+
#|emp1|     John| null|
#|emp2|NEW VALUE| null|
#|emp3|NEW VALUE|345.0|
#|emp4|    Cindy|456.0|
#+----+---------+-----+

df.na.fill(0).show()
#+----+-----+-----+
#|  Id| Name|Sales|
#+----+-----+-----+
#|emp1| John|  0.0|
#|emp2| null|  0.0|
#|emp3| null|345.0|
#|emp4|Cindy|456.0|
#+----+-----+-----+

# Usually you should specify what columns you want to fill with the subset parameter
df.na.fill("No Name", subset=["Name"]).show()
#+----+-------+-----+
#|  Id|   Name|Sales|
#+----+-------+-----+
#|emp1|   John| null|
#|emp2|No Name| null|
#|emp3|No Name|345.0|
#|emp4|  Cindy|456.0|
#+----+-------+-----+

# A very common practice is to fill values with the mean value for the column, for example:
from pyspark.sql.functions import mean
df.na.fill(df.select(mean(df["Sales"])).collect()[0][0], ["Sales"]).show()
#+----+-----+-----+
#|  Id| Name|Sales|
#+----+-----+-----+
#|emp1| John|400.5|
#|emp2| null|400.5|
#|emp3| null|345.0|
#|emp4|Cindy|456.0|
#+----+-----+-----+
```

## <a name="MLlib">MLlib</a>

### Introduction

**MLlib** is Spark’s machine learning (ML) library. Its goal is to make practical machine learning scalable and easy. At a high level, it provides tools such as:

* ML Algorithms: common learning algorithms such as classification, regression, clustering, and collaborative filtering
* Featurization: feature extraction, transformation, dimensionality reduction, and selection
* Pipelines: tools for constructing, evaluating, and tuning ML Pipelines
* Persistence: saving and load algorithms, models, and Pipelines
* Utilities: linear algebra, statistics, data handling, etc.

**The MLlib RDD-based API is now in maintenance mode**. As of Spark 2.0, the **RDD-based** APIs in the `spark.mllib` package have entered maintenance mode. The primary Machine Learning API for Spark is now the **DataFrame-based** API in the `spark.ml` package.

*What are the implications?*

* MLlib will still support the RDD-based API in `spark.mllib` with bug fixes.
* MLlib will not add new features to the RDD-based API.
* In the Spark 2.x releases, MLlib will add features to the DataFrames-based API to reach feature parity with the RDD-based API.
* After reaching feature parity (roughly estimated for Spark 2.3), the RDD-based API will be deprecated.
* The RDD-based API is expected to be removed in Spark 3.0.

*Why is MLlib switching to the DataFrame-based API?*

* DataFrames provide a more user-friendly API than RDDs. The many benefits of DataFrames include Spark Datasources, SQL/DataFrame queries, Tungsten and Catalyst optimizations, and uniform APIs across languages.
* The DataFrame-based API for MLlib provides a uniform API across ML algorithms and across multiple languages.
* DataFrames facilitate practical ML Pipelines, particularly feature transformations.

*What is “Spark ML”?*

* “Spark ML” is not an official name but occasionally used to refer to the MLlib DataFrame-based API. This is majorly due to the `org.apache.spark.ml` Scala package name used by the DataFrame-based API, and the “Spark ML Pipelines” term we used initially to emphasize the pipeline concept.

*Is MLlib deprecated?*

* No. MLlib includes both the RDD-based API and the DataFrame-based API. The RDD-based API is now in maintenance mode. But neither API is deprecated, nor MLlib as a whole.

One of the main “quirks” of using MLlib is that you need to format your data so that eventually it just has one or two columns: **Features, Labels (Supervised)** and **Features (Unsupervised)**.

This requires a little more data processing work than some other machine learning libraries, but the big upside is that this exact same syntax works with distributed data, which is no small feat for what is going on “under the hood”!

## <a name="Linear Regression">Linear Regression</a>

### Linear Regression Code Along

Basically what we do here is examine a dataset with Ecommerce Customer Data for a company's website and mobile app. Then we want to see if we can build a regression model that will predict the customer's yearly spend on the company's product.

```python
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression

spark = SparkSession.builder.appName("lr").getOrCreate()
data = spark.read.csv("Ecommerce_Customers.csv",inferSchema=True,header=True)
#root
# |-- Email: string (nullable = true)
# |-- Address: string (nullable = true)
# |-- Avatar: string (nullable = true)
# |-- Avg Session Length: double (nullable = true)
# |-- Time on App: double (nullable = true)
# |-- Time on Website: double (nullable = true)
# |-- Length of Membership: double (nullable = true)
# |-- Yearly Amount Spent: double (nullable = true)
```

A few things we need to do before Spark can accept the data! It needs to be in the form of two columns: (features", "label").

```python
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler([
inputCols = ["Avg Session Length", "Time on App", "Time on Website",'Length of Membership'],
outputCol = "features"
])

output = assembler.transform(data)

final_data = output.select("features", "Yearly Amount Spent")
train_data, test_data = final_data.randomSplit([0.7, 0.3])

# Create a Linear Regression Model object
lr = LinearRegression(labelCol="Yearly Amount Spent")

# Fit the model to the data and call this model lrModel
lrModel = lr.fit(train_data,)

# Print the coefficients and intercept for linear regression
print("Coefficients: {} Intercept: {}".format(lrModel.coefficients, lrModel.intercept))

# Evaluate the model
test_results = lrModel.evaluate(test_data)
test_results.residuals.show()
print("RMSE: {}".format(test_results.rootMeanSquaredError))
print("MSE: {}".format(test_results.meanSquaredError))

# Make new predictions
unlabeled_data = test_data.select("features")
predictions = lrModel.transform(unlabeled_data)
predictions.show()
```

### Data Transformations

```python
df = spark.read.csv('fake_customers.csv',inferSchema=True,header=True)
df.printSchema()
#root
# |-- Name: string (nullable = true)
# |-- Phone: long (nullable = true)
# |-- Group: string (nullable = true)
```

#### StringIndexer

`StringIndexer` encodes a string column of labels to a column of label indices. The indices are in [0, numLabels), ordered by label frequencies, so the most frequent label gets index 0. The unseen labels will be put at index numLabels if user chooses to keep them. If the input column is numeric, we cast it to string and index the string values. When downstream pipeline components such as `Estimator` or `Transformer` make use of this string-indexed label, you must set the input column of the component to this string-indexed column name. In many cases, you can set the input column with `setInputCol`.

```python
from pyspark.ml.feature import StringIndexer

df = spark.createDataFrame([(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")], ["id", "category"])
indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
indexed = indexer.fit(df).transform(df)
indexed.show()
#+---+--------+-------------+
#| id|category|categoryIndex|
#+---+--------+-------------+
#|  0|       a|          0.0|
#|  1|       b|          2.0|
#|  2|       c|          1.0|
#|  3|       a|          0.0|
#|  4|       a|          0.0|
#|  5|       c|          1.0|
#+---+--------+-------------+
```

#### VectorAssembler

`VectorAssembler` is a transformer that combines a given list of columns into a single vector column. It is useful for combining raw features and features generated by different feature transformers into a single feature vector, in order to train ML models like logistic regression and decision trees. `VectorAssembler` accepts the following input column types: all numeric types, boolean type, and vector type. In each row, the values of the input columns will be concatenated into a vector in the specified order.

```python
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

dataset = spark.createDataFrame([(0, 18, 1.0, Vectors.dense([0.0, 10.0, 0.5]), 1.0)], ["id", "hour", "mobile", "userFeatures", "clicked"])
dataset.show()
#+---+----+------+--------------+-------+
#| id|hour|mobile|  userFeatures|clicked|
#+---+----+------+--------------+-------+
#|  0|  18|   1.0|[0.0,10.0,0.5]|    1.0|
#+---+----+------+--------------+-------+

assembler = VectorAssembler(inputCols=["hour", "mobile", "userFeatures"], outputCol="features")
output = assembler.transform(dataset)
print("Assembled columns 'hour', 'mobile', 'userFeatures' to vector column 'features'")
output.select("features", "clicked").show()
#Assembled columns 'hour', 'mobile', 'userFeatures' to vector column 'features'
#+--------------------+-------+
#|            features|clicked|
#+--------------------+-------+
#|[18.0,1.0,0.0,10....|    1.0|
#+--------------------+-------+
```

#### MinMaxScaler
`MinMaxScaler` transforms a dataset of `Vector` rows, rescaling each feature to a specific range (often [0, 1]). It takes parameters:
* min: 0.0 by default. Lower bound after transformation, shared by all features.
* max: 1.0 by default. Upper bound after transformation, shared by all features.

`MinMaxScaler` computes summary statistics on a data set and produces a `MinMaxScalerModel`. The model can then transform each feature individually such that it is in the given range.

```python
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import MinMaxScaler

dataFrame = spark.createDataFrame([
    (0, Vectors.dense([1.0, 0.1, -1.0]),),
    (1, Vectors.dense([2.0, 1.1, 1.0]),),
    (2, Vectors.dense([3.0, 10.1, 3.0]),)
], ["id", "features"])

scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
scaled = scaler.fit(dataFrame).transform(dataFrame)
scaledData.select("features", "scaledFeatures").show()
#+--------------+--------------+
#|      features|scaledFeatures|
#+--------------+--------------+
#|[1.0,0.1,-1.0]| [0.0,0.0,0.0]|
#| [2.0,1.1,1.0]| [0.5,0.1,0.5]|
#|[3.0,10.1,3.0]| [1.0,1.0,1.0]|
#+--------------+--------------+
```

#### StandardScaler

`StandardScaler` transforms a dataset of `Vector` rows, normalizing each feature to have unit standard deviation and/or zero mean. It takes parameters:

* withStd: True by default. Scales the data to unit standard deviation.
* withMean: False by default. Centers the data with mean before scaling. It will build a dense output, so take care when applying to sparse input.

```python
from pyspark.ml.feature import StandardScaler

dataFrame = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=False)

# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(dataFrame)

# Normalize each feature to have unit standard deviation.
scaledData = scalerModel.transform(dataFrame)
scaledData.show()
```

## <a name="Logistic Regression">Logistic Regression</a>

```python
from pyspark.ml.classification import LogisticRegression

training = spark.read.format("libsvm").load("sample_libsvm_data.txt")
training.printSchema()
#root
# |-- label: double (nullable = true)
# |-- features: vector (nullable = true)
lr = LogisticRegression()
lrModel = lr.fit(training)
trainingSummary = lrModel.summary
trainingSummary.predictions.show()

# Usually would do this on a separate test set!
predictionAndLabels = lrModel.evaluate(training)
predictionAndLabels = predictionAndLabels.predictions.select("label", "prediction")
predictionAndLabels.show()

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

bi_evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="label")
mu_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")
auc = bi_evaluator.evaluate(predictionAndLabels)
acc = mu_evaluator.evaluate(predictionAndLabels)
```

```python
data = spark.read.csv('titanic.csv',inferSchema=True,header=True)
data.printSchema()
#root
# |-- PassengerId: integer (nullable = true)
# |-- Survived: integer (nullable = true)
# |-- Pclass: integer (nullable = true)
# |-- Name: string (nullable = true)
# |-- Sex: string (nullable = true)
# |-- Age: double (nullable = true)
# |-- SibSp: integer (nullable = true)
# |-- Parch: integer (nullable = true)
# |-- Ticket: string (nullable = true)
# |-- Fare: double (nullable = true)
# |-- Cabin: string (nullable = true)
# |-- Embarked: string (nullable = true)
my_cols = data.select(['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
my_final_data = ml_cols.na.drop()
```

### Working with categorical columns

```python
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoderEstimator

gender_indexer = StringIndexer(inputCol="Sex", outputCol="SexIndex")
gender_encoder = OneHotEncoderEstimator(inputCols=["SexIndex"], outputCols=["SexVec"])
embark_indexer = StringIndexer(inputCol="Embarked", outputCol="EmbarkIndex")
embark_encoder = OneHotEncoderEstimator(inputCols=["EmbarkIndex"], outputCols=["EmbarkVec"])
assembler = VectorAssembler(inputCols=['Pclass','SexVec','Age','SibSp','Parch','Fare','EmbarkVec'],outputCol='features')
```

### Pipelines

```python
from pyspark.ml import Pipeline

log_reg_titanic = LogisticRegression(featuresCol="features", labelCol="Survived")
pipeline = Pipeline(stages=[gender_indexer, embark_indexer, gender_encoder, embark_encoder, assembler, log_reg_titanic])
train_titanic_data, test_titanic_data = my_final_data.randomSplit([0.7, 0.3])
fit_model = pipeline.fit(train_titanic_data)
results = fit_model.transform(test_titanic_data)
results.select('Survived','prediction').show()

from pyspark.ml.evaluation import BinaryClassificationEvaluator

my_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='Survived')
auc = my_eval.evaluate(results)
auc
# 0.78140625000
```

## <a name="Decision trees and random forests">Decision trees and random forests</a>

### Random Forest Example

```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

data = spark.read.format("libsvm").load("sample_libsvm_data.txt")
trainingData, testData = data.randomSplit([0.7, 0.3])
trainingData.printSchema()
#root
# |-- label: double (nullable = true)
# |-- features: vector (nullable = true)
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=20)
model = rf.fit(trainingData)
predictions = model.transform(testData)
predictions.printSchema()
#root
# |-- label: double (nullable = true)
# |-- features: vector (nullable = true)
# |-- rawPrediction: vector (nullable = true)
# |-- probability: vector (nullable = true)
# |-- prediction: double (nullable = true)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy)) # Test Error = 0.04
model.featureImportances
```

### Gradient Boosted Trees

```python
from pyspark.ml.classification import GBTClassifier

data = spark.read.format("libsvm").load("sample_libsvm_data.txt")
trainingData, testData = data.randomSplit([0.7, 0.3])
gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10)
model = gbt.fit(trainingData)
predictions = model.transform(testData)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))  # Test Error = 0
```

We will then code along with some data and test out 3 different tree methods. We will be using a college dataset to try to classify colleges as Private or Public based off some features.

```python
data = spark.read.csv('College.csv',inferSchema=True,header=True)
data.head()
#Row(School=u'Abilene Christian University', Private=u'Yes', Apps=1660, Accept=1232, Enroll=721, Top10perc=23, Top25perc=52, F_Undergrad=2885, P_Undergrad=537, Outstate=7440, Room_Board=3300, Books=450, Personal=2200, PhD=70, Terminal=78, S_F_Ratio=18.1, perc_alumni=12, Expend=7041, Grad_Rate=60)

## Spark Formatting of Data
from pyspark.ml.feature import VectorAssembler
data.columns
assembler = VectorAssembler(inputCols=['Apps', 'Accept', 'Enroll', 'Top10perc', 'Top25perc', 'F_Undergrad', 'P_Undergrad', 'Outstate', 'Room_Board', 'Books', 'Personal', 'PhD', 'Terminal', 'S_F_Ratio', 'perc_alumni', 'Expend', 'Grad_Rate'],  outputCol="features")
output = assembler.transform(data)
from pyspark.ml.features import StringIndexer
indexer = StringIndexer(inputCol="Private", outputCol="PrivateIndex")
output_fixed = indexer.fit(output).transform(output)
final_data = output_fixed.select("features", "PrivateIndex")
train_data, test_data = final_data.randomSplit([0.7, 0.3])

## The Classifiers
from pyspark.ml.classification import DecisionTreeClassifier,GBTClassifier,RandomForestClassifier
from pyspark.ml import Pipeline

dtc = DecisionTreeClassifier(labelCol='PrivateIndex',featuresCol='features')
rfc = RandomForestClassifier(labelCol='PrivateIndex',featuresCol='features')
gbt = GBTClassifier(labelCol='PrivateIndex',featuresCol='features')
dtc_model = dtc.fit(train_data)
rfc_model = rfc.fit(train_data)
gbt_model = gbt.fit(train_data)
dtc_predictions = dtc_model.transform(test_data)
rfc_predictions = rfc_model.transform(test_data)
gbt_predictions = gbt_model.transform(test_data)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
acc_evaluator = MulticlassClassificationEvaluator(labelCol="PrivateIndex", predictionCol="prediction", metricName="accuracy")
dtc_acc = acc_evaluator.evaluate(dtc_predictions)
rfc_acc = acc_evaluator.evaluate(rfc_predictions)
gbt_acc = acc_evaluator.evaluate(gbt_predictions)
```

## <a name="K-means clustering">K-means clustering</a>

```python
from pyspark.ml.clustering import KMeans

dataset = spark.read.format("libsvm").load("sample_kmeans_data.txt")
dataset.head()
# Row(label=0.0, features=SparseVector(3, {}))
# Trains a k-means model.
kmeans = KMeans(k=2, seed=1)
model = kmeans.fit(dataset)
# Evaluate clustering by computing Within Set Sum of Squared Errors.
wssse = model.computeCost(dataset)
print("Within Set Sum of Squared Errors = " + str(wssse))
# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)
```

```python
dataset = spark.read.csv("seeds_dataset.csv",header=True,inferSchema=True)
dataset.head()
#Row(area=15.26, perimeter=14.84, compactness=0.871, length_of_kernel=5.763, width_of_kernel=3.312, asymmetry_coefficient=2.221, length_of_groove=5.22)

# Format the Data
from pyspark.ml.feature import VectorAssembler

vec_assembler = VectorAssembler(inputCols = dataset.columns, outputCol='features')
final_data = vec_assembler.transform(dataset)

# Scale the Data
from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
final_data = scaler.fit(final_data).transform(final_data)

# Train the Model and Evaluate
kmeans = KMeans(featuresCol="scaledFeatures", k=3)
model = kmeans.fit(final_data)
wssse = model.computeCost(final_data)
print("Within Set Sum of Squared Errors = " + str(wssse))
# Within Set Sum of Squared Errors = 429.075596715
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)
#Cluster Centers:
#[ 6.31670546 12.37109759 37.39491396 13.91155062  9.748067    2.39849968
# 12.2661748 ]
#[ 4.87257659 10.88120146 37.27692543 12.3410157   8.55443412  1.81649011
# 10.32998598]
#[ 4.06105916 10.13979506 35.80536984 11.82133095  7.50395937  3.27184732
# 10.42126018]
model.transform(final_data).select("prediction").show()
```

## <a name="NLP">NLP</a>

### Tokenizer, RegexTokenizer

**Tokenization** is the process of taking text (such as a sentence) and breaking it into individual terms (usually words). A simple `Tokenizer` class provides this functionality. The example below shows how to split sentences into sequences of words.

`RegexTokenizer` allows more advanced tokenization based on regular expression (regex) matching. By default, the parameter “pattern” (regex, default: "\\s+") is used as delimiters to split the input text. Alternatively, users can set parameter “gaps” to false indicating the regex “pattern” denotes “tokens” rather than splitting gaps, and find all matching occurrences as the tokenization result.

```python
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType

sentenceDataFrame = spark.createDataFrame([(0, "Hi I heard about Spark"),(1, "I wish Java could use case classes"),(2, "Logistic,regression,models,are,neat")], ["id", "sentence"])
tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
regexTokenizer = RegexTokenizer(inputCol="sentence", outputCol="words", pattern="\\W")

countTokens = udf(lambda words: len(words), IntegerType())

tokenized = tokenizer.transform(sentenceDataFrame)
tokenized.select("sentence", "words").withColumn("tokens", countTokens(col("words"))).show(truncate=False)
regexTokenized = regexTokenizer.transform(sentenceDataFrame)
regexTokenized.select("sentence", "words").withColumn("tokens", countTokens(col("words"))).show(truncate=False)
```

### Stop Words Removal

Stop words are words which should be excluded from the input, typically because the words appear frequently and don’t carry as much meaning.

`StopWordsRemover` takes as input a sequence of strings (e.g. the output of a `Tokenizer`) and drops all the stop words from the input sequences. The list of stopwords is specified by the `stopWords` parameter. Default stop words for some languages are accessible by calling `StopWordsRemover.loadDefaultStopWords(language)`, for which available options are “danish”, “dutch”, “english”, “finnish”, “french”, “german”, “hungarian”, “italian”, “norwegian”, “portuguese”, “russian”, “spanish”, “swedish” and “turkish”. A boolean parameter `caseSensitive` indicates if the matches should be case sensitive (false by default).

```python
from pyspark.ml.feature import StopWordsRemover

sentenceData = spark.createDataFrame([(0, ["I", "saw", "the", "red", "balloon"]), (1, ["Mary", "had", "a", "little", "lamb"])], ["id", "raw"])
remover = StopWordsRemover(inputCol="raw", outputCol="filtered")
remover.transform(sentenceData).show(truncate=False)
```

### n-grams

An n-gram is a sequence of n tokens (typically words) for some integer n. The `NGram` class can be used to transform input features into n-grams.

`NGram` takes as input a sequence of strings (e.g. the output of a `Tokenizer`.  The parameter n is used to determine the number of terms in each n-gram. The output will consist of a sequence of n-grams where each n-gram is represented by a space-delimited string of $n$ consecutive words.  If the input sequence contains fewer than n strings, no output is produced.

```python
from pyspark.ml.feature import NGram

wordDataFrame = spark.createDataFrame([(0, ["Hi", "I", "heard", "about", "Spark"]), (1, ["I", "wish", "Java", "could", "use", "case", "classes"]), (2, ["Logistic", "regression", "models", "are", "neat"])], ["id", "words"])
ngram = NGram(n=2, inputCol="words", outputCol="ngrams")

ngramDataFrame = ngram.transform(wordDataFrame)
ngramDataFrame.select("ngrams").show(truncate=False)
```

### TF-IDF

Term frequency-inverse document frequency (TF-IDF) is a feature vectorization method widely used in text mining to reflect the importance of a term to a document in the corpus. Denote a term by $$t$$, a document by $$d$$ , and the corpus by $$D$$. Term frequency $$TF(t, d)$$ is the number of times that term $$t$$ appears in document $$d$$, while document frequency $$DF(t, D)$$ is the number of documents that contains term $$t$$. If we only use term frequency to measure the importance, it is very easy to over-emphasize terms that appear very often but carry little information about the document, e.g. “a”, “the”, and “of”. If a term appears very often across the corpus, it means it doesn’t carry special information about a particular document. Inverse document frequency is a numerical measure of how much information a term provides.

In MLlib, we separate **TF** and **IDF** to make them flexible. Both `HashingTF` and `CountVectorizer` can be used to generate the term frequency vectors. `HashingTF` is a Transformer which takes sets of terms and converts those sets into fixed-length feature vectors. In text processing, a “set of terms” might be a bag of words. `HashingTF` utilizes the hashing trick. A raw feature is mapped into an index (term) by applying a hash function. Then term frequencies are calculated based on the mapped indices. This approach avoids the need to compute a global term-to-index map, which can be expensive for a large corpus, but it suffers from potential hash collisions, where different raw features may become the same term after hashing. To reduce the chance of collision, we can increase the target feature dimension, i.e. the number of buckets of the hash table. Since a simple modulo on the hashed value is used to determine the vector index, it is advisable to use a power of two as the feature dimension, otherwise the features will not be mapped evenly to the vector indices. The default feature dimension is $$2^18 = 262,144$$.

`CountVectorizer` converts text documents to vectors of term counts. Refer to CountVectorizer for more details.

`IDF` is an Estimator which is fit on a dataset and produces an `IDFModel`. The IDFModel takes feature vectors (generally created from `HashingTF` or `CountVectorizer`) and scales each feature. Intuitively, it down-weights features which appear frequently in a corpus.

```python
from pyspark.ml.feature import HashingTF, IDF, Tokenizer

sentenceData = spark.createDataFrame([
    (0.0, "Hi I heard about Spark"),
    (0.0, "I wish Java could use case classes"),
    (1.0, "Logistic regression models are neat")
], ["label", "sentence"])

tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
wordsData = tokenizer.transform(sentenceData)
wordsData.show()

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)

idf = IDF(inputCol="rawFeatures", outputCol="features")
rescaledData = idf.fit(featurizedData).transform(featurizedData)
rescaledData.select("label", "features").show()
```

### CountVectorizer

`CountVectorizer` and `CountVectorizerModel` aim to help convert a collection of text documents to vectors of token counts. When an a-priori dictionary is not available, `CountVectorizer` can be used as an Estimator to extract the vocabulary, and generates a `CountVectorizerModel`. The model produces sparse representations for the documents over the vocabulary, which can then be passed to other algorithms like LDA.

During the fitting process, `CountVectorizer` will select the top `vocabSize` words ordered by term frequency across the corpus. An optional parameter `minDF` also affects the fitting process by specifying the minimum number (or fraction if < 1.0) of documents a term must appear in to be included in the vocabulary.

```python
from pyspark.ml.feature import CountVectorizer

df = spark.createDataFrame([(0, "a b c".split(" ")),(1, "a b b c a".split(" "))], ["id", "words"])
cv = CountVectorizer(inputCol="words", outputCol="features", vocabSize=3, minDF=2.0)
result = cv.fit(df).transform(df)
result.show(truncate=False)
```

For this code along we will build a spam filter! We'll use the various NLP tools we learned about as well as a new classifier, Naive Bayes.

```python
data = spark.read.csv("smsspamcollection/SMSSpamCollection",inferSchema=True,sep='\t')
data = data.withColumnRenamed("_c0", "class").withColumnRenamed("_c1", "text")
data.printSchema()
#root
# |-- class: string (nullable = true)
# |-- text: string (nullable = true)

## Clean and Prepare the Data
from pyspark.sql.functions import length
data = data.withColumn('length', length(data["text"]))
data.groupby("class").mean().show()
#+-----+-----------------+
#|class|      avg(length)|
#+-----+-----------------+
#|  ham|71.45431945307645|
#| spam|138.6706827309237|
#+-----+-----------------+
```

### Feature Transformations

```python
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer

tokenizer = Tokenizer(inputCol="text", outputCol="token_text")
stopremove = StopWordsRemover(inputCol="token_text", outputCol="stop_tokens")
count_vec = CountVectorizer(inputCol="stop_tokens", outputCol="c_vec")
idf = IDF(inputCol="c_vec", outputCol="tf_idf")
ham_spam_to_num = StringIndexer(inputCol="class", outputCol="label")

from pyspark.ml.feature import VectorAssembler
clean_up = VectorAssembler(inputCols=["tf_idf", "length"], outputCol="features")
```

### The Model

```python
from pyspark.ml.classification import NaiveBayes

nb = NaiveBayes()
```

### Pipeline

```python
from pyspark.ml import Pipeline

data_prep_pipe = Pipeline(stages=[ham_spam_to_num, tokenizer, stopremove, count_vec, idf, clean_up])
cleaner = data_prep_pipe.fit(data)
clean_data = cleaner.transform(data)
```

### Training and Evaluation

```python
clean_data = clean_data.select(["label", "features"])
training, testing = clean_data.randomSplit([0.7, 0.3])
spam_predictor = nb.fit(training)
test_results = spam_predictor.transform(testing)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(test_results)
print("Accuracy of model at predicting spam was: {}".format(acc))
# Accuracy of model at predicting spam was: 0.928636111661
```

## <a name="Spark Streaming">Spark Streaming</a>


References:

* [Spark official doc](https://spark.apache.org/docs/latest/index.html)
