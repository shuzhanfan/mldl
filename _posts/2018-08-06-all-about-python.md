---
layout:         post
title:          All about Python
subtitle:
card-image:     /mldl/assets/images/cards/cat12.gif
date:           2018-08-06 09:00:00
tags:           [python]
categories:     [python]
post-card-type: image
mathjax:        true
---

* <a href="#Deep Copy and Shallow Copy">Deep Copy and Shallow Copy</a>
* <a href="#Pickling">Pickling</a>
* <a href="#Collections">Collections</a>
  * <a href="#defaultdict">defaultdict</a>
  * <a href="#OrderedDict">OrderedDict</a>
  * <a href="#Counter">Counter</a>
  * <a href="#deque">deque</a>
  * <a href="#namedtuple">namedtuple</a>
* <a href="#Generators">Generators</a>
* <a href="#"></a>
* <a href="#"></a>
* <a href="#"></a>
* <a href="#"></a>
* <a href="#"></a>
* <a href="#"></a>

## <a name="Deep Copy and Shallow Copy">Deep Copy and Shallow Copy</a>

Python defines a module which allows to deep copy or shallow copy mutable object using the inbuilt functions present in the module `copy`.

Assignment statements in Python do not copy objects, they create bindings between a target and an object. For collections that are mutable or contain mutable items, a copy is sometimes needed so one can change one copy without changing the other.

In case of **deep copy**, a copy of object is copied in other object. It means that any changes made to a copy of object do not reflect in the original object.

In python, this is implemented using “copy.deepcopy()” function.

![deep copy](/mldl/assets/images/2018-08-06/deep_copy.jpg)

```python
import copy

l1 = [1, 2, [3,4], 5]
l2 = copy.deepcopy(l1)
l2[2][0] = 6
l2 # [1, 2, [6, 4], 5]
l1 # [1, 2, [3, 4], 5]
```

In the above example, the change made in the list did not effect in other list, indicating the list is deep copied.

In case of **shallow copy**, a reference of object is copied in other object. It means that any changes made to a copy of object do reflect in the original object.

In python, this is implemented using “copy()” function.

![shallow copy](/mldl/assets/images/2018-08-06/shallow_copy.jpg)

```python
l1 = [1,2,[3,4],5]
l2 = copy.copy(l1)
l2[2][0] = 6
l2 # [1, 2, [6, 4], 5]
l1 # [1, 2, [6, 4], 5]
```

In the above example, the change made in the list did effect in other list, indicating the list is shallow copied.

Note: The difference between shallow and deep copying is only relevant for compound objects (objects that contain other objects, like lists or class instances):

* A shallow copy constructs a new compound object and then (to the extent possible) inserts references into it to the objects found in the original.
* A deep copy constructs a new compound object and then, recursively, inserts copies into it of the objects found in the original.

```python
l1 = [1, 2, 3, 4]
l2 = copy.copy(l1)
l2[2] = 5
l2 # [1, 2, 5, 4]
l1 # [1, 2, 3, 4]

l1 = [1, 2, [3,4], 5]
l2 = copy.copy(l1)
l2[0] = 6
l2 # [6, 2, [3, 4], 5]
l1 # [1, 2, [3, 4], 5]
```

## <a name="Pickling">Pickling</a>

The `pickle` module implements a fundamental, but powerful algorithm for serializing and de-serializing a Python object structure. “Pickling” is the process whereby a Python object hierarchy is converted into a byte stream, and “unpickling” is the inverse operation, whereby a byte stream is converted back into an object hierarchy. Pickling (and unpickling) is alternatively known as “serialization”, “marshalling,” or “flattening”,

Any object in Python can be pickled so that it can be saved on disk. What pickle does is that it “serializes” the object first before writing it to file. Pickling is a way to convert a python object (list, dict, etc.) into a character stream. The idea is that this character stream contains all the information necessary to reconstruct the object in another python script.

To serialize an object hierarchy, you first create a pickler, then you call the pickler’s `dump()` method. To de-serialize a data stream, you first create an unpickler, then you call the unpickler’s `load()` method.

`pickle.dump(obj, file)` writes a pickled representation of *obj* to the open file object *file*.

`pickle.load(file)` reads a string from the open file object *file* and interpret it as a pickle data stream, reconstructing and returning the original object hierarchy.

## <a name="Collections">Collections</a>

Much of what you need to do with Python can be done using built-in containers like `dict`, `list`, `set`, and `tuple`. But these aren't always the most optimal. In this guide, I'll cover why and when to use collections and provide interesting examples of each. This is designed to supplement the documentation with examples and explanation, not replace it.

Python ships with a module that contains a number of container data types called `Collections`. We will talk about a few of them and discuss their usefulness.

The ones which we will talk about are:

* <a href="#defaultdict">defaultdict</a>
* <a href="#OrderedDict">OrderedDict</a>
* <a href="#Counter">Counter</a>
* <a href="#deque">deque</a>
* <a href="#namedtuple">namedtuple</a>

### <a name="defaultdict">`defaultdict`</a>

Suppose you have a sequence of key-value pairs. Perhaps you are keeping track of how many miles you run each day, and you want to know which day of the week you are most active.

```python
from collections import defaultdict

days = [('monday', 2.5), ('wednesday', 2), ('friday', 1.5), ('monday', 3), ('tuesday', 3.5), ('thursday', 2), ('friday', 2.5)]
active_days = defaultdict(float)
for k, v in days:
    active_days[k] += v
# defaultdict(<type 'float'>, {'tuesday': 3.5, 'friday': 4.0, 'thursday': 2.0, 'wednesday': 2.0, 'monday': 5.5})
```

This can be accomplished using many other data types, but `defaultdict` allows us to specify the default type of the value. This is simpler and faster than using a regular `dict` with `dict.setdefault`.

You pass in the default type upon instantiation. Then you can immediately begin setting values even if the key is not yet set. This would obviously throw a `KeyError` if you tried this with a normal dictionary.

Here is an example using a list as the default value. Here we have a list of sets. Each set has a letter and a number, and the letters are both uppercase and lowercase. Suppose we want to make a list of values grouped by letter ignoring case.

```python
letters = [('A', 10), ('B', 3), ('C', 4), ('a', 36), ('b', 8), ('c', 10)]
grouped_letters = defaultdict(list)
for k, v in letters:
    grouped_letters[k.lower()].append(v)
# defaultdict(<type 'list'>, {'a': [10, 36], 'c': [4, 10], 'b': [3, 8]})
```

### <a name="OrderedDict">`OrderedDict`</a>

`OrderedDict` act just like regular dictionaries except they remember the order that items were added. This matters primarily when you are iterating over the `OrderedDict` as the order will reflect the order in which the keys were added.

```python
## A regular dictionary doesn't care about order:
d = {}
d['a'] = 1
d['b'] = 10
d['c'] = 8
for letter in d:
    print letter
# a
# c
# b

## You can imagine what an OrderedDict would do:
d = OrderedDict()
d['a'] = 1
d['b'] = 10
d['c'] = 8
for letter in d:
    print letter
# a
# b
# c
```

It simply maintains the order. As a subclass of dict, `OrderedDict` has all of the same methods. Being that it cares about order, there are a few added methods. `OrderedDict.popitem` pops the most recently added element (LIFO), unless `last=False` is specified in which case it takes the first element added (FIFO).

```python
from collections import OrderedDict
d = OrderedDict()
d['a'] = 1
d['b'] = 10
d['c'] = 8
d.popitem()
# ('c', 8)
d
# OrderedDict([('a', 1), ('b', 10)])
d.popitem(last=False)
# ('a', 1)
d
# OrderedDict([('b', 10)])
```

Since order matters in iteration, you can iterate over an `OrderedDict` backwards using reversed().

```python
d = OrderedDict()
d['a'] = 1
d['b'] = 10
d['c'] = 8
for letter in reversed(d):
    print letter
# c
# b
# a
```

### <a name="Counter">`Counter`</a>

A counter is a dictionary-like object designed to keep tallies. With a counter, the key is the item to be counted and value is the count. You could certainly use a regular dictionary to keep a count, but a counter provides much more control.

```python
## A counter object ends up looking just like a dictionary and even contains a dictionary interface.
ctr = Counter({'birds': 200, 'lizards': 340, 'hamsters': 120})
ctr['hamsters'] # 120
```

One thing to note is that if you try to access a key that doesn't exist, the counter will return 0 rather than raising a `KeyError` as a standard dictionary would. Counters come with a brilliant set of methods that will make your life easier if you learn how to use them.

```python
from collections import Counter
## Get the most common word in a text file
import re
words = re.findall(r'\w+', open('ipencil.txt').read().lower())
Counter(words).most_common(1) # [('the', 148)]

## Get the count of each number in a long string of numbers
numbers = """
73167176531330624919225119674426574742355349194934
96983520312774506326239578318016984801869478851843
85861560789112949495459501737958331952853208805511
12540698747158523863050715693290963295227443043557
66896648950445244523161731856403098711121722383113
62229893423380308135336276614282806444486645238749
30358907296290491560440772390713810515859307960866
70172427121883998797908792274921901699720888093776
65727333001053367881220235421809751254540594752243
52584907711670556013604839586446706324415722155397
53697817977846174064955149290862569321978468622482
83972241375657056057490261407972968652414535100474
82166370484403199890008895243450658541227588666881
16427171479924442928230863465674813919123162824586
17866458359124566529476545682848912883142607690042
24219022671055626321111109370544217506941658960408
07198403850962455444362981230987879927244284909188
84580156166097919133875499200524063689912560717606
05886116467109405077541002256983155200055935729725
71636269561882670428252483600823257530420752963450
"""
numbers = re.sub("\n", "", numbers)
Counter(numbers).most_common()
#[('2', 112),
# ('5', 107),
# ('4', 107),
# ('6', 103),
# ('9', 100),
# ('8', 100),
# ('1', 99),
# ('0', 97),
# ('7', 91),
# ('3', 84)]
```

`most_common` is a very valuable method. If you pass in an integer as the first parameter, it will return that many results. If you call it without any arguments, it will return the frequency of all elements. As you can see it returns a list of tuples - the tuple structured like this (value, frequency).

When dealing with multiple `Counter` objects you can perform operations against them. For instance, you can add two counters which would add the counts for each key. You can also perform intersection or union. If I wanted to compare the values for given keys between two counters, I can return the minimum or maximum values only.

For example, a student has taken 4 quizzes two times each. She is allowed to keep the highest score for each quiz.

```python
first_attempt = Counter({1: 90, 2: 65, 3: 78, 4: 88})
second_attempt = Counter({1: 88, 2: 84, 3: 95, 4: 92})
final = first_attempt | second_attempt
final # Counter({3: 95, 4: 92, 1: 90, 2: 84})
```

### <a name="deque">`deque`</a>

`deque` stands for "double-ended queue" and is used as a stack or queue. Although lists offer many of the same operations, they are not optimized for variable-length operations.

How do you know when to use a deque vs. a list?

Basically if you're structuring the data in a way that requires quickly appending to either end or retrieving from either end then you would want to use a deque. For instance, if you're creating a queue of objects that need to be processed and you want to process them in the order they arrived, you would want to append new objects to one end and pop objects off of the other end for processing.

```python
from collections import deque

queue = deque()
# append values to wait for processing
queue.appendleft("first")
queue.appendleft("second")
queue.appendleft("third")
# pop values when ready
process(queue.pop()) # would process "first"
# add values while processing
queue.appendleft("fourth")
# what does the queue look like now?
queue # deque(['fourth', 'third', 'second'])
```

As you can see we're adding items to the left and popping them from the right. Deque provides four commonly used methods for appending and popping from either side of the queue: `append`, `appendleft`, `pop`, and `popleft`.

In the above example we started with an empty deque, but we can also create a deque from another iterable.

```python
numbers = [0, 1, 2, 3, 5, 7, 11, 13]
queue = deque(numbers)
print queue # deque([0, 1, 2, 3, 5, 7, 11, 13])

queue = deque(range(0, 10))
print queue # deque([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

### <a name="namedtuple">`namedtuple`</a>

A namedtuple is a ... named tuple. When you use a standard tuple it's difficult to convey the meaning of each position of the tuple. A named tuple is just like a normal tuple, but it allows you to give names to each position making the code more readable and self-documenting. Also with a namedtuple you can access the positions by name as well as index.

```python
from collections import namedtuple

## To instantiate we pass in the name of the type we want to create. Then we pass in a list of field names.
coordinate = namedtuple('Coordinate', ['x', 'y'])
## Now when we want to use our named tuple, coordinate, we can use it like a tuple.
c = coordinate(10, 20)
## Or we can instantiate by name:
c = coordinate(x=10, y=20)
```

And just like a normal tuple we can still access by index and unpack, but our namedtuple allows to access values to name as well.

```python
x, y = c
x, y # (10, 20)
c.x  # 10
c.y  # 20
c[0] # 10
c[1] # 20
```

## <a name="Generators">Generators</a> ([ref](https://jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained/))

### What is a Python Generator (Textbook Definition)

A Python generator is a function which returns a *generator iterator* (just an object we can iterate over) by calling `yield`. `yield` may be called with a value, in which case that value is treated as the "generated" value. The next time `next()` is called on the generator iterator (i.e. in the next step in a `for` loop, for example),the generator resumes execution from where it called `yield`, not from the beginning of the function. All of the state, like the values of local variables, is recovered and the generator continues to execute until the next call to `yield`.

### Subroutines and Coroutines

When we call a normal Python function, execution starts at function's first line and continues until a `return` statement, `exception`, or the end of the function (which is seen as an implicit `return None`) is encountered. Once a function returns control to its caller, that's it. Any work done by the function and stored in local variables is lost. A new call to the function creates everything from scratch.

This is all very standard when discussing functions (more generally referred to as **subroutines**) in computer programming. There are times, though, when it's beneficial to have the ability to create a "function" which, instead of simply returning a single value, is able to yield a series of values. To do so, such a function would need to be able to "save its work," so to speak.

I said, "yield a series of values" because our hypothetical function doesn't "return" in the normal sense. `return` implies that the function is returning control of execution to the point where the function was called. "Yield," however, implies that the transfer of control is temporary and voluntary, and our function expects to regain it in the future.

In Python, "functions" with these capabilities are called `generators`, and they're incredibly useful. `generators` (and the `yield` statement) were initially introduced to give programmers a more straightforward way to write code responsible for producing a series of values. Previously, creating something like a random number generator required a class or module that both generated values and kept track of state between calls. With the introduction of `generators`, this became much simpler.

To better understand the problem `generators` solve, let's take a look at an example. Throughout the example, keep in mind the core problem being solved: generating a series of values.

Note: Outside of Python, all but the simplest `generators` would be referred to as `coroutines`. I'll use the latter term later in the post. The important thing to remember is, in Python, everything described here as a `coroutine` is still a `generator`. Python formally defines the term `generator`; `coroutine` is used in discussion but has no formal definition in the language.

### Example: Fun With Prime Numbers

Suppose our boss asks us to write a function that takes a `list` of `int`s and returns some Iterable containing the elements which are prime1 numbers. Remember, an Iterable is just an object capable of returning its members one at a time. "Simple," we say, and we write the following:

```python
def get_primes(input_list):
    return (element for element in input_list if is_prime(element))
```

A few days later, our boss comes back and tells us she's run into a small problem: she wants to use our `get_primes` function on a very large list of numbers. In fact, the list is so large that merely creating it would consume all of the system's memory. To work around this, she wants to be able to call `get_primes` with a start value and get all the primes larger than start.

Once we think about this new requirement, it becomes clear that it requires more than a simple change to `get_primes`. Clearly, we can't return a list of all the prime numbers from start to infinity. The chances of solving this problem using a normal function seem bleak.

Before we give up, let's determine the core obstacle preventing us from writing a function that satisfies our boss's new requirements. Thinking about it, we arrive at the following: **functions only get one chance to return results, and thus must return all results at once**. It seems pointless to make such an obvious statement; "functions just work that way," we think. The real value lies in asking, "but what if they didn't?"

Imagine what we could do if `get_primes` could simply return the next value instead of all the values at once. It wouldn't need to create a list at all. No list, no memory issues. Since our boss told us she's just iterating over the results, she wouldn't know the difference. Functions, though, can't do this. When they `return`, they're done for good. Even if we could guarantee a function would be called again, we have no way of saying, "OK, now, instead of starting at the first line like we normally do, start up where we left off at line 4." Functions have a single `entry point`: the first line.

### Enter the Generator

This sort of problem is so common that a new construct was added to Python to solve it: the `generator`. A generator "generates" values. Creating generators was made as straightforward as possible through the concept of `generator functions`, introduced simultaneously.

A `generator function` is defined like a normal function, but whenever it needs to generate a value, it does so with the `yield` keyword rather than `return`. If the body of a `def` contains `yield`, the function automatically becomes a generator function (even if it also contains a `return` statement). There's nothing else we need to do to create one.

generator functions create `generator iterators`. That's the last time you'll see the term generator iterator, though, since they're almost always referred to as "generators". Just remember that a `generator` is a special type of `iterator`. To be considered an iterator, generators must define a few methods, one of which is `__next__()`. To get the next value from a generator, we use the same built-in function as for iterators: `next()`.

This point bears repeating: **to get the next value from a generator, we use the same built-in function as for iterators: `next()`**.

(`next()` takes care of calling the generator's `__next__()` method). Since a generator is a type of iterator, it can be used in a `for` loop.

So whenever `next()` is called on a generator, the generator is responsible for passing back a value to whomever called `next()`. It does so by calling `yield` along with the value to be passed back (e.g. yield 7). The easiest way to remember what `yield` does is to think of it as `return` (plus a little magic) for generator functions.

Again, this bears repeating: `yield` is just `return` (plus a little magic) for generator functions.

### Magic?

What's the magic part? Glad you asked! When a `generator function` calls `yield`, the "state" of the generator function is frozen; the values of all variables are saved and the next line of code to be executed is recorded until `next()` is called again. Once it is, the generator function simply resumes where it left off. If `next()` is never called again, the state recorded during the yield call is (eventually) discarded.

Let's rewrite `get_primes` as a generator function. Using a simple `while` loop, we can create our own infinite sequence:

```python
def get_primes(number):
    while True:
        if is_prime(number):
            yield number
        number += 1
```

If a generator function calls `return` or reaches the end of its definition, a `StopIteration` exception is raised. This signals to whoever was calling `next()`` that the generator is exhausted (this is normal iterator behavior). It is also the reason using the `while True`: loop is present in `get_primes`. If it weren't, the first time `next()` was called we would check if the number is prime and possibly yield it. If `next()` were called again, we would uselessly add 1 to number and hit the end of the generator function (causing `StopIteration` to be raised). Once a generator has been exhausted, calling `next()` on it will result in an error, so you can only consume all the values of a generator once.

Thus, the `while` loop is there to make sure we never reach the end of `get_primes`. It allows us to generate a value for as long as `next()` is called on the generator. This is a common idiom when dealing with infinite series (and `generators` in general).

### Visualizing the flow

Let's go back to the code that was calling `get_primes`: `solve_number_10`.

```python
def solve_number_10():
    total = 2
    for next_prime in get_primes(3):
        if next_prime < 2000000:
            total += next_prime
        else:
            print(total)
            return
```

It's helpful to visualize how the first few elements are created when we call `get_primes` in `solve_number_10`'s `for` loop. When the `for` loop requests the first value from `get_primes`, we enter `get_primes` as we would in a normal function.

1. We enter the `while` loop on line 3
2. The `if` condition holds (3 is prime)
3. We yield the value 3 and control to `solve_number_10`.

Then, back in `solve_number_10`:

1. The value 3 is passed back to the `for` loop
2. The `for` loop assigns `next_prime` to this value
3. `next_prime` is added to `total`
4. The `for` loop requests the next element from `get_primes`

This time, though, instead of entering `get_primes` back at the top, we resume at line 5, where we left off. Most importantly, `number` still has the same value it did when we called `yield` (i.e. 3). Remember, `yield` both passes a value to whoever called `next()`, and saves the "state" of the generator function. Clearly, then, `number` is incremented to 4, we hit the top of the `while` loop, and keep incrementing number until we hit the next prime number (5). Again we yield the value of number to the `for` loop in `solve_number_10`. This cycle continues until the `for` loop stops (at the first prime greater than 2,000,000).


References:
