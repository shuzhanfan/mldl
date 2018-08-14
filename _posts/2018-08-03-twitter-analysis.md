---
layout:         post
title:          Twitter Analysis
subtitle:
card-image:     /mldl/assets/images/cards/cat11.gif
date:           2018-08-03 09:00:00
tags:           [nlp]
categories:     [nlp]
post-card-type: image
mathjax:        true
---

* <a href="#DATA PREPROCESSING">DATA PREPROCESSING</a>
    * <a href="#Introduction of the data">Introduction of the data</a>
    * <a href="#Pandas Tips before everything">Pandas Tips before everything<a>
    * <a href="#Pandas functions">Pandas functions</a>
    * <a href="#Data Preprocessing">Data Preprocessing</a>
* <a href="#EDA DATA VISUALIZATION">EDA & DATA VISUALIZATION</a>
    * <a href="#Zipf's Law">Zipf's Law</a>
    * <a href="#Tweet Tokens Visualization">Tweet Tokens Visualization</a>
* <a href="#FEATURE EXTRACTION">FEATURE EXTRACTION</a>
    * <a href="#CountVectorizer">CountVectorizer</a>
    * <a href="#Tfidf Vectorizer">Tfidf Vectorizer</a>
* <a href="#Doc2Vec">Doc2Vec</a>
  * <a href="#DBOW">DBOW</a>
  * <a href="#DMC">DMC</a>
  * <a href="#DMM">DMM</a>
  * <a href="#DBOW DMC">DBOW + DMC</a>
  * <a href="#DBOW DMM">DBOW + DMM</a>
  * <a href="#Phrase Modeling">Phrase Modeling</a>
* <a href="#Feature Selection Dimensionality reduction">FEATURE SELECTION & DIMENSIONALITY REDUCTION</a>
  * <a href="#Chi2 Feature Selection">Chi2 Feature Selection</a>
  * <a href="#PCA Dimnsionality Reduction">PCA Dimnsionality Reduction</a>
* <a href="#NEURAL NETWORKS WITH TFIDF VECTORS">NEURAL NETWORKS WITH TFIDF VECTORS</a>
* <a href="#NEURAL NETWORKS WITH Doc2Vec and Word2Vec">NEURAL NETWORKS WITH Doc2Vec and Word2Vec</a>
    * <a href="#Doc2Vec">Doc2Vec</a>
    * <a href="#Word2Vec">Word2Vec</a>
* <a href="#CNN with Word2Vec">CNN with Word2Vec</a>


## <a name="DATA PREPROCESSING">DATA PREPROCESSING</a>

## <a name="Introduction of the data">Introduction of the data</a>

The dataset for training, I chose “Sentiment140”, which originated from Stanford University. More info on the dataset can be found from the [<u>link</u>](http://help.sentiment140.com/for-students/). The dataset can be downloaded from the below [<u>link</u>](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip).

The data is a CSV with emoticons removed. Data file format has 6 fields:

* 0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
* 1 - the id of the tweet (2087)
* 2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
* 3 - the query (lyx). If there is no query, then this value is NO_QUERY
* 4 - the user that tweeted (robotickilldozr)
* 5 - the text of the tweet (Lyx is cool)

## <a name="Pandas Tips before everything">Pandas Tips before everything</a>

1. `axis` in pandas: `axis=0` along the rows (`index` in pandas), and `axis=1` along the columns (`columns` in pandas)
2. `df["sentiment"]` returns a pandas `Series`, and `df[["sentiment"]]` returns a pandas `DataFrame` object.
3. pandas `apply()` function applies function along input axis of DataFrame. If we do `df["sentiment"].apply(lambda x: len(x), axis=0)`, there will be an error because `df["sentiment"]` returns a `Series` object and it doesn't make sense to have an `axis` hence the error. We can just do `df["sentiment"].apply(lambda x: len(x))` and it will apply the function to each row.
4. For pandas `apply()` function, `axis=0` means apply function to each **column**, and `axis=1` means apply function to each **row**.

## <a name="Pandas functions">Pandas functions</a>

1. `pandas.Series.value_counts()` returns object containing counts of unique values.
2. `pandas.DataFrame.dropna()` removes missing values.
3. `pandas.DataFrame.reset_index()` For DataFrame with multi-level index, return new DataFrame with labeling information in the columns under the index names.
4. `pandas.DataFrame.values` returns a Numpy representation of the DataFrame.
5. `pandas.Series.str.cat()` concatenates strings in the Series/Index with given separator. Returns a string.
6. `pandas.DataFrame.sort_values()` sorts by the values along either axis.


## <a name="Data Preprocessing">Data Preprocessing</a>

### Import packages

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re
from nltk.tokenize import WordPunctTokenizer
from wordcloud import WordCloud
from scipy.stats import hmean
from scipy.stats import norm
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import NearestCentroid
from sklearn.feature_selection import SelectFromModel
%matplotlib inline
plt.style.use("seaborn-darkgrid")
```

### Read data

```python
cols = ['sentiment','id','date','query_string','user','text']
df = pd.read_csv("../data/training.1600000.processed.noemoticon.csv", header=None, names=cols)
```

### Drop unused columns

```python
df.drop(['id','date','query_string','user'], axis=1, inplace=True)
df["sentiment"].value_counts()
# 4    800000
# 0    800000
# Name: sentiment, dtype: int64
```

### Add column: pre_clean_len

```python
df["pre_clean_len"] = df["text"].apply(lambda x: len(x))
```

### Map sentiment value of 4 (positive) to 1

```python
df["sentiment"] = df["sentiment"].apply(lambda x: 1 if x==4 else x)
```

### Define data cleaning function

The order of the data cleaning steps is:

1. Souping
2. url address(‘http:’pattern), twitter ID ('@') removing and url address(‘www.'pattern) removing
3. BOM removing
4. negation handling
5. removing numbers and special characters
6. lower-case
7. tokenizing and joining

```python
pat1 = r"@[A-Za-z0-9_]+"
pat2 = r"https?://[^ ]+"
pat3 = r"www.[^ ]+"
combined_pat = r"|".join((pat1, pat2, pat3))
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                 "don't":"do not", "doesn't":"does not", "didn't":"did not",
                 "haven't":"have not", "hasn't":"has not", "hadn't":"had not",
                 "won't":"will not", "wouldn't":"would not",
                 "can't":"can not", "couldn't":"could not",
                 "shouldn't":"should not",
                 "mightn't":"might not",
                 "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
tok = WordPunctTokenizer()


def tweet_clean(text):
  # 1. decode html to general text
  soup = BeautifulSoup(text, "lxml")
  souped = soup.get_text()
  # 2. '@' and URL links handling
  stripped = re.sub(combined_pat, "", souped)
  # 3. UTF-8 BOM (byte order mark) handling
  try:
      clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
  except:
      clean = stripped
  # 4. negation words handling
  neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], clean)
  # 5. numbers or special characters handling
  letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
  # 6. convert to lowercase
  lower_case = letters_only.lower()
  # 7. remove unneccessary white spaces and one character
  words = [x for x in tok.tokenize(lower_case) if len(x) > 1]
  text_cleaned = (" ".join(words)).strip()
  return text_cleaned
```

### Clean data

```python
%%time
print "Cleaning Twitter messages...\n"
clean_tweets = []
for i in xrange(0, len(df)):
    if ((i+1) % 100000) == 0:
        print "Tweets %d of %d has been processed" % ( i+1, len(df) )
    clean_tweets.append(tweet_clean(df.text[i]))
```

### Save data

```python
clean_df = pd.DataFrame(clean_tweets, columns=["text"])
clean_df["target"] = df.sentiment
clean_df.to_csv("clean_tweet.csv", encoding="utf-8")
```

### Play with data

```python
## Read data
my_df = pd.read_csv("clean_tweet.csv", index_col=0) # index_col: Column to use as the row labels of the DataFrame.
my_df.info()
#<class 'pandas.core.frame.DataFrame'>
#Int64Index: 1600000 entries, 0 to 1599999
#Data columns (total 2 columns):
#text      1596019 non-null object
#target    1600000 non-null int64
#dtypes: int64(1), object(1)
#memory usage: 36.6+ MB

## Drop the rows whose text are NaN
my_df.dropna(inplace=True)
my_df.info()
#<class 'pandas.core.frame.DataFrame'>
#Int64Index: 1596019 entries, 0 to 1599999
#Data columns (total 2 columns):
#text      1596019 non-null object
#target    1596019 non-null int64
#dtypes: int64(1), object(1)
#memory usage: 36.5+ MB

## Reset the index
my_df.reset_index(drop=True, inplace=True)

## WordCloud
neg_tweets_df = my_df[my_df["target"]==0]
neg_tweets_nd = neg_tweets_df.text.values # convert to numpy arrays
neg_tweets_text = pd.Series(neg_tweets_nd).str.cat(sep=" ")
wc = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_tweets_text)

plt.figure(figsize=(12,10))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")

pos_tweets_df = my_df[my_df["target"]==1]
pos_tweets_nd = pos_tweets_df.text.values
pos_tweets_text = pd.Series(pos_tweets_nd).str.cat(sep=" ")
wc1 = WordCloud(width=1600, height=800, max_font_size=200).generate(pos_tweets_text)

plt.figure(figsize=(12,10))
plt.imshow(wc1, interpolation="bilinear")
plt.axis("off")

## Data visualization preparation

count_vectorizor = CountVectorizer()
count_vectorizor.fit(my_df.text)
neg_mat = count_vectorizor.transform(my_df[my_df.target==0].text)
pos_mat = count_vectorizor.transform(my_df[my_df.target==1].text)
neg_tf  = np.sum(neg_mat, axis=0)
pos_tf  = np.sum(pos_mat, axis=0)
neg = np.squeeze(np.asarray(neg_tf))
pos = np.squeeze(np.asarray(pos_tf))
term_freq_df = pd.DataFrame([neg,pos], columns=count_vectorizor.get_feature_names()).transpose()
term_freq_df.columns=["negative", "positive"]
term_freq_df["total"] = term_freq_df.negative + term_freq_df.positive

term_freq_df.sort_values(by="total", ascending=False).iloc[:10]
term_freq_df.to_csv("term_freq_df.csv", encoding="utf-8")
```

## <a name="EDA DATA VISUALIZATION">EDA & DATA VISUALIZATION</a>

### <a name="Zipf's Law">Zipf's Law</a>

Zipf’s Law states that a small number of words are used all the time, while the vast majority are used very rarely. And “given some corpus of natural language utterances, the frequency of any word is inversely proportional to its rank in the frequency table. Thus the most frequent word will occur approximately twice as often as the second most frequent word, three times as often as the third most frequent word, etc.”

```python
## View Zipf's Law
term_freq_df = pd.read_csv("term_freq_df.csv", index_col=0)
x_coords = np.arange(500)
plt.figure(figsize=(10,8))
zipf_values = [term_freq_df.sort_values(by="total", ascending=False)["total"][0]/(i+1)**1 for i in x_coords]
plt.bar(x_coords, term_freq_df.sort_values(by="total", ascending=False)["total"][:500], align="center", alpha=0.5)
plt.plot(x_coords, zipf_values, color="r", linestyle="--", linewidth=2,alpha=0.5)
plt.ylabel("Frequency")
plt.title("Top 500 tokens in tweets")
```

### <a name="Tweet Tokens Visualization">Tweet Tokens Visualization</a>

Let's remove stop words first and re-create term frequencies dataframe:

```python
my_df = pd.read_csv("clean_tweet.csv", index_col=0)
my_df.dropna(inplace=True)
my_df.reset_index(drop=True, inplace=True)
count_vectorizer = CountVectorizer(stop_words="english", max_features=10000)
count_vectorizer.fit(my_df.text)

neg_mat = count_vectorizer.transform(my_df[my_df["target"]==0].text)
pos_mat = count_vectorizer.transform(my_df[my_df["target"]==1].text)
neg_tf  = np.sum(neg_mat, axis=0)
pos_tf  = np.sum(pos_mat, axis=0)
neg     = np.squeeze(np.asarray(neg_tf))
pos     = np.squeeze(np.asarray(pos_tf))
term_freq_df2 = pd.DataFrame([neg, pos], columns=count_vectorizer.get_feature_names()).transpose()
term_freq_df2.columns = ["negative", "positive"]
term_freq_df2["total"] = term_freq_df2["negative"] + term_freq_df2["positive"]
```

And visualize top 50 positive tokens on a bar chart:

```python
x_coords = np.arange(50)
plt.figure(figsize=(12,10))
plt.bar(x_coords, term_freq_df2.sort_values(by="positive", ascending=False)["positive"][:50], align="center", alpha=0.5)
plt.xticks(x_coords, term_freq_df2.sort_values(by="positive", ascending=False)["positive"][:50].index, rotation="vertical")
plt.title("Top 50 tokens in positive tweets")
plt.xlabel("Top 50 positive tokens")
plt.ylabel("Top 50 tokens in positive tweets")
```

Visualize top 50 negative tokens on a bar chart：

```python
x_coords = np.arange(50)
plt.figure(figsize=(12,10))
plt.bar(x_coords, term_freq_df2.sort_values(by="negative", ascending=False)["negative"][:50], align="center", alpha=0.5)
plt.xticks(x_coords, term_freq_df2.sort_values(by="negative", ascending=False)["negative"][:50].index, rotation="vertical")
plt.title("Top 50 tokens in negative tweets")
plt.xlabel("Top 50 negative tokens")
plt.ylabel("Top 50 tokens in negative tweets")
```

Visualize token negative frequency vs. token positive frequency:

```python
plt.figure(figsize=(12,10))
plt.scatter(term_freq_df2.negative, term_freq_df2.positive)
plt.ylabel('Positive Frequency')
plt.xlabel('Negative Frequency')
plt.title('Negative Frequency vs Positive Frequency')
```

We now explore to find a meaningful metric which can characterize important tokens in each class.

The **first version** is:
{% raw %}
$$
pos\_rate = \frac{positive\space frequency}{positive\space frequency + negative\space frequency}
$$
{% endraw %}

```python
term_freq_df2["pos_rate"] = term_freq_df2["positive"]*1. / term_freq_df2["total"]
term_freq_df2.sort_values(by="pos_rate", ascending=False).iloc[:10]
#             negative	positive	total	pos_rate
# dividends	     0	       83      	 83	    1.000000
# emailunlimited 0	      100	    100	    1.000000
# mileymonday	 0	      161	    161	    1.000000
# shareholder	 1	       80	     81	    0.987654
# fuzzball	     2	       99	    101	    0.980198
# recommends	 3	      109	    112	    0.973214
# delongeday	 6	      162	    168	    0.964286
# atcha	         3	       80	     83	    0.963855
# timestamp 	 3 	       68	     71	    0.957746
# shaundiviney	 4	       89	     93	    0.956989
```

Words with highest `pos_rate` have zero frequency in the negative tweets, but overall frequency of these words are too low to consider it as a guideline for positive tweets.

The **second version** is:
{% raw %}
$$
pos\_freq\_pct = \frac{positive\space frequency}{\sum{positive\space frequency}}
$$
{% endraw %}

```python
term_freq_df2['pos_freq_pct'] = term_freq_df2['positive'] * 1./term_freq_df2['positive'].sum()
term_freq_df2.sort_values(by='pos_freq_pct', ascending=False).iloc[:10]
#           negative   positive	total	pos_rate  pos_freq_pct
# just	     64004	    62944	126948	0.495825	0.014383
# good	     29209	    62118	91327	0.680171	0.014194
# day	     41374	    48186	89560	0.538030	0.011010
# love	     16990	    47694	64684	0.737338	0.010898
# like	     41050	    37520	78570	0.477536	0.008573
# lol	     23123	    36118	59241	0.609679	0.008253
# thanks	 5768	    34375	40143	0.856314	0.007855
# going	     33689	    30939	64628	0.478724	0.007070
# time	     27526	    30432	57958	0.525070	0.006954
# today	     38116	    30100	68216	0.441245	0.006878
```

But since `pos_freq_pct` is just the frequency scaled over the total sum of the frequency, the rank of `pos_freq_pct` is exactly the same as just the positive frequency.

The **third version** is combining `pos_rate` and `pos_freq_pct` and calculate the harmonic mean:
{% raw %}
$$
H = \frac{n}{\sum_{i=1}^{n}{\frac{1}{x_i}}}
$$
{% endraw %}

```python
def harmonic_mean_pos(x):
    pr  = x["pos_rate"]
    pfp = x["pos_freq_pct"]
    if pr > 0 and pfp > 0:
        return hmean([pr, pfp])
    else:
        return 0

term_freq_df2["pos_hmean"] = term_freq_df2.apply(harmonic_mean_pos, axis=1)
term_freq_df2.sort_values(by='pos_hmean', ascending=False).iloc[:10]
#           negative	positive	total	pos_rate	pos_freq_pct	pos_hmean
# just	    64004	     62944	    126948	0.495825	0.014383	     0.027954
# good	    29209	     62118	    91327	0.680171	0.014194	     0.027808
# day	    41374      	 48186	    89560	0.538030	0.011010	     0.021579
# love  	16990	     47694	    64684	0.737338	0.010898	     0.021479
# like  	41050	     37520	    78570	0.477536	0.008573	     0.016844
# lol	    23123	     36118	    59241	0.609679	0.008253	     0.016285
# thanks	5768	     34375	    40143	0.856314	0.007855	     0.015567
# going	    33689	     30939	    64628	0.478724	0.007070	     0.013933
# time	    27526	     30432	    57958	0.525070	0.006954         0.013726
# today	    38116	     30100	    68216	0.441245	0.006878	     0.013545
```

The harmonic mean rank seems like the same as `pos_freq_pct`. By calculating the harmonic mean, the impact of small value (in this case, `pos_freq_pct`) is too aggravated and ended up dominating the mean value. This is again exactly same as just the frequency value rank and doesn’t provide a much meaningful result.

The **fourth version** is CDF (Cumulative Distribution Function) value of both pos_rate and pos_freq_pct.

CDF can be explained as “distribution function of X, evaluated at x, is the probability that X will take a value less than or equal to x”.

```python
term_freq_df2["pos_rate_normcdf"]     = normcdf(term_freq_df2["pos_rate"])
term_freq_df2["pos_freq_pct_normcdf"] = normcdf(term_freq_df2["pos_freq_pct"])
```

Next, we calculate a harmonic mean of these two CDF values, as we did earlier.

```python
term_freq_df2["pos_normcdf_hmean"]    = hmean([term_freq_df2["pos_rate_normcdf"], term_freq_df2["pos_freq_pct_normcdf"]])
term_freq_df2.sort_values(by='pos_normcdf_hmean', ascending=False).iloc[:10]
#           negative	positive	total	pos_rate	pos_freq_pct	pos_hmean	pos_rate_normcdf	pos_freq_pct_normcdf	pos_normcdf_hmean
# welcome	620	6702	7322	0.915324	0.001531	0.003058	0.995611	0.999370	0.997487
# thank	    2282	15736	18018	0.873349	0.003596	0.007162	0.990769	1.000000	0.995363
# thanks	5768	34375	40143	0.856314	0.007855	0.015567	0.987741	1.000000	0.993833
# awesome	3821	14469	18290	0.791088	0.003306	0.006585	0.966978	1.000000	0.983212
# glad	    2273	8255	10528	0.784100	0.001886	0.003763	0.963602	0.999971	0.981450
# follow	2552	9154	11706	0.781992	0.002092	0.004172	0.962531	0.999996	0.980906
# enjoy	    1642	5876	7518	0.781591	0.001343	0.002681	0.962324	0.997443	0.979569
# sweet    	1610	5646	7256	0.778115	0.001290	0.002576	0.960492	0.996334	0.978084
# yay	    3165	10501	13666	0.768403	0.002399	0.004784	0.954987	1.000000	0.976975
# hello	    1122	4524	5646	0.801275	0.001034	0.002065	0.971433	0.982299	0.976836
```

Next step is to apply the same calculation to the negative frequency of each word:

```python
def harmonic_mean_neg(x):
    nr  = x["neg_rate"]
    nfp = x["neg_freq_pct"]
    if nr > 0 and nfp > 0:
        return hmean([nr, nfp])
    else:
        return 0

term_freq_df2["neg_rate"]             = term_freq_df2["negative"] * 1. / term_freq_df2["total"]
term_freq_df2["neg_freq_pct"]         = term_freq_df2["negative"] * 1. / term_freq_df2["negative"].sum()
term_freq_df2["neg_hmean"]            = term_freq_df2.apply(harmonic_mean_neg, axis=1)
term_freq_df2["neg_rate_normcdf"]     = normcdf(term_freq_df2["neg_rate"])
term_freq_df2["neg_freq_pct_normcdf"] = normcdf(term_freq_df2["neg_freq_pct"])
term_freq_df2["neg_normcdf_hmean"]    = hmean([term_freq_df2["neg_rate_normcdf"], term_freq_df2["neg_freq_pct_normcdf"]])
term_freq_df2.sort_values(by='neg_normcdf_hmean', ascending=False).iloc[:10]
#       negative	positive	total	pos_rate	pos_freq_pct	pos_hmean	pos_rate_normcdf	pos_freq_pct_normcdf	pos_normcdf_hmean	neg_rate	neg_freq_pct	neg_hmean	neg_rate_normcdf	neg_freq_pct_normcdf	neg_normcdf_hmean
# sad	27911	1510	29421	0.051324	0.000345	0.000685	0.002395	0.709549	0.004773	0.948676	0.006090	0.012101	0.997605	1.000000	0.998801
# hurts	7204	456	7660	0.059530	0.000104	0.000208	0.002810	0.503771	0.005588	0.940470	0.001572	0.003138	0.997190	0.999745	0.998466
# sick	14617	1419	16036	0.088488	0.000324	0.000646	0.004843	0.693298	0.009620	0.911512	0.003189	0.006356	0.995157	1.000000	0.997572
# sucks	9902	982	10884	0.090224	0.000224	0.000448	0.004999	0.610356	0.009917	0.909776	0.002160	0.004311	0.995001	0.999999	0.997494
# poor	7333	719	8052	0.089295	0.000164	0.000328	0.004915	0.557585	0.009745	0.910705	0.001600	0.003194	0.995085	0.999801	0.997437
# ugh	9056	998	10054	0.099264	0.000228	0.000455	0.005885	0.613511	0.011659	0.900736	0.001976	0.003943	0.994115	0.999995	0.997046
# missing	7282	991	8273	0.119787	0.000226	0.000452	0.008431	0.612132	0.016633	0.880213	0.001589	0.003172	0.991569	0.999781	0.995658
# headache	5317	421	5738	0.073371	0.000096	0.000192	0.003659	0.496583	0.007264	0.926629	0.001160	0.002317	0.996341	0.993846	0.995092
# hate	17207	2614	19821	0.131880	0.000597	0.001189	0.010347	0.868734	0.020450	0.868120	0.003754	0.007476	0.989653	1.000000	0.994800
# miss	30713	5676	36389	0.155981	0.001297	0.002573	0.015319	0.996499	0.030174	0.844019	0.006701	0.013296	0.984681	1.000000	0.992281
```

Visualize neg_hmean vs pos_hmean:

```python
plt.figure(figsize=(12,10))
plt.scatter(term_freq_df2["neg_hmean"], term_freq_df2["pos_hmean"], alpha=0.5)
plt.ylabel('Positive Rate and Frequency Harmonic Mean')
plt.xlabel('Negative Rate and Frequency Harmonic Mean')
plt.title('neg_hmean vs pos_hmean')
```

Visualize neg_normcdf_hmean vs pos_normcdf_hmean:

```python
plt.figure(figsize=(12,10))
plt.scatter(term_freq_df2["neg_normcdf_hmean"], term_freq_df2["pos_normcdf_hmean"], alpha=0.5)
plt.ylabel('Positive Rate and Frequency CDF Harmonic Mean')
plt.xlabel('Negative Rate and Frequency CDF Harmonic Mean')
plt.title('neg_normcdf_hmean vs pos_normcdf_hmean')
```

It seems like the harmonic mean of rate CDF and frequency CDF has created an interesting pattern on the plot. If a data point is near to the upper left corner, it is more positive, and if it is closer to the bottom right corner, it is more negative.


## <a name="FEATURE EXTRACTION">FEATURE EXTRACTION</a>

```python
my_df = pd.read_csv("clean_tweet.csv", index_col=0)
my_df.dropna(inplace=True)
my_df.reset_index(drop=True, inplace=True)
X = my_df.text
y = my_df.target
```

### Train test split

```python
random_seed = 2000
X_train, X_vali_test, y_train, y_vali_test = train_test_split(X, y, test_size=0.02, random_state=random_seed)
X_vali, X_test, y_vali, y_test = train_test_split(X_vali_test, y_vali_test, test_size=0.5, random_state=random_seed)

print("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".
      format(len(X_train),
             (len(X_train[y_train == 0]) / (len(X_train)*1.))*100,
             (len(X_train[y_train == 1]) / (len(X_train)*1.))*100)
     )

print("Validation set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".
      format(len(X_vali),
             (len(X_vali[y_vali == 0]) / (len(X_vali)*1.))*100,
             (len(X_vali[y_vali == 1]) / (len(X_vali)*1.))*100)
     )

print("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".
      format(len(X_test),
             (len(X_test[y_test == 0]) / (len(X_test)*1.))*100,
             (len(X_test[y_test == 1]) / (len(X_test)*1.))*100)
     )
```

### Baseline model: TextBlob

When comparing various machine learning algorithms, baseline provides a point of reference to compare. The most popular baseline is the Zero Rule (ZeroR). ZeroR classifier simply predicts the majority category (class). Although there is no predictability power in ZeroR, it is useful for determining a baseline performance as a benchmark for other classification methods.

Another baseline I wanted to compare the validation results with is TextBlob. TextBlob is a python library for processing textual data. Apart from other useful tools such as POS tagging, n-gram, The package has built-in sentiment classification.

```python
tbresult = [TextBlob(i).sentiment.polarity for i in X_vali]
tbpred   = [0 if s < 0 else 1 for s in tbresult]

conf_mat  = confusion_matrix(y_vali, tbpred, labels=[1,0])
confusion = pd.DataFrame(conf_mat, index=["positive", "negative"], columns=["predicted_positive", "predicted_negative"])
accuracy = accuracy_score(y_vali, tbpred)
class_report = classification_report(y_vali, tbpred)

print("Accuracy Score: {0:.2f}%".format(accuracy*100))
print("-"*80)
print("Confusion Matrix\n")
print(confusion)
print("-"*80)
print("Classification Report\n")
print(class_report)
# Accuracy Score: 60.63%
# --------------------------------------------------------------------------------
# Confusion Matrix
#
#           predicted_positive  predicted_negative
# positive                7092                 824
# negative                5460                2584
# --------------------------------------------------------------------------------
# Classification Report
#
#              precision    recall  f1-score   support
#
#           0       0.76      0.32      0.45      8044
#           1       0.57      0.90      0.69      7916
#
# avg / total       0.66      0.61      0.57     15960
```

### <a name="CountVectorizer">CountVectorizer</a>

We are gonna evaluate models with different number of features (i.e. the number of vocabulary's count vector). And also evaluate the model without stopwords, with stopwords, and without top 10 frequent stopwords.

```python
count_vector = CountVectorizer()
logis_regres = LogisticRegression()
num_features = np.arange(10000, 100001, 10000)

def accuracy_summary(pipeline, xtrain, ytrain, xtest, ytest):
    if len(y_test[y_test==0])/(len(y_test)*1.) > 0.5:
        null_accuracy = len(y_test[y_test==0])/(len(y_test)*1.)
    else:
        null_accuracy = 1 - len(y_test[y_test==0])/(len(y_test)*1.)
    t0 = time()
    model_fit = pipeline.fit(xtrain, ytrain)
    y_pred    = model_fit.predict(xtest)
    t1 = time()
    accuracy = accuracy_score(ytest, y_pred)
    print("null accuracy: {0:.2f}%".format(null_accuracy*100))
    print("accuracy score: {0:.2f}%".format(accuracy*100))
    if accuracy > null_accuracy:
        print("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100))
    elif accuracy == null_accuracy:
        print("model has the same accuracy with the null accuracy")
    else:
        print("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100))
    print("train and test time: {0:.2f}s".format(t1-t0))
    print("-"*80)
    return accuracy, (t1-t0)

def nfeature_accuracy_checker(vectorizer=count_vector, n_features=num_features, stopwords=None,
                              classifier=logis_regres, ngram_range=(1,1)):
    result = []
    print(classifier)
    print("\n")
    for n in num_features:
        vectorizer.set_params(stop_words=stopwords, max_features=n, ngram_range=ngram_range)
        pipeline = Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])
        print("Validation result for {} features".format(n))
        nfeature_accuracy, tt_time = accuracy_summary(pipeline, X_train, y_train, X_vali, y_vali)
        result.append((n, nfeature_accuracy, tt_time))
    return result


```

Now we run `nfeature_accuracy_checker()` on three different conditions. First with **stop words removal**, second **without stop words removal**, third with **custom defined stop words removal**.

```python
%%time
print("RESULT FOR UNIGRAM WITHOUT STOP WORDS\n")
feature_result_wosw = nfeature_accuracy_checker(stopwords='english')

%%time
print("RESULT FOR UNIGRAM WITH STOP WORDS\n")
feature_result_ug = nfeature_accuracy_checker()

csv = 'term_freq_df.csv'
term_freq_df = pd.read_csv(csv,index_col=0)
my_stop_words = frozenset(list(term_freq_df.sort_values(by='total', ascending=False).iloc[:10].index))
%%time
print("RESULT FOR UNIGRAM WITHOUT CUSTOM STOP WORDS (Top 10 frequent words)\n")
feature_result_wocsw = nfeature_accuracy_checker(stopwords=my_stop_words)
```

Show the result from above accuracy check with a graph.

```python
nfeature_plot_ug    = pd.DataFrame(feature_result_ug, columns=['nfeatures','validation_accuracy','train_test_time'])
nfeature_plot_wosw  = pd.DataFrame(feature_result_wosw, columns=['nfeatures','validation_accuracy','train_test_time'])
nfeature_plot_wocsw = pd.DataFrame(feature_result_wocsw, columns=['nfeatures','validation_accuracy','train_test_time'])

plt.figure(figsize=(10,8))
plt.plot(nfeature_plot_ug.nfeatures, nfeature_plot_ug.validation_accuracy, label='with stop words')
plt.plot(nfeature_plot_wosw.nfeatures, nfeature_plot_wosw.validation_accuracy, label='without stop words')
plt.plot(nfeature_plot_wocsw.nfeatures, nfeature_plot_wocsw.validation_accuracy, label='without custom stop words')
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()
```

By looking at the evaluation result, removing stop words did not improve the model performance, but keeping the stop words yielded better performance.

### n-gram

We also try to apply **bigram** and **trigram** and see how it affects the performance.

```python
%%time
print("RESULT FOR BIGRAM WITH STOP WORDS\n")
feature_result_bg = nfeature_accuracy_checker(ngram_range=(1, 2))

%%time
print("RESULT FOR TRIGRAM WITH STOP WORDS\n")
feature_result_tg = nfeature_accuracy_checker(ngram_range=(1, 3))
```

Show the result from above accuracy check with a graph.

```python
nfeatures_plot_tg = pd.DataFrame(feature_result_tg,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_bg = pd.DataFrame(feature_result_bg,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_ug = pd.DataFrame(feature_result_ug,columns=['nfeatures','validation_accuracy','train_test_time'])

plt.figure(figsize=(8,6))
plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy,label='trigram')
plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy,label='bigram')
plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='unigram')
plt.title("N-gram(1~3) test result : Accuracy")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()
```

Below we defined another function to take a closer look at best performing number of features with each n-gram. Below function not only reports accuracy but also gives confusion matrix and classification report.

```python
def train_test_and_evaluate(pipeline, x_train, y_train, x_test, y_test):
    if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.5:
        null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
    else:
        null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
    sentiment_fit = pipeline.fit(x_train, y_train)
    y_pred = sentiment_fit.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    conmat = np.array(confusion_matrix(y_test, y_pred, labels=[0,1]))
    confusion = pd.DataFrame(conmat, index=['negative', 'positive'],
                         columns=['predicted_negative','predicted_positive'])
    print("null accuracy: {0:.2f}%".format(null_accuracy*100))
    print("accuracy score: {0:.2f}%".format(accuracy*100))
    if accuracy > null_accuracy:
        print("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100))
    elif accuracy == null_accuracy:
        print("model has the same accuracy with the null accuracy")
    else:
        print("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100))
    print("-"*80)
    print("Confusion Matrix\n")
    print(confusion)
    print("-"*80)
    print("Classification Report\n")
    print(classification_report(y_test, y_pred, target_names=['negative','positive']))

%%time
ug_cvec = CountVectorizer(max_features=80000)
ug_pipeline = Pipeline([
        ('vectorizer', ug_cvec),
        ('classifier', logis_regres)
    ])
train_test_and_evaluate(ug_pipeline, X_train, y_train, X_vali, y_vali)

%%time
bg_cvec = CountVectorizer(max_features=70000,ngram_range=(1, 2))
bg_pipeline = Pipeline([
        ('vectorizer', bg_cvec),
        ('classifier', logis_regres)
    ])
train_test_and_evaluate(bg_pipeline, X_train, y_train, X_vali, y_vali)

%%time
tg_cvec = CountVectorizer(max_features=80000,ngram_range=(1, 3))
tg_pipeline = Pipeline([
        ('vectorizer', tg_cvec),
        ('classifier', logis_regres)
    ])
train_test_and_evaluate(tg_pipeline, X_train, y_train, X_vali, y_vali)
```

### <a name="Tfidf Vectorizer">Tfidf Vectorizer</a>

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tvec = TfidfVectorizer()

%%time
print "RESULT FOR UNIGRAM WITH STOP WORDS (Tfidf)\n"
feature_result_ugt = nfeature_accuracy_checker(vectorizer=tvec)

%%time
print "RESULT FOR BIGRAM WITH STOP WORDS (Tfidf)\n"
feature_result_bgt = nfeature_accuracy_checker(vectorizer=tvec,ngram_range=(1, 2))

%%time
print "RESULT FOR TRIGRAM WITH STOP WORDS (Tfidf)\n"
feature_result_tgt = nfeature_accuracy_checker(vectorizer=tvec,ngram_range=(1, 3))
```

It seems like TFIDF vectorizer is yielding better results when fed to logistic regression. Let's plot the results from count vectorizer together with TFIDF vectorizer.

```python
nfeatures_plot_tgt = pd.DataFrame(feature_result_tgt,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_bgt = pd.DataFrame(feature_result_bgt,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_ugt = pd.DataFrame(feature_result_ugt,columns=['nfeatures','validation_accuracy','train_test_time'])

plt.figure(figsize=(8,6))
plt.plot(nfeatures_plot_tgt.nfeatures, nfeatures_plot_tgt.validation_accuracy,label='trigram tfidf vectorizer',color='royalblue')
plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy,label='trigram count vectorizer',linestyle=':', color='royalblue')
plt.plot(nfeatures_plot_bgt.nfeatures, nfeatures_plot_bgt.validation_accuracy,label='bigram tfidf vectorizer',color='orangered')
plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy,label='bigram count vectorizer',linestyle=':',color='orangered')
plt.plot(nfeatures_plot_ugt.nfeatures, nfeatures_plot_ugt.validation_accuracy, label='unigram tfidf vectorizer',color='gold')
plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='unigram count vectorizer',linestyle=':',color='gold')
plt.title("N-gram(1~3) test result : Accuracy")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()
```

From above chart, we can see including bigram and trigram boost the model performance both in count vectorizer and TFIDF vectorizer. And for every case of unigram to trigram, TFIDF yields better results than count vectorizer.

### Algorithms Comparison

The best result I can get with logistic regression was by using TFIDF vectorizer of 100,000 features including up to trigram. With this I will first fit various different models and compare their validation results, then I will build an ensemble (voting) classifier with top 5 models.

I haven't included some of computationally expensive models, such as KNN, random forest, considering the size of data and the scalability of models. And the fine-tuning of models will come after I try some other different vectorization of textual data.

```python
names = ["Logistic Regression", "Linear SVC", "LinearSVC with L1-based feature selection","Multinomial NB",
         "Bernoulli NB", "Ridge Classifier", "AdaBoost", "Perceptron","Passive-Aggresive", "Nearest Centroid"]
classifiers = [
    LogisticRegression(),
    LinearSVC(),
    Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
  ('classification', LinearSVC(penalty="l2"))]),
    MultinomialNB(),
    BernoulliNB(),
    RidgeClassifier(),
    AdaBoostClassifier(),
    Perceptron(),
    PassiveAggressiveClassifier(),
    NearestCentroid()
    ]
zipped_clf = zip(names,classifiers)

tvec = TfidfVectorizer()
def classifier_comparator(vectorizer=tvec, n_features=10000, stop_words=None, ngram_range=(1, 1), classifier=zipped_clf):
    result = []
    vectorizer.set_params(stop_words=stop_words, max_features=n_features, ngram_range=ngram_range)
    for n,c in classifier:
        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', c)
        ])
        print "Validation result for {}".format(n)
        print c
        clf_accuracy,tt_time = accuracy_summary(checker_pipeline, x_train, y_train, x_validation, y_validation)
        result.append((n,clf_accuracy,tt_time))
    return result

%%time
trigram_result = classifier_comparator(n_features=100000,ngram_range=(1,3))

from sklearn.ensemble import VotingClassifier
clf1 = LogisticRegression()
clf2 = LinearSVC()
clf3 = MultinomialNB()
clf4 = RidgeClassifier()
clf5 = PassiveAggressiveClassifier()
eclf = VotingClassifier(estimators=[('lr', clf1), ('svc', clf2), ('mnb', clf3), ('rcs', clf4), ('pac', clf5)], voting='hard')
for clf, label in zip([clf1, clf2, clf3, clf4, clf5, eclf], ['Logistic Regression', 'Linear SVC', 'Multinomial NB', 'Ridge Classifier', 'Passive Aggresive Classifier', 'Ensemble']):
    checker_pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=100000,ngram_range=(1, 3))),
            ('classifier', clf)
        ])
    print "Validation result for {}".format(label)
    print clf
    clf_accuracy,tt_time = accuracy_summary(checker_pipeline, x_train, y_train, x_validation, y_validation)
```

It seems like the voting classifier does no better than the simple logistic regression model. Thus later part, I will try to finetune logistic regression model. But before that, I would like to try another method of sentiment classification.


### Lexical Approach

What I have demonstrated above are machine learning approaches to text classification problem, which tries to solve the problem by training classifiers on the labelled data set. Another famous approach to sentiment analysis task is a lexical approach. "In the lexical approach the definition of sentiment is based on the analysis of individual words and/or phrases; emotional dictionaries are often used: emotional lexical items from the dictionary are searched in the text, their sentiment weights are calculated, and some aggregated weight function is applied."

## <a name="Doc2Vec">Doc2Vec</a>

Below are the methods I used to get the vectors for each tweet.

1. DBOW (Distributed Bag of Words)
2. DMC (Distributed Memory Concatenated)
3. DMM (Distributed Memory Mean)
4. DBOW + DMC
5. DBOW + DMM

With the vectors we got from above models, we fit a simple logistic regression model, and evaluated the result on the validation set.

As a preparation, in addition to loading the needed dependencies, we also need to labelise each tweet with unique IDs using Gensim’s LabeledSentence function.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import utils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import multiprocessing
from tqdm import tqdm
plt.style.use("seaborn-darkgrid")
%matplotlib inline

df = pd.read_csv("clean_tweet.csv", index_col=0)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
X = df.text
y = df.target
SEED = 2000
X_train, X_vali_test, y_train, y_vali_test = train_test_split(X, y, test_size=0.02, random_state=SEED)
X_vali, X_test, y_vali, y_test = train_test_split(X_vali_test, y_vali_test, test_size=0.5, random_state=SEED)

## Label sentence
tqdm.pandas(desc="process-bar")
def labelize_tweets_ug(tweets, prefix):
    result = []
    for index, text in zip(tweets.index, tweets):
        result.append(LabeledSentence(text.split(), [prefix + "_" + str(index)]))
    return result
all_x_w2v = labelize_tweets_ug(X, "all")

## Models
cores = multiprocessing.cpu_count()
#DBOW (distributed bag of words)
model_ug_dbow = Doc2Vec(dm=0, size=100, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)

#DMC (distributed memory concatenated)
model_ug_dmc  = Doc2Vec(dm=1, dm_concat=1, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)

#DMM (distributed memory mean)
model_ug_dmm  = Doc2Vec(dm=1, dm_mean=1, size=100, negative=5, window=4, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
```

### <a name="DBOW">DBOW (Distributed bag of words)</a>

According to the developer Radim Řehůřek who created Gensim, "One caveat of the way this algorithm runs is that, since the learning rate decrease over the course of iterating over the data, labels which are only seen in a single LabeledSentence during training will only be trained with a fixed learning rate. This frequently produces less than optimal results."

Below iteration implement explicit multiple-pass, alpha-reduction approach with added shuffling. This has been already presented in Gensim's IMDB tutorial.

```python
model_ug_dbow.build_vocab([x for x in tqdm(all_x_w2v)])

%%time
for echo in range(30):
    model_ug_dbow.train(utils.shuffle([x for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_dbow.alpha -= 0.002
    model_ug_dbow.min_alpha = model_ug_dbow.alpha

def get_vectors(model, corpus, size):
    vecs = np.zeros((len(corpus), size))
    n = 0
    for index in corpus.index:
        prefix = "all_" + str(index)
        vecs[n] = model.docvecs[prefix]
        n += 1
    return vecs

train_vecs_dbow = get_vectors(model_ug_dbow, X_train, 100)
vali_vecs_dbow  = get_vectors(model_ug_dbow, X_vali, 100)

clf = LogisticRegression()
clf.fit(train_vecs_dbow, y_train)

clf.score(vali_vecs_dbow, y_vali) # 0.7370927318295739
```

### <a name="DMC">DMC (Distributed memory with concatenation)</a>

```python
model_ug_dmc.build_vocab([x for x in tqdm(all_x_w2v)])
%%time
for echo in range(30):
    model_ug_dmc.train(utils.shuffle([x for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_dmc.alpha -= 0.002
    model_ug_dmc.min_alpha = model_ug_dmc.alpha

train_vecs_dmc = get_vectors(model_ug_dmc, X_train, 100)
vali_vecs_dmc  = get_vectors(model_ug_dmc, X_vali, 100)

clf = LogisticRegression()
clf.fit(train_vecs_dmc, y_train)

clf.score(vali_vecs_dmc, y_vali) # 0.662406015037594

model_ug_dmc.most_similar("good")
model_ug_dmc.most_similar(positive=["bigger", "happy"], negative=["big"])
```

### <a name="DMM">DMM (Distributed memory with mean)</a>

```python
model_ug_dmm.build_vocab([x for x in tqdm(all_x_w2v)])
%%time
for echo in range(30):
    model_ug_dmm.train(utils.shuffle([x for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_dmm.alpha -= 0.002
    model_ug_dmm.min_alpha = model_ug_dmm.alpha

train_vecs_dmm = get_vectors(model_ug_dmm, X_train, 100)
vali_vecs_dmm  = get_vectors(model_ug_dmm, X_vali, 100)

clf = LogisticRegression()
clf.fit(train_vecs_dmm, y_train)

clf.score(vali_vecs_dmm, y_vali) # 0.7274436090225563
```

### <a name="DBOW DMC">Concatenate DBOW and DMC</a>

```python
def get_concat_vectors(model1,model2, corpus, size):
    vecs = np.zeros((len(corpus), size))
    n = 0
    for i in corpus.index:
        prefix = 'all_' + str(i)
        vecs[n] = np.append(model1.docvecs[prefix],model2.docvecs[prefix])
        n += 1
    return vecs

model_ug_dbow = Doc2Vec.load("d2v_model_ug_dbow.doc2vec")
model_ug_dmc  = Doc2Vec.load("d2v_model_ug_dmc.doc2vec")
model_ug_dmm  = Doc2Vec.load("d2v_model_ug_dmm.doc2vec")

train_vecs_dbow_dmc = get_concat_vectors(model_ug_dbow,model_ug_dmc, X_train, 200)
validation_vecs_dbow_dmc = get_concat_vectors(model_ug_dbow,model_ug_dmc, X_vali, 200)

%%time
clf = LogisticRegression()
clf.fit(train_vecs_dbow_dmc, y_train)

clf.score(validation_vecs_dbow_dmc, y_vali) # 0.7429197994987469
```

### <a name="DBOW DMM">Concatenate DBOW and DMM</a>

```python
train_vecs_dbow_dmm = get_concat_vectors(model_ug_dbow,model_ug_dmm, X_train, 200)
validation_vecs_dbow_dmm = get_concat_vectors(model_ug_dbow,model_ug_dmm, X_vali, 200)

%%time
clf = LogisticRegression()
clf.fit(train_vecs_dbow_dmm, y_train)

clf.score(validation_vecs_dbow_dmm, y_vali) # 0.7551
```

In case of unigram, we learned that concatenating document vectors in different combination boosted the model performance. The best validation accuracy we got from single model is from DBOW at 73.89%. With concatenated vectors, we get the highest validation accuracy of 75.51% with DBOW+DMM model.

### <a name="Phrase Modeling">Phrase Modeling</a>

Another thing that can be implemented with Gensim library is phrase detection. It is similar to n-gram, but instead of getting all the n-gram by sliding the window, it detects frequently used phrases and stick them together.

```python
def get_vectors(model, corpus, size):
    vecs = np.zeros((len(corpus), size))
    n = 0
    for i in corpus.index:
        prefix = 'all_' + str(i)
        vecs[n] = model.docvecs[prefix]
        n += 1
    return vecs

def get_concat_vectors(model1,model2, corpus, size):
    vecs = np.zeros((len(corpus), size))
    n = 0
    for i in corpus.index:
        prefix = 'all_' + str(i)
        vecs[n] = np.append(model1.docvecs[prefix],model2.docvecs[prefix])
        n += 1
    return vecs
```

By feeding all the tokenized tweets corpus, it will detect the frequently used phrase and connect them together with underbar in the middle.

```python
from gensim.models.phrases import Phrases
from gensim.models.phrases import Phraser

tokenized_train = [t.split() for t in x_train]
%%time
phrases = Phrases(tokenized_train)
bigram = Phraser(phrases)

sent = [u'the', u'mayor', u'of', u'new', u'york', u'was', u'there']
print(bigram[sent]) # [u'the', u'mayor', u'of', u'new_york', u'was', u'there']

## Now let's transform our corpus with this bigram model.
def labelize_tweets_bg(tweets,label):
    result = []
    prefix = label
    for i, t in zip(tweets.index, tweets):
        result.append(LabeledSentence(bigram[t.split()], [prefix + '_%s' % i]))
    return result
all_x = pd.concat([x_train,x_validation,x_test])
all_x_w2v_bg = labelize_tweets_bg(all_x, 'all')
```

After we get the corpus with bigram phrases detected, we went over the same process of Doc2Vec we did with unigram.

```python
## DBOW Bigram
cores = multiprocessing.cpu_count()
model_bg_dbow = Doc2Vec(dm=0, size=100, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_bg_dbow.build_vocab([x for x in tqdm(all_x_w2v_bg)])
%%time
for epoch in range(30):
    model_bg_dbow.train(utils.shuffle([x for x in tqdm(all_x_w2v_bg)]), total_examples=len(all_x_w2v_bg), epochs=1)
    model_bg_dbow.alpha -= 0.002
    model_bg_dbow.min_alpha = model_bg_dbow.alpha
train_vecs_dbow_bg = get_vectors(model_bg_dbow, x_train, 100)
validation_vecs_dbow_bg = get_vectors(model_bg_dbow, x_validation, 100)
%%time
clf = LogisticRegression()
clf.fit(train_vecs_dbow_bg, y_train)
clf.score(validation_vecs_dbow_bg, y_validation) # 0.73790726817042607

## DMC Bigram
cores = multiprocessing.cpu_count()
model_bg_dmc = Doc2Vec(dm=1, dm_concat=1, size=100, window=2, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_bg_dmc.build_vocab([x for x in tqdm(all_x_w2v_bg)])
%%time
for epoch in range(30):
    model_bg_dmc.train(utils.shuffle([x for x in tqdm(all_x_w2v_bg)]), total_examples=len(all_x_w2v_bg), epochs=1)
    model_bg_dmc.alpha -= 0.002
    model_bg_dmc.min_alpha = model_bg_dmc.alpha
model_bg_dmc.most_similar('new_york')
train_vecs_dmc_bg = get_vectors(model_bg_dmc, x_train, 100)
validation_vecs_dmc_bg = get_vectors(model_bg_dmc, x_validation, 100)
%%time
clf = LogisticRegression()
clf.fit(train_vecs_dmc_bg, y_train)
clf.score(validation_vecs_dmc_bg, y_validation) # 0.64974937343358397

## DMM Bigram
cores = multiprocessing.cpu_count()
model_bg_dmm = Doc2Vec(dm=1, dm_mean=1, size=100, window=4, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_bg_dmm.build_vocab([x for x in tqdm(all_x_w2v_bg)])
%%time
for epoch in range(30):
    model_bg_dmm.train(utils.shuffle([x for x in tqdm(all_x_w2v_bg)]), total_examples=len(all_x_w2v_bg), epochs=1)
    model_bg_dmm.alpha -= 0.002
    model_bg_dmm.min_alpha = model_bg_dms.alpha
train_vecs_dmm_bg = get_vectors(model_bg_dmm, x_train, 100)
validation_vecs_dmm_bg = get_vectors(model_bg_dmm, x_validation, 100)
%%time
clf = LogisticRegression()
clf.fit(train_vecs_dmm_bg, y_train)
clf.score(validation_vecs_dmm_bg, y_validation) # 0.72863408521303263

## DBOW + DMC
train_vecs_dbow_dmc_bg = get_concat_vectors(model_bg_dbow,model_bg_dmc, x_train, 200)
validation_vecs_dbow_dmc_bg = get_concat_vectors(model_bg_dbow,model_bg_dmc, x_validation, 200)
%%time
clf = LogisticRegression()
clf.fit(train_vecs_dbow_dmc_bg, y_train)
clf.score(validation_vecs_dbow_dmc_bg, y_validation) # 0.74517543859649127

## DBOW + DMM
train_vecs_dbow_dmm_bg = get_concat_vectors(model_bg_dbow,model_bg_dmm, x_train, 200)
validation_vecs_dbow_dmm_bg = get_concat_vectors(model_bg_dbow,model_bg_dmm, x_validation, 200)
%%time
clf = LogisticRegression()
clf.fit(train_vecs_dbow_dmm_bg, y_train)
clf.score(validation_vecs_dbow_dmm_bg, y_validation) # 0.75369674185463664
```

**Trigram**. And if we run the same phrase detection again on bigram detected corpus, now it will detect trigram phrases.

```python
%%time
tg_phrases = Phrases(bigram[tokenized_train])
trigram = Phraser(tg_phrases)

def labelize_tweets_tg(tweets,label):
    result = []
    prefix = label
    for i, t in zip(tweets.index, tweets):
        result.append(LabeledSentence(trigram[bigram[t.split()]], [prefix + '_%s' % i]))
    return result

all_x = pd.concat([x_train,x_validation,x_test])
all_x_w2v_tg = labelize_tweets_tg(all_x, 'all')
```

```python
## DBOW Trigram
model_tg_dbow = Doc2Vec(dm=0, size=100, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_tg_dbow.build_vocab([x for x in tqdm(all_x_w2v_tg)])
%%time
for epoch in range(30):
    model_tg_dbow.train(utils.shuffle([x for x in tqdm(all_x_w2v_tg)]), total_examples=len(all_x_w2v_tg), epochs=1)
    model_tg_dbow.alpha -= 0.002
    model_tg_dbow.min_alpha = model_tg_dbow.alpha
train_vecs_dbow_tg = get_vectors(model_tg_dbow, x_train, 100)
validation_vecs_dbow_tg = get_vectors(model_tg_dbow, x_validation, 100)
%%time
clf = LogisticRegression()
clf.fit(train_vecs_dbow_tg, y_train)
clf.score(validation_vecs_dbow_tg, y_validation) # 0.73684210526315785

## DMC Trigram
cores = multiprocessing.cpu_count()
model_tg_dmc = Doc2Vec(dm=1, dm_concat=1, size=100, window=2, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_tg_dmc.build_vocab([x for x in tqdm(all_x_w2v_tg)])
%%time
for epoch in range(30):
    model_tg_dmc.train(utils.shuffle([x for x in tqdm(all_x_w2v_tg)]), total_examples=len(all_x_w2v_tg), epochs=1)
    model_tg_dmc.alpha -= 0.002
    model_tg_dmc.min_alpha = model_tg_dmc.alpha
train_vecs_dmc_tg = get_vectors(model_tg_dmc, x_train, 100)
validation_vecs_dmc_tg = get_vectors(model_tg_dmc, x_validation, 100)
%%time
clf = LogisticRegression()
clf.fit(train_vecs_dmc_tg, y_train)
clf.score(validation_vecs_dmc_tg, y_validation) # 65507518796992481

## DMM Trigram
cores = multiprocessing.cpu_count()
model_tg_dmm = Doc2Vec(dm=1, dm_mean=1, size=100, window=4, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_tg_dmm.build_vocab([x for x in tqdm(all_x_w2v_tg)])
%%time
for epoch in range(30):
    model_tg_dmm.train(utils.shuffle([x for x in tqdm(all_x_w2v_tg)]), total_examples=len(all_x_w2v_tg), epochs=1)
    model_tg_dmm.alpha -= 0.002
    model_tg_dmc.min_alpha = model_tg_dmc.alpha
train_vecs_dmm_tg = get_vectors(model_tg_dmm, x_train, 100)
validation_vecs_dmm_tg = get_vectors(model_tg_dmm, x_validation, 100)
%%time
clf = LogisticRegression()
clf.fit(train_vecs_dmm_tg, y_train)
clf.score(validation_vecs_dmm_tg, y_validation) # 0.73840852130325818

## DBOW + DMC
train_vecs_dbow_dmc_tg = get_concat_vectors(model_tg_dbow,model_tg_dmc, x_train, 200)
validation_vecs_dbow_dmc_tg = get_concat_vectors(model_tg_dbow,model_tg_dmc, x_validation, 200)
%%time
clf = LogisticRegression()
clf.fit(train_vecs_dbow_dmc_tg, y_train)
clf.score(validation_vecs_dbow_dmc_tg, y_validation) # 0.7461152882205514

## DBOW + DMM
train_vecs_dbow_dmm_tg = get_concat_vectors(model_tg_dbow,model_tg_dmm, x_train, 200)
validation_vecs_dbow_dmm_tg = get_concat_vectors(model_tg_dbow,model_tg_dmm, x_validation, 200)
%%time
clf = LogisticRegression()
clf.fit(train_vecs_dbow_dmm_tg, y_train)
clf.score(validation_vecs_dbow_dmm_tg, y_validation) # 0.75657894736842102
```

The best validation accuracy I can get was from dbow+dmm model. DMM model tends to perform better with increased n-gram, while pure DBOW model tends to perform worse with increased n-gram. In terms of a joint model, two models performance got lower with bigram and got higher with trigram.


## <a name="Feature Selection Dimensionality reduction">FEATURE SELECTION & DIMENSIONALITY REDUCTION</a>

### <a name="Chi2 Feature Selection">Chi2 Feature Selection</a>

I will first transform the train data into Tfidf vectors of 100,000 features and see which features chi2 has chosen as useful features.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2

tvec = TfidfVectorizer(max_features=100000,ngram_range=(1, 3))
x_train_tfidf = tvec.fit_transform(x_train)
x_validation_tfidf = tvec.transform(x_validation)
chi2score = chi2(x_train_tfidf, y_train)[0]
wscores = zip(tvec.get_feature_names(), chi2score)
wchi2 = sorted(wscores, key=lambda x: x[1])
topchi2 = zip(*wchi2[-20:])
x = range(len(topchi2[1]))

plt.figure(figsize=(15,10))
wscores = zip(tvec.get_feature_names(), chi2score)
wchi2 = sorted(wscores, key=lambda x: x[1])
topchi2 = zip(*wchi2[-20:])
x = range(len(topchi2[1]))
labels = topchi2[0]
plt.barh(x,topchi2[1], align='center', alpha=0.2)
plt.plot(topchi2[1], x, '-o', markersize=5, alpha=0.8)
plt.yticks(x, labels)
plt.xlabel('$\chi^2$')

%%time
ch2_result = []
for n in np.arange(10000, 100001, 10000):
    ch2 = SelectKBest(chi2, k=n)
    X_train_chi2_selected = ch2.fit_transform(X_train_tfidf, y_train)
    X_vali_chi2_selected  = ch2.transform(X_vali_tfidf)
    clf = LogisticRegression()
    clf.fit(X_train_chi2_selected, y_train)
    score = clf.score(X_vali_chi2_.selected, y_vali)
    ch2_result.append(score)
    print "chi2 feature selection evaluation calculated for {} features".format(n)

plt.figure(figsize=(8,6))
plt.plot(nfeatures_plot_tgt.nfeatures, nfeatures_plot_tgt.validation_accuracy,label='trigram tfidf vectorizer',color='royalblue')
plt.plot(np.arange(10000,100000,10000), ch2_result,label='tfidf dimesions reduced from 100,000 features',linestyle=':', color='orangered')
plt.title("tfidft vectorizer: features limited within tfidft vectorizer VS reduced dimensions with chi2")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()
```

On the above graph, the red dotted line is validation set accuracy from dimensionality reduction, and the blue line is the result of limiting the number of features in the first place when fitting Tfidf vectorizer. We can see that limiting the number of features in the first place with Tfidf vectorizer yield better result than reducing the dimensions from bigger features. This is not a general statement, but what I have found within this particular setting.

### <a name="PCA Dimensionality Reduction">PCA Dimensionality Reduction</a>

Next, let’s try to reduce dimensions of doc2vec vectors with PCA. We can also plot the result on a graph and see if it’s feasible to reduce the number of features to a smaller set of principal components, and how much of the variance the given number of principal components can explain about the original features.

```python
def get_concat_vectors(model1,model2, corpus, size):
    vecs = np.zeros((len(corpus), size))
    n = 0
    for i in corpus.index:
        prefix = 'all_' + str(i)
        vecs[n] = np.append(model1.docvecs[prefix],model2.docvecs[prefix])
        n += 1
    return vecs

model_ug_dbow = Doc2Vec.load('d2v_model_ug_dbow.doc2vec')
model_bg_dmm = Doc2Vec.load('d2v_model_bg_dmm.doc2vec')
model_ug_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
model_bg_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

train_vecs_ugdbow_bgdmm = get_concat_vectors(model_ug_dbow,model_bg_dmm, X_train, 200)
validation_vecs_ugdbow_bgdmm = get_concat_vectors(model_ug_dbow,model_bg_dmm, X_vali, 200)
scaler = StandardScaler()
d2v_ugdbow_bgdmm_std = scaler.fit_transform(train_vecs_ugdbow_bgdmm)
d2v_ugdbow_bgdmm_std_val = scaler.fit_transform(validation_vecs_ugdbow_bgdmm)

d2v_pca = PCA().fit(d2v_ugdbow_bgdmm_std)

fig, ax = plt.subplots(figsize=(8,6))
x_values = range(1, d2v_pca.n_components_+1)
ax.plot(x_values, d2v_pca.explained_variance_ratio_, lw=2, label='explained variance')
ax.plot(x_values, np.cumsum(d2v_pca.explained_variance_ratio_), lw=2, label='cumulative explained variance')
ax.set_title('Doc2vec (unigram DBOW + trigram DMM) : explained variance of components')
ax.set_xlabel('principal component')
ax.set_ylabel('explained variance')
plt.show()
```

In the above graph, the red line represents cumulative explained variance and the blue line represents explained the variance of each principal component. By looking at the graph above, even though the red line is not perfectly linear, but very close to a straight line. Is this good? No. This means each of the principal components contributes to the variance explanation almost equally, and there’s not much point in reducing the dimensions based on PCA. This can also be seen from the blue line, which is very close to a straight line in the bottom.


## <a name="NEURAL NETWORKS WITH TFIDF VECTORS">NEURAL NETWORKS WITH TFIDF VECTORS</a>

My first idea was, if logistic regression is the best performing classifier, then this idea can be extended to neural networks. In terms of its structure, logistic regression can be thought as a neural network with no hidden layer, and just one output node.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
plt.style.use("seaborn-darkgrid")
%matplotlib inline

csv = 'clean_tweet.csv'
my_df = pd.read_csv(csv,index_col=0)
my_df.dropna(inplace=True)
my_df.reset_index(drop=True,inplace=True)
x = my_df.text
y = my_df.target
SEED = 2000
x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)
x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)
```

The best performing Tfidf vectors I got is with 100,000 features including up to trigram with logistic regression. Validation accuracy is 82.91%, while train set accuracy is 84.19%. I would want to see if a neural network can boost the performance of my existing Tf-Idf vectors.

I will first start by loading required dependencies. In order to run Keras with TensorFlow backend, you need to install both TensorFlow and Keras.

```python
seed = 7
np.random.seed(seed)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
```

Before I feed the data and train the model, I need to deal with one more thing. Keras NN model cannot handle sparse matrix directly. The data has to be dense array or matrix, but transforming the whole training data Tfidf vectors of 1.5 million to dense array won't fit into my RAM. So I had to define a function, which generates iterable generator object so that it can be fed to NN model. Note that the output should be a generator class object rather than arrays, this can be achieved by using "yield" instead of "return".

```python
def batch_generator(X_data, y_data, batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    index = np.arange(np.shape(y_data)[0])
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_data[index_batch,:].toarray()
        y_batch = y_data[y_data.index[index_batch]]
        counter += 1
        yield X_batch,y_batch
        if (counter > number_of_batches):
            counter=0
```

```python
%%time
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=100000))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit_generator(generator=batch_generator(x_train_tfidf, y_train, 32),
                    epochs=5, validation_data=(x_validation_tfidf, y_validation),
                    steps_per_epoch=x_train_tfidf.shape[0]/32)
```

It looks like the model had the best validation accuracy after 2 epochs, and after that, it fails to generalize so validation accuracy slowly decreases, while training accuracy increases. But if you remember the result I got from logistic regression (train accuracy: 84.19%, validation accuracy: 82.91%), you can see that the above neural network failed to outperform logistic regression in terms of validation.

Let's see if normalizing inputs have any effect on the performance.

```python
from sklearn.preprocessing import Normalizer
norm = Normalizer().fit(x_train_tfidf)
x_train_tfidf_norm = norm.transform(x_train_tfidf)
x_validation_tfidf_norm = norm.transform(x_validation_tfidf)
```

```python
%%time
model_n = Sequential()
model_n.add(Dense(64, activation='relu', input_dim=100000))
model_n.add(Dense(1, activation='sigmoid'))
model_n.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_n.fit_generator(generator=batch_generator(x_train_tfidf_norm, y_train, 32),
                    epochs=5, validation_data=(x_validation_tfidf_norm, y_validation),
                    steps_per_epoch=x_train_tfidf_norm.shape[0]/32)
```

By the look of the result, normalizing seems to have almost no effect on the performance. And it is at this point I realized that Tfidf is already normalized by the way it is calculated.

If the problem of the model is a poor generalization, then there is another thing I can add to the model. Even though the neural network is a very powerful model, sometimes overfitting to the training data can be a problem. Dropout is a technique that addresses this problem.

```python
model1 = Sequential()
model1.add(Dense(64, activation='relu', input_dim=100000))
model1.add(Dropout(0.2))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model1.fit_generator(generator=batch_generator(x_train_tfidf, y_train, 32),
                    epochs=5, validation_data=(x_validation_tfidf, y_validation),
                    steps_per_epoch=x_train_tfidf.shape[0]/32)
```

Through 5 epochs, the train set accuracy didn't get as high as the model without dropout, but validation accuracy didn't drop as low as the previous model. Even though the dropout added some generalization to the model, but the validation accuracy is still underperforming compared to logistic regression result.

There is another method I can try to prevent overfitting. By presenting the data in the same order for every epoch, there's a possibility that the model learns the parameters which also includes the noise of the training data, which eventually leads to overfitting. This can be improved by shuffling the order of the data we feed the model. Below I added shuffling to the batch generator function and tried with the same model structure and compared the result.

```python
def batch_generator_shuffle(X_data, y_data, batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    index = np.arange(np.shape(y_data)[0])
    np.random.shuffle(index)
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_data[index_batch,:].toarray()
        y_batch = y_data[y_data.index[index_batch]]
        counter += 1
        yield X_batch,y_batch
        if (counter > number_of_batches):
            np.random.shuffle(index)
            counter=0
```

```python
%%time
model_s = Sequential()
model_s.add(Dense(64, activation='relu', input_dim=100000))
model_s.add(Dense(1, activation='sigmoid'))
model_s.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_s.fit_generator(generator=batch_generator_shuffle(x_train_tfidf, y_train, 32),
                    epochs=5, validation_data=(x_validation_tfidf, y_validation),
                    steps_per_epoch=x_train_tfidf.shape[0]/32)
```

The same model with non-shuffled training data had training accuracy of 87.36%, and validation accuracy of 79.78%. With shuffling, training accuracy decreased to 84.80% but the validation accuracy after 5 epochs has increased to 82.61%. It seems like the shuffling did improve the model's performance on the validation set. And another thing I noticed is that with or without shuffling also for both with or without dropout, validation accuracy tends to peak after 2 epochs, and gradually decrease afterwards.

As I was going through the "deeplearning.ai" course by Andrew Ng, he states that the first thing he would try to improve a neural network model is tweaking the learning rate. I decided to follow his advice and try different learning rates with the model. Please note that except for the learning rate, the parameter for 'beta_1', 'beta_2', and 'epsilon' are set to the default values presented by the original paper.

```python
%%time
import keras
custom_adam = keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model_testing_2 = Sequential()
model_testing_2.add(Dense(64, activation='relu', input_dim=100000))
model_testing_2.add(Dense(1, activation='sigmoid'))
model_testing_2.compile(optimizer=custom_adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_testing_2.fit_generator(generator=batch_generator_shuffle(x_train_tfidf, y_train, 32),
                    epochs=2, validation_data=(x_validation_tfidf, y_validation),
                    steps_per_epoch=x_train_tfidf.shape[0]/32)
```

Having tried four different learning rates (0.0005, 0.005, 0.01, 0.1), none of them outperformed the default learning rate of 0.001.

Maybe I can try to increase the number of hidden nodes, and see how it affects the performance. Below model has 128 nodes in the hidden layer.

```python
%%time
model_s_2 = Sequential()
model_s_2.add(Dense(128, activation='relu', input_dim=100000))
model_s_2.add(Dense(1, activation='sigmoid'))
model_s_2.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_s_2.fit_generator(generator=batch_generator_shuffle(x_train_tfidf, y_train, 32),
                    epochs=2, validation_data=(x_validation_tfidf, y_validation),
                    steps_per_epoch=x_train_tfidf.shape[0]/32)
```

With 128 hidden nodes, validation accuracy got close to the performance of logistic regression. I could experiment further with increasing the number of hidden layers, but for the above 2 epochs to run, it took 5 hours. Considering that logistic regression took less than a minute to fit, even if the neural network can be improved further, this doesn't look like an efficient way.

As a result, in this particular case, neural network models failed to outperform logistic regression. This might be due to the high dimensionality and sparse characteristics of the textual data. I have also found a research paper, which compared model performance with high dimension data. According to "An Empirical Evaluation of Supervised Learning in High Dimensions" by Caruana et al.(2008), logistic regression showed as good performance as neural networks, in some cases outperforms neural networks. http://icml2008.cs.helsinki.fi/papers/632.pdf

Through all the trials above I learned some valuable lessons. Implementing and tuning neural networks is a highly iterative process and includes many trials and errors. Even though the neural network is a more complex version of logistic regression, it doesn't always outperform logistic regression, and sometimes with high dimension sparse data, logistic regression can deliver good performance with much less computation time than neural network.

In the next post, I will implement a neural network with Doc2Vec vectors I got from the previous post. Hopefully, with dense vectors such as Doc2Vec, a neural network might show some boost. Fingers crossed.

## <a name="NEURAL NETWORKS WITH Doc2Vec and Word2Vec">NEURAL NETWORKS WITH Doc2Vec and Word2Vec</a>

Before I jump into neural network modelling with the vectors I got from Doc2Vec, I would like to give you some background on how I got these document vectors. I have implemented Doc2Vec using Gensim library in the 6th part of this series.

There are three different methods used to train Doc2Vec. Distributed Bag of Words, Distributed Memory (Mean), Distributed Memory (Concatenation). These models were trained with 1.5 million tweets through 30 epochs and the output of the models are 100 dimension vectors for each tweet. After I got document vectors from each model, I have tried concatenating these (so the concatenated document vectors have 200 dimensions) in combination: DBOW + DMM, DBOW + DMC, and saw an improvement to the performance when compared with models with one pure method. Using different methods of training and concatenating them to improve the performance has already been demonstrated by Le and Mikolov (2014) in their research paper. https://cs.stanford.edu/~quocle/paragraph_vector.pdf

Finally, I have applied phrase modelling to detect bigram phrase and trigram phrase as a pre-step of Doc2Vec training and tried different combination across n-grams. When tested with a logistic regression model, I got the best performance result from 'unigram DBOW + trigram DMM' document vectors.

### <a name="Doc2Vec">Doc2Vec</a>

I will first start by loading Gensim's Doc2Vec, and define a function to extract document vectors, then load the doc2vec model I trained.

```python
from gensim.models import Doc2Vec

def get_concat_vectors(model1,model2, corpus, size):
    vecs = np.zeros((len(corpus), size))
    n = 0
    for i in corpus.index:
        prefix = 'all_' + str(i)
        vecs[n] = np.append(model1.docvecs[prefix],model2.docvecs[prefix])
        n += 1
    return vecs

model_ug_dbow = Doc2Vec.load('d2v_model_ug_dbow.doc2vec')
model_tg_dmm = Doc2Vec.load('d2v_model_tg_dmm.doc2vec')
model_ug_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
model_tg_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
train_vecs_ugdbow_tgdmm = get_concat_vectors(model_ug_dbow,model_tg_dmm, x_train, 200)
validation_vecs_ugdbow_tgdmm = get_concat_vectors(model_ug_dbow,model_tg_dmm, x_validation, 200)

%%time
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(train_vecs_ugdbow_tgdmm, y_train)
%%time
clf.score(train_vecs_ugdbow_tgdmm, y_train)
%%time
clf.score(validation_vecs_ugdbow_tgdmm, y_validation)
```

When fed to a simple logistic regression, the concatenated document vectors (unigram DBOW + trigram DMM) yields 75.90% training set accuracy, and 75.76% validation set accuracy.

I will try different numbers of hidden layers, hidden nodes to compare the performance. In the below code block, you see I first define the seed as "7" but not setting the random seed, "np.random.seed()" will be defined at the start of each model. This is for a reproducibility of various results from different model structures.

*Side Note (reproducibility)*: To be honest, this took me a while to figure out. I first tried by setting the random seed before I import Keras, and ran one model after another. However, if I define the same model structure after it has run, I couldn't get the same result. But I also realised if I restart the kernel, and re-run code blocks from start it gives me the same result as the last kernel. So I figured, after running a model the random seed changes, and that is the reason why I cannot get the same result with the same structure if I run them in the same kernel consecutively. Anyway, that is why I set the random seed every time I try a different model. For your information, I am running Keras with Theano backend, and only using CPU not GPU. If you are on the same setting, this should work. I explicitly specified backend as Theano by launching Jupyter Notebook in the command line as follows: "KERAS_BACKEND=theano jupyter notebook"

Please note that not all of the dependencies loaded in the below cell has been used for this post, but imported for later use.

```python
seed = 7

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

%%time
np.random.seed(seed)
model_d2v_01 = Sequential()
model_d2v_01.add(Dense(64, activation='relu', input_dim=200))
model_d2v_01.add(Dense(1, activation='sigmoid'))
model_d2v_01.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_d2v_01.fit(train_vecs_ugdbow_tgdmm, y_train, validation_data=(validation_vecs_ugdbow_tgdmm, y_validation), epochs=10, batch_size=32, verbose=2)
```

After trying 12 different models with a range of hidden layers (from 1 to 3) and a range of hidden nodes for each hidden layer (64, 128, 256, 512), the best validation accuracy (79.93%) is from "model_d2v_09" at epoch 7, which has 3 hidden layers of 256 hidden nodes for each hidden layer.

Now I know which model gives me the best result, I will run the final model of "model_d2v_09", but this time with callback functions in Keras. I was not quite familiar with callback functions in Keras before I received a comment in my previous post. After I got the comment, I did some digging and found all the useful functions in Keras callbacks. With my final model of Doc2Vec below, I used "checkpoint" and "earlystop". You can set the "checkpoint" function with options, and with the below parameter setting, "checkpoint" will save the best performing model up until the point of running, and only if a new epoch outperforms the saved model it will save it as a new model. And "early_stop" I defined it as to monitor validation accuracy, and if it doesn't outperform the best validation accuracy so far for 5 epochs, it will stop.

```python
from keras.callbacks import ModelCheckpoint, EarlyStopping

filepath="d2v_09_best_weights.{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_acc', patience=5, mode='max')
callbacks_list = [checkpoint, early_stop]
np.random.seed(seed)
model_d2v_09_es = Sequential()
model_d2v_09_es.add(Dense(256, activation='relu', input_dim=200))
model_d2v_09_es.add(Dense(256, activation='relu'))
model_d2v_09_es.add(Dense(256, activation='relu'))
model_d2v_09_es.add(Dense(1, activation='sigmoid'))
model_d2v_09_es.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_d2v_09_es.fit(train_vecs_ugdbow_tgdmm, y_train, validation_data=(validation_vecs_ugdbow_tgdmm, y_validation), epochs=100, batch_size=32, verbose=2, callbacks=callbacks_list)
```

If I evaluate the model I just run, it will give me the result as same as I got from the last epoch.

```python
model_d2v_09_es.evaluate(x=validation_vecs_ugdbow_tgdmm, y=y_validation)
```

But if I load the saved model at the best epoch, then this model will give me the result at that epoch.

```python
from keras.models import load_model
loaded_model = load_model('d2v_09_best_weights.07-0.7993.hdf5')
loaded_model.evaluate(x=validation_vecs_ugdbow_tgdmm, y=y_validation)
```

If you remember the validation accuracy with the same vector representation of the tweets with a logistic regression model (75.76%), you can see that feeding the same information to neural networks yields a significantly better result. It's amazing to see how neural network can boost the performance of dense vectors, but the best validation accuracy is still lower than the Tfidf vectors + logistic regression model, which gave me 82.92% validation accuracy.

If you have read my posts on Doc2Vec, or familiar with Doc2Vec, you might know that you can also extract word vectors for each word from the trained Doc2Vec model. I will move on to Word2Vec, and try different methods to see if any of those can outperform the Doc2Vec result (79.93%), ultimately outperform the Tfidf + logistic regression model (82.92%).

### <a name="Word2Vec">Word2Vec</a>

To make use of word vectors extracted from Doc2Vec model, I can no longer use the concatenated vectors of different n-grams, since they will not consist of the same vocabularies. Thus below, I load the model for unigram DMM and create concatenated vectors with unigram DBOW of 200 dimensions for each word in the vocabularies.

What I will do first before I try neural networks with document representations computed from word vectors is that I will fit a logistic regression with various methods of document representation and with the one that gives me the best validation accuracy, I will finally define neural network models.

I will also give you the summary of result from all the different word vectors fit with logistic regression as a table.

#### Word vectors extracted from Doc2Vec models (Average/Sum)

There could be a number of different ways to come up with document representational vectors with individual word vectors. One obvious choice is to average them. For every word in a tweet, see if trained Doc2Vec has word vector representation of the word, if so, sum them up throughout the document while counting how many words were detected as having word vectors, and finally by dividing the summed vector by the count you get the averaged word vector for the whole document which will have the same dimension (200 in this case) as the individual word vectors.

Another method is just the sum of the word vectors without averaging them. This might distort the vector representation of the document if some tweets only have a few words in the Doc2Vec vocabulary and some tweets have most of the words in the Doc2Vec vocabulary. But I will try both summing and averaging and compare the results.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale

model_ug_dmm = Doc2Vec.load('d2v_model_ug_dmm.doc2vec')
model_ug_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

def get_w2v_ugdbowdmm(tweet, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tweet.split():
        try:
            vec += np.append(model_ug_dbow[word],model_ug_dmm[word]).reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

def get_w2v_ugdbowdmm_sum(tweet, size):
    vec = np.zeros(size).reshape((1, size))
    for word in tweet.split():
        try:
            vec += np.append(model_ug_dbow[word],model_ug_dmm[word]).reshape((1, size))
        except KeyError:
            continue
    return vec

train_vecs_w2v_dbowdmm = np.concatenate([get_w2v_ugdbowdmm(z, 200) for z in x_train])
validation_vecs_w2v_dbowdmm = np.concatenate([get_w2v_ugdbowdmm(z, 200) for z in x_validation])
%%time
clf = LogisticRegression()
clf.fit(train_vecs_w2v_dbowdmm, y_train)
clf.score(validation_vecs_w2v_dbowdmm, y_validation)  #0.7173558897243107
```

The validation accuracy with averaged word vectors of unigram DBOW + unigram DMM is 71.74%, which is significantly lower than document vectors extracted from unigram DBOW + trigram DMM (75.76%), and also from the results I got from the 6th part of this series, I know that document vectors extracted from unigram DBOW + unigram DMM will give me 75.51% validation accuracy.

I also tried scaling the vectors using ScikitLearn's scale function, and saw significant improvement in computation time and a slight improvement of the accuracy. And let's also see how summed word vectors perform compared to the averaged counter part.

```python
train_vecs_w2v_dbowdmm_s = scale(train_vecs_w2v_dbowdmm)
validation_vecs_w2v_dbowdmm_s = scale(validation_vecs_w2v_dbowdmm)
%%time
clf = LogisticRegression()
clf.fit(train_vecs_w2v_dbowdmm_s, y_train)
clf.score(validation_vecs_w2v_dbowdmm_s, y_validation)  #0.7241854636591478

train_vecs_w2v_dbowdmm_sum = np.concatenate([get_w2v_ugdbowdmm_sum(z, 200) for z in x_train])
validation_vecs_w2v_dbowdmm_sum = np.concatenate([get_w2v_ugdbowdmm_sum(z, 200) for z in x_validation])
%%time
clf = LogisticRegression()
clf.fit(train_vecs_w2v_dbowdmm_sum, y_train)
clf.score(validation_vecs_w2v_dbowdmm_sum, y_validation)  #0.7251253132832081

#The summation method gave me higher accuracy without scaling compared to the average method. But the simple logistic regression with the summed vectors took more than 3 hours to run. So again I tried scaling these vectors.
train_vecs_w2v_dbowdmm_sum_s = scale(train_vecs_w2v_dbowdmm_sum)
validation_vecs_w2v_dbowdmm_sum_s = scale(validation_vecs_w2v_dbowdmm_sum)
%%time
clf = LogisticRegression()
clf.fit(train_vecs_w2v_dbowdmm_sum_s, y_train)
clf.score(validation_vecs_w2v_dbowdmm_sum_s, y_validation)  #0.725250626566416
```

Surprising! With scaling, logistic regression fitting only took 3 minutes! That's quite a difference.

#### Word vectors extracted from Doc2Vec models with TFIDF weighting (Average/Sum)

In the 5th part of this series, I have already explained what TF-IDF is. TF-IDF is a way of weighting each word by calculating the product of relative term frequency and inverse document frequency. Since it gives one scalar value for each word in the vocabulary, this can also be used as a weighting factor of each word vectors. Correa Jr. et al (2017) has implemented this Tf-idf weighting in their paper "NILC-USP at SemEval-2017 Task 4: A Multi-view Ensemble for Twitter Sentiment Analysis" http://www.aclweb.org/anthology/S17-2100

In order to get the Tfidf value for each word, I first fit and transform the training set with TfidfVectorizer and create a dictionary containing "word", "tfidf value" pairs.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tvec = TfidfVectorizer(min_df=2)
tvec.fit_transform(x_train)
tfidf = dict(zip(tvec.get_feature_names(), tvec.idf_))
print 'vocab size :', len(tfidf)  # vocab size : 103691

def get_w2v_general(tweet, size, vectors, aggregation='mean'):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tweet.split():
        try:
            vec += vectors[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if aggregation == 'mean':
        if count != 0:
            vec /= count
        return vec
    elif aggregation == 'sum':
        return vec
```

The below code can also be implemented within the word vector averaging or summing function, but it seems like it's taking quite a long time, so I separated this and tried to make a dictionary of word vectors weighted by Tfidf values. To be honest, I am still not sure why it took so long to compute the Tfidf weighting of the word vectors, but after 5 hours it finally finished computing. You can also see later that I tried another method of weighting but that took less than 10 seconds. If you have an answer to this, any insight would be appreciated.

```python
%%time
w2v_tfidf = {}
for w in model_ug_dbow.wv.vocab.keys():
    if w in tvec.get_feature_names():
        w2v_tfidf[w] = np.append(model_ug_dbow[w],model_ug_dmm[w]) * tfidf[w]
# CPU times: user 4h 53min 1s, sys: 6min 1s, total: 4h 59min 2s
# Wall time: 4h 58min 17s

import cPickle as pickle
with open('w2v_tfidf.p', 'wb') as fp:
    pickle.dump(w2v_tfidf, fp, protocol=pickle.HIGHEST_PROTOCOL)


import cPickle as pickle
with open('w2v_tfidf.p', 'rb') as fp:
    w2v_tfidf = pickle.load(fp)

%%time
train_vecs_w2v_tfidf_mean = scale(np.concatenate([get_w2v_general(z, 200, w2v_tfidf, 'mean') for z in x_train]))
validation_vecs_w2v_tfidf_mean = scale(np.concatenate([get_w2v_general(z, 200, w2v_tfidf, 'mean') for z in x_validation]))
%%time
clf = LogisticRegression()
clf.fit(train_vecs_w2v_tfidf_mean, y_train)
clf.score(validation_vecs_w2v_tfidf_mean, y_validation)  #0.7057017543859649

%%time
train_vecs_w2v_tfidf_sum = scale(np.concatenate([get_w2v_general(z, 200, w2v_tfidf, 'sum') for z in x_train]))
validation_vecs_w2v_tfidf_sum = scale(np.concatenate([get_w2v_general(z, 200, w2v_tfidf, 'sum') for z in x_validation]))
%%time
clf = LogisticRegression()
clf.fit(train_vecs_w2v_tfidf_sum, y_train)
clf.score(validation_vecs_w2v_tfidf_sum, y_validation)  #0.7031954887218045
```

The result is not what I expected, especially after 5 hours of waiting. By weighting word vectors with Tfidf values, the validation accuracy dropped around 2% both for averaging and summing.

#### Word vectors extracted from Doc2Vec models with custom weighting (Average/Sum)

In the 3rd part of this series, I have defined a custom metric called "pos_normcdf_hmean", which is a metric borrowed from the presentation by Jason Kessler in PyData 2017 Seattle. If you want to know more in detail about the calculation, you can either check my previous post or you can also watch Jason Kessler's presentation. To give you a high-level intuition, by calculating harmonic mean of CDF(Cumulative Distribution Function) transformed values of term frequency rate within the whole document and the term frequency within a class, you can get a meaningful metric which shows how each word is related to a certain class.

I have used this metric to visualise tokens in the 3rd part of the series, and also used this again to create custom lexicon to be used for classification purpose in the 5th part. I will use this again as a weighting factor for the word vectors, and see how it affects the performance.

```python
from sklearn.feature_extraction.text import CountVectorizer
cvec = CountVectorizer(max_features=100000)
cvec.fit(x_train)

neg_train = x_train[y_train == 0]
pos_train = x_train[y_train == 1]
neg_doc_matrix = cvec.transform(neg_train)
pos_doc_matrix = cvec.transform(pos_train)
neg_tf = np.sum(neg_doc_matrix,axis=0)
pos_tf = np.sum(pos_doc_matrix,axis=0)

from scipy.stats import hmean
from scipy.stats import norm
def normcdf(x):
    return norm.cdf(x, x.mean(), x.std())

neg = np.squeeze(np.asarray(neg_tf))
pos = np.squeeze(np.asarray(pos_tf))
term_freq_df2 = pd.DataFrame([neg,pos],columns=cvec.get_feature_names()).transpose()
term_freq_df2.columns = ['negative', 'positive']
term_freq_df2['total'] = term_freq_df2['negative'] + term_freq_df2['positive']
term_freq_df2['pos_rate'] = term_freq_df2['positive'] * 1./term_freq_df2['total']
term_freq_df2['pos_freq_pct'] = term_freq_df2['positive'] * 1./term_freq_df2['positive'].sum()
term_freq_df2['pos_rate_normcdf'] = normcdf(term_freq_df2['pos_rate'])
term_freq_df2['pos_freq_pct_normcdf'] = normcdf(term_freq_df2['pos_freq_pct'])
term_freq_df2['pos_normcdf_hmean'] = hmean([term_freq_df2['pos_rate_normcdf'], term_freq_df2['pos_freq_pct_normcdf']])
term_freq_df2.sort_values(by='pos_normcdf_hmean', ascending=False).iloc[:10]

pos_hmean = term_freq_df2.pos_normcdf_hmean
%%time
w2v_pos_hmean = {}
for w in model_ug_dbow.wv.vocab.keys():
    if w in pos_hmean.keys():
        w2v_pos_hmean[w] = np.append(model_ug_dbow[w],model_ug_dmm[w]) * pos_hmean[w]
# CPU times: user 4.81 s, sys: 1.93 s, total: 6.75 s
# Wall time: 9.51 s

with open('w2v_hmean.p', 'wb') as fp:
    pickle.dump(w2v_pos_hmean, fp, protocol=pickle.HIGHEST_PROTOCOL)
with open('w2v_hmean.p', 'rb') as fp:
    w2v_pos_hmean = pickle.load(fp)

train_vecs_w2v_poshmean_mean = scale(np.concatenate([get_w2v_general(z, 200, w2v_pos_hmean, 'mean') for z in x_train]))
validation_vecs_w2v_poshmean_mean = scale(np.concatenate([get_w2v_general(z, 200, w2v_pos_hmean, 'mean') for z in x_validation]))
%%time
clf = LogisticRegression()
clf.fit(train_vecs_w2v_poshmean_mean, y_train)
clf.score(validation_vecs_w2v_poshmean_mean, y_validation)  #0.7327067669172932

train_vecs_w2v_poshmean_sum = scale(np.concatenate([get_w2v_general(z, 200, w2v_pos_hmean, 'sum') for z in x_train]))
validation_vecs_w2v_poshmean_sum = scale(np.concatenate([get_w2v_general(z, 200, w2v_pos_hmean, 'sum') for z in x_validation]))
%%time
clf = LogisticRegression()
clf.fit(train_vecs_w2v_poshmean_sum, y_train)
clf.score(validation_vecs_w2v_poshmean_sum, y_validation)  #0.7093984962406015
```

Unlike Tfidf weighting, this time with custom weighting it actually gave me some performance boost when used with averaging method. But with summing, this weighting has performed no better than the word vectors without weighting.

#### Word vectors extracted from pre-trained GloVe (Average/Sum)

GloVe is another kind of word representation in vectors proposed by Pennington et al. (2014) from the Stanford NLP Group. https://nlp.stanford.edu/pubs/glove.pdf

The difference between Word2Vec and Glove is how the two models compute the word vectors. In Word2Vec, the word vectors you are getting is a kind of a by-product of a shallow neural network, when it tries to predict either centre word given surrounding words or vice versa. But with GloVe, the word vectors you are getting is the object matrix of GloVe model, and it calculates this using term co-occurrence matrix and dimensionality reduction.

The good news is you can now easily load and use the pre-trained GloVe vectors from Gensim thanks to its latest update (Gensim 3.2.0). In addition to some pre-trained word vectors, new datasets are also added and this also can be easily downloaded using their downloader API. If you want to know more about this, please check this blog post by RaRe Technologies. https://rare-technologies.com/new-download-api-for-pretrained-nlp-models-and-datasets-in-gensim/

The Stanford NLP Group has made their pre-trained GloVe vectors publicly available, and among them there are GloVe vectors trained specifically with Tweets. This sounds like something definitely worth trying. They have four different versions of Tweet vectors each with different dimensions (25, 50, 100, 200) trained on 2 billion Tweets. You can find more detail in their website. https://nlp.stanford.edu/projects/glove/

For this post, I will use 200 dimension pre-trained GloVe vectors.

```python
import gensim.downloader as api
glove_twitter = api.load("glove-twitter-200")
train_vecs_glove_mean = scale(np.concatenate([get_w2v_general(z, 200, glove_twitter,'mean') for z in x_train]))
validation_vecs_glove_mean = scale(np.concatenate([get_w2v_general(z, 200, glove_twitter,'mean') for z in x_validation]))
%%time
clf = LogisticRegression()
clf.fit(train_vecs_glove_mean, y_train)
clf.score(validation_vecs_glove_mean, y_validation)  #0.76265664160401

train_vecs_glove_sum = scale(np.concatenate([get_w2v_general(z, 200, glove_twitter,'sum') for z in x_train]))
validation_vecs_glove_sum = scale(np.concatenate([get_w2v_general(z, 200, glove_twitter,'sum') for z in x_validation]))
%%time
clf = LogisticRegression()
clf.fit(train_vecs_glove_sum, y_train)
clf.score(validation_vecs_glove_sum, y_validation)  #0.7659774436090225
```

By using pre-trained GloVe vectors, I can see that the validation accuracy significantly improved. So far the best validation accuracy was from the averaged word vectors with custom weighting, which gave me 73.27% accuracy, and compared to this, GloVe vectors yields 76.27%, 76.60% for average and sum respectively.

#### Word vectors extracted from pre-trained Google News Word2Vec (Average/Sum)

With new updated Gensim, I can also load the famous pre-trained Google News word vectors. These word vectors are trained using Word2Vec model on Google News dataset (about 100 billion words) and published by Google. The model contains 300-dimensional vectors for 3 million words and phrases. You can find more detail in the Google project archive. https://code.google.com/archive/p/word2vec/

```python
import gensim.downloader as api
googlenews = api.load("word2vec-google-news-300")

train_vecs_googlenews_mean = scale(np.concatenate([get_w2v_general(z, 300, googlenews,'mean') for z in x_train]))
validation_vecs_googlenews_mean = scale(np.concatenate([get_w2v_general(z, 300, googlenews,'mean') for z in x_validation]))
%%time
clf = LogisticRegression()
clf.fit(train_vecs_googlenews_mean, y_train)
clf.score(validation_vecs_googlenews_mean, y_validation)  #0.749561403508772

train_vecs_googlenews_sum = scale(np.concatenate([get_w2v_general(z, 300, googlenews,'sum') for z in x_train]))
validation_vecs_googlenews_sum = scale(np.concatenate([get_w2v_general(z, 300, googlenews,'sum') for z in x_validation]))
%%time
clf = LogisticRegression()
clf.fit(train_vecs_googlenews_sum, y_train)
clf.score(validation_vecs_googlenews_sum, y_validation)  #0.7491854636591478
```

Even though it gives me a better result than the word vectors extracted from custom trained Doc2Vec models, but it fails to outperform GloVe vectors. And the vector dimension is even larger in Google News word vectors.

But, this is trained with Google News, and GloVe vector I used was trained specifically with Tweets, thus it is hard to comapre each other directly. What if Word2Vec is specifically trained with Tweets?

#### Separately trained Word2Vec (Average/Sum)

I know I have already tried word vectors I extracted from Doc2Vec models, but what if I train separate Word2Vec models? Even though Doc2Vec models gave good representational vectors of document level, would it be more efficently learning word vectors if I train pure Word2Vec?

In order to answer my own questions, I trained two Word2Vec models using CBOW (Continuous Bag Of Words) and Skip Gram models. In terms of parameter setting, I set the same parameters I used for Doc2Vec:

size of vectors: 100 dimensions; negative sampling: 5; window: 2; minimum word count: 2; alpha: 0.065 (decrease alpha by 0.002 per epoch); number of epochs: 30.

With above settings, I defined CBOW model by passing "sg=0", and Skip Gram model by passing "sg=1". And once I get the results from two models, I concatenate vectors of two models for each word so that the concatenated vectors will have 200 dimensional representation of each word.

Please note that in the 6th part, where I trained Doc2Vec, I used "LabeledSentence" function imported from Gensim. This has now been deprecated, thus for this post I used "TaggedDocument" function instead. The usage is the same.

```python
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from sklearn import utils

def labelize_tweets_ug(tweets,label):
    result = []
    prefix = label
    for i, t in zip(tweets.index, tweets):
        result.append(TaggedDocument(t.split(), [prefix + '_%s' % i]))
    return result
all_x = pd.concat([x_train,x_validation,x_test])
all_x_w2v = labelize_tweets_ug(all_x, 'all')

cores = multiprocessing.cpu_count()
model_ug_cbow = Word2Vec(sg=0, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_cbow.build_vocab([x.words for x in tqdm(all_x_w2v)])
%%time
for epoch in range(30):
    model_ug_cbow.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_cbow.alpha -= 0.002
    model_ug_cbow.min_alpha = model_ug_cbow.alpha

train_vecs_cbow_mean = scale(np.concatenate([get_w2v_general(z, 100, model_ug_cbow,'mean') for z in x_train]))
validation_vecs_cbow_mean = scale(np.concatenate([get_w2v_general(z, 100, model_ug_cbow,'mean') for z in x_validation]))
%%time
clf = LogisticRegression()
clf.fit(train_vecs_cbow_mean, y_train)
clf.score(validation_vecs_cbow_mean, y_validation)  #0.7600250626566416

model_ug_sg = Word2Vec(sg=1, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_sg.build_vocab([x.words for x in tqdm(all_x_w2v)])
%%time
for epoch in range(30):
    model_ug_sg.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_sg.alpha -= 0.002
    model_ug_sg.min_alpha = model_ug_sg.alpha

train_vecs_sg_mean = scale(np.concatenate([get_w2v_general(z, 100, model_ug_sg,'mean') for z in x_train]))
validation_vecs_sg_mean = scale(np.concatenate([get_w2v_general(z, 100, model_ug_sg,'mean') for z in x_validation]))
%%time
clf = LogisticRegression()
clf.fit(train_vecs_sg_mean, y_train)
clf.score(validation_vecs_sg_mean, y_validation)  #0.7604010025062656
```

```python
def get_w2v_mean(tweet, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tweet.split():
        try:
            vec += np.append(model_ug_cbow[word],model_ug_sg[word]).reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

train_vecs_cbowsg_mean = scale(np.concatenate([get_w2v_mean(z, 200) for z in x_train]))
validation_vecs_cbowsg_mean = scale(np.concatenate([get_w2v_mean(z, 200) for z in x_validation]))
%%time
clf = LogisticRegression()
clf.fit(train_vecs_cbowsg_mean, y_train)
clf.score(validation_vecs_cbowsg_mean, y_validation)  #0.7650375939849624

def get_w2v_sum(tweet, size):
    vec = np.zeros(size).reshape((1, size))
    for word in tweet.split():
        try:
            vec += np.append(model_ug_cbow[word],model_ug_sg[word]).reshape((1, size))
        except KeyError:
            continue
    return vec

train_vecs_cbowsg_sum = scale(np.concatenate([get_w2v_sum(z, 200) for z in x_train]))
validation_vecs_cbowsg_sum = scale(np.concatenate([get_w2v_sum(z, 200) for z in x_validation]))
%%time
clf = LogisticRegression()
clf.fit(train_vecs_cbowsg_sum, y_train)
clf.score(validation_vecs_cbowsg_sum, y_validation)   #0.7675438596491229
```


The concatenated vectors of unigram CBOW and unigram Skip Gram models has yielded 76.50%, 76.75% validation accuracy respectively with mean and sum method. These results are even higher than the results I got from GloVe vectors. But please do not confuse this as a general statement. This is an empirical finding in this particular setting.

#### Separately trained Word2Vec with custom weighting (Average/Sum)

As a final step, I will apply the custom weighting I have implemented above and see if this affects the performance.

```python
%%time
w2v_pos_hmean_01 = {}
for w in model_ug_cbow.wv.vocab.keys():
    if w in pos_hmean.keys():
        w2v_pos_hmean_01[w] = np.append(model_ug_cbow[w],model_ug_sg[w]) * pos_hmean[w]

train_vecs_w2v_poshmean_mean_01 = scale(np.concatenate([get_w2v_general(z, 200, w2v_pos_hmean_01, 'mean') for z in x_train]))
validation_vecs_w2v_poshmean_mean_01 = scale(np.concatenate([get_w2v_general(z, 200, w2v_pos_hmean_01, 'mean') for z in x_validation]))
%%time
clf = LogisticRegression()
clf.fit(train_vecs_w2v_poshmean_mean_01, y_train)
clf.score(validation_vecs_w2v_poshmean_mean_01, y_validation)   #0.7797619047619048

train_vecs_w2v_poshmean_sum_01 = scale(np.concatenate([get_w2v_general(z, 200, w2v_pos_hmean_01, 'sum') for z in x_train]))
validation_vecs_w2v_poshmean_sum_01 = scale(np.concatenate([get_w2v_general(z, 200, w2v_pos_hmean_01, 'sum') for z in x_validation]))
%%time
clf = LogisticRegression()
clf.fit(train_vecs_w2v_poshmean_sum_01, y_train)
clf.score(validation_vecs_w2v_poshmean_sum_01, y_validation)   #0.7451754385964913
```

Finally I get the best performing word vectors. Averaged word vectors (separately trained Word2Vec models) weighted with custom metric has yielded the best validation accuray of 77.97%! Below is the table of all the results I tried above.

Word vectors extracted from | Vector dimensions | Weightings | Validation Accuracy with mean | Validation accuracy with sum
--- | --- | --- | --- | ---
Doc2Vec (unigram DBOW + unigram DMM) | 200 | N/A    | 72.42% | 72.51%
Doc2Vec (unigram DBOW + unigram DMM) | 200 | TF-IDF | 70.57% | 70.32%
Doc2Vec (unigram DBOW + unigram DMM) | 200 | custom | 73.27% | 70.94%
pre-trained GloVe (Tweets)	| 200 | N/A             | 76.27% | 76.60%
pre-trained Word2Vec (Google News) | 300 | N/A      | 74.96% | 74.92%
Word2Vec (unigram CBOW + unigram SG) | 200 | N/A    | 76.50% | 76.75%
Word2Vec (unigram CBOW + unigram SG) | 200 | custom	| 77.98% | 74.52%


#### Neural Network with Word2Vec

The best performing word vectors with logistic regression was chosen to feed to a neural network model. This time I did not try various different architecture. Based on what I have observed during trials of different artchitectures with Doc2Vec document vectors, the best performing architecture was one with 3 hiddel layers with 256 hidden nodes at each hidden layer.

I will finally fit a neural network with early stopping and checkpoint so that I can save the best performing weights on validation accuracy.

```python
train_w2v_final = train_vecs_w2v_poshmean_mean_01
validation_w2v_final = validation_vecs_w2v_poshmean_mean_01

from keras.callbacks import ModelCheckpoint, EarlyStopping

filepath="w2v_01_best_weights.{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_acc', patience=5, mode='max')
callbacks_list = [checkpoint, early_stop]
np.random.seed(seed)
model_w2v_01 = Sequential()
model_w2v_01.add(Dense(256, activation='relu', input_dim=200))
model_w2v_01.add(Dense(256, activation='relu'))
model_w2v_01.add(Dense(256, activation='relu'))
model_w2v_01.add(Dense(1, activation='sigmoid'))
model_w2v_01.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_w2v_01.fit(train_w2v_final, y_train, validation_data=(validation_w2v_final, y_validation),
                 epochs=100, batch_size=32, verbose=2, callbacks=callbacks_list)

from keras.models import load_model
loaded_w2v_model = load_model('w2v_01_best_weights.10-0.8048.hdf5')
loaded_w2v_model.evaluate(x=validation_w2v_final, y=y_validation)  #[0.4244666022615026, 0.8047619047619048]
```

The best validation accuracy is 80.48%. Surprisingly this is even hihger than the best accuracy I got by feeding document vectors to neurla network models in the above.

It took quite some time for me to try different settings, different calculations, but I learned some valuable lessons through all the trial and errors. Specifically trained Word2Vec with carefully engineered weighting can even outperform Doc2Vec in classification task.

In the next post, I will try more sophisticated neural network model, Convolutional Neural Network. Again I hope this will give me some boost of the performance.

## <a name="CNN with Word2Vec">CNN with Word2Vec</a>

```python
csv = 'clean_tweet.csv'
my_df = pd.read_csv(csv,index_col=0)
my_df.head()
my_df.dropna(inplace=True)
my_df.reset_index(drop=True,inplace=True)
my_df.info()
x = my_df.text
y = my_df.target
from sklearn.cross_validation import train_test_split
SEED = 2000
x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)
x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from sklearn import utils

def labelize_tweets_ug(tweets,label):
    result = []
    prefix = label
    for i, t in zip(tweets.index, tweets):
        result.append(TaggedDocument(t.split(), [prefix + '_%s' % i]))
    return result

all_x = pd.concat([x_train,x_validation,x_test])
all_x_w2v = labelize_tweets_ug(all_x, 'all')
cores = multiprocessing.cpu_count()
model_ug_cbow = Word2Vec(sg=0, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_cbow.build_vocab([x.words for x in tqdm(all_x_w2v)])
%%time
for epoch in range(30):
    model_ug_cbow.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_cbow.alpha -= 0.002
    model_ug_cbow.min_alpha = model_ug_cbow.alpha

model_ug_sg = Word2Vec(sg=1, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_sg.build_vocab([x.words for x in tqdm(all_x_w2v)])
%%time
for epoch in range(30):
    model_ug_sg.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_sg.alpha -= 0.002
    model_ug_sg.min_alpha = model_ug_sg.alpha

model_ug_cbow.save('w2v_model_ug_cbow.word2vec')
model_ug_sg.save('w2v_model_ug_sg.word2vec')
```

### Prepration for Convolutional Neural Network

```python
from gensim.models import KeyedVectors
model_ug_cbow = KeyedVectors.load('w2v_model_ug_cbow.word2vec')
model_ug_sg = KeyedVectors.load('w2v_model_ug_sg.word2vec')
```

By running below code block, I am constructing a sort of dictionary I can extract the word vectors from. Since I have two different Word2Vec models, below "embedding_index" will have concatenated vectors of the two models. For each model, I have 100 dimension vector representation of the words, and by concatenating each word will have 200 dimension vector representation.

```python
embeddings_index = {}
for w in model_ug_cbow.wv.vocab.keys():
    embeddings_index[w] = np.append(model_ug_cbow.wv[w],model_ug_sg.wv[w])
print('Found %s word vectors.' % len(embeddings_index))
```

Now we have our reference to word vectors ready, but we still haven't prepared data to be in the format I have explained at the start of the post. Keras' `Tokenizer` will split each word in a sentence, then we can call `texts_to_sequences` method to get the sequential representation of each sentence. We also need to pass `num_words` which is a number of vocabularies you want to use, and this will be applied when you call `texts_to_sequences` method. This might be a bit counter-intuitive since if you check the length of all the word index, it will not be the number of words you defined, but the actual screening process happens when you call `texts_to_sequences` method.

```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=100000)
tokenizer.fit_on_texts(x_train)
sequences = tokenizer.texts_to_sequences(x_train)
```

Each word is represented as a number, and we can see that the number of words in each sentence is matching the length of numbers in the "sequences". We can later make connections of which word each number represents. But we still didn't pad our data, so each sentence has varying length. Let's deal with this.

The maximum number of words in a sentence within the training data is 40. Let's decide the maximum length to be a bit longer than this, let's say 45.

```python
x_train_seq = pad_sequences(sequences, maxlen=45)
print('Shape of data tensor:', x_train_seq.shape)   #(1564098, 45)
```

As you can see from the padded sequences, all the data now transformed to have the same length of 45, and by default, Keras zero-pads at the beginning, if a sentence length is shorter than the maximum length. If you want to know more in detail, please check the Keras documentation on sequence preprocessing. https://keras.io/preprocessing/sequence/

```python
sequences_val = tokenizer.texts_to_sequences(x_validation)
x_val_seq = pad_sequences(sequences_val, maxlen=45)
```

There's still one more thing left to do before we can feed the sequential text data to a model. When we transformed a sentence into a sequence, each word is represented by an integer number. Actually, these numbers are where each word is stored in the tokenizer's word index. Keeping this in mind, let's build a matrix of these word vectors, but this time we will use the word index number so that our model can refer to the corresponding vector when fed with integer sequence.

Below, I am defining the number of words to be 100,000. This means I will only care about 100,000 most frequent words in the training set. If I don't limit the number of words, the total number of vocabulary will be more than 200,000.

```python
num_words = 100000
embedding_matrix = np.zeros((num_words, 200))
for word, i in tokenizer.word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
```

Now we are ready with the data preparation. Before we jump into CNN, I would like to test one more thing (sorry for the delay). When we feed this sequential vector representation of data, we will use Embedding layer in Keras. With Embedding layer, I can either pass pre-defined embedding, which I prepared as 'embedding_matrix' above, or Embedding layer itself can learn word embeddings as the whole model trains. And another possibility is we can still feed the pre-defined embedding but make it trainable so that it will update the values of vectors as the model trains.

In order to check which method performs better, I defined a simple shallow neural network one hidden layer. For this model structure, I will not try to refine models by tweaking parameters, since the main purpose of this post is to implement CNN.
https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

```python
seed = 7
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding

model_ptw2v = Sequential()
e = Embedding(100000, 200, weights=[embedding_matrix], input_length=45, trainable=False)
model_ptw2v.add(e)
model_ptw2v.add(Flatten())
model_ptw2v.add(Dense(256, activation='relu'))
model_ptw2v.add(Dense(1, activation='sigmoid'))
model_ptw2v.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_ptw2v.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_validation), epochs=5, batch_size=32, verbose=2)

model_ptw2v = Sequential()
e = Embedding(100000, 200, input_length=45)
model_ptw2v.add(e)
model_ptw2v.add(Flatten())
model_ptw2v.add(Dense(256, activation='relu'))
model_ptw2v.add(Dense(1, activation='sigmoid'))
model_ptw2v.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_ptw2v.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_validation), epochs=5, batch_size=32, verbose=2)

model_ptw2v = Sequential()
e = Embedding(100000, 200, weights=[embedding_matrix], input_length=45, trainable=True)
model_ptw2v.add(e)
model_ptw2v.add(Flatten())
model_ptw2v.add(Dense(256, activation='relu'))
model_ptw2v.add(Dense(1, activation='sigmoid'))
model_ptw2v.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_ptw2v.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_validation), epochs=5, batch_size=32, verbose=2)
```

As a result, the best validation accuracy is from the third method (fine-tune pre-trained Word2Vec) at 82.22%. The best training accuracy is the second method (learn word embedding from scratch) at 90.52%. Using pre-trained Word2Vec without updating its vector values showed the lowest accuracy both in training and validation. However, what's interesting is that in terms of training set accuracy, fine-tuning pre-trained word vectors couldn't outperform the word embeddings learned from scratch through the embedding layer. Before I tried the above three methods, my first guess was that if I fine-tune the pre-trained word vectors, it would give me the best training accuracy.

Feeding pre-trained word vectors for an embedding layer to update is like providing the first initialisation guideline to the embedding layer so that it can learn more efficiently the task-specific word vectors. But the result is somewhat counterintuitive, and in this case, it turns out that it is better to force the embedding layer to learn from scratch.

But premature generalization is a dangerous step to take. For this reason, I will compare three methods again in the context of CNN.

### Convolutional Neural Network

What we do with text data represented in word vectors is making use of 1D Convolutional Neural Network. If a filter's column width is as same as the data column width, then it has no room to stride horizontally, and only stride vertically. For example, if our sentence is represented in 45X200 matrix, then a filter column width will also have 200 columns, and the length of row (height) will be similar to the concept of n-gram. If the filter height is 2, the filter will stride through the document computing the calculation above with all the bigrams, if the filter height is 3, it will go through all the trigrams in the document, and so on.

If a 2X200 filter is applied with stride size of 1 to 45X200 matrix, we will get 44X1 dimensional output. If we apply 100 2X200 filters with stride size of 1 to 45X200 matrix, we will have 44X100 dimension output.

```python
from keras.layers import Conv1D, GlobalMaxPooling1D

structure_test = Sequential()
e = Embedding(100000, 200, input_length=45)
structure_test.add(e)
structure_test.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
structure_test.summary()
#Layer (type)                 Output Shape              Param #
#=================================================================
#embedding_44 (Embedding)     (None, 45, 200)           20000000
#_________________________________________________________________
#conv1d_37 (Conv1D)           (None, 44, 100)           40100
#=================================================================
#Total params: 20,040,100
#Trainable params: 20,040,100
#Non-trainable params: 0
#_________________________________________________________________
```

Now if we add Global Max Pooling layer, then the pooling layer will extract the maximum value from each filter, and the output dimension will be a just 1-dimensional vector with length as same as the number of filters we applied. This can be directly passed on to a dense layer without flattening.

```python
structure_test = Sequential()
e = Embedding(100000, 200, input_length=45)
structure_test.add(e)
structure_test.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
structure_test.add(GlobalMaxPooling1D())
structure_test.summary()
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #
#=================================================================
#embedding_47 (Embedding)     (None, 45, 200)           20000000
#_________________________________________________________________
#conv1d_40 (Conv1D)           (None, 44, 100)           40100
#_________________________________________________________________
#global_max_pooling1d_21 (Glo (None, 100)               0
#=================================================================
#Total params: 20,040,100
#Trainable params: 20,040,100
#Non-trainable params: 0
#_________________________________________________________________
```

Now, let's define a simple CNN going through bigrams on a tweet. The output from global max pooling layer will be fed to a fully connected layer, then finally the output layer. Again I will try three different inputs, static word vectors extracted from Word2Vec, word embedding being learned from scratch with embedding layer, Word2Vec word vectors being updated through training.

```python
model_cnn_01 = Sequential()
e = Embedding(100000, 200, weights=[embedding_matrix], input_length=45, trainable=False)
model_cnn_01.add(e)
model_cnn_01.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
model_cnn_01.add(GlobalMaxPooling1D())
model_cnn_01.add(Dense(256, activation='relu'))
model_cnn_01.add(Dense(1, activation='sigmoid'))
model_cnn_01.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_cnn_01.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_validation), epochs=5, batch_size=32, verbose=2)

model_cnn_02 = Sequential()
e = Embedding(100000, 200, input_length=45)
model_cnn_02.add(e)
model_cnn_02.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
model_cnn_02.add(GlobalMaxPooling1D())
model_cnn_02.add(Dense(256, activation='relu'))
model_cnn_02.add(Dense(1, activation='sigmoid'))
model_cnn_02.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_cnn_02.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_validation), epochs=5, batch_size=32, verbose=2)

model_cnn_03 = Sequential()
e = Embedding(100000, 200, weights=[embedding_matrix], input_length=45, trainable=True)
model_cnn_03.add(e)
model_cnn_03.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
model_cnn_03.add(GlobalMaxPooling1D())
model_cnn_03.add(Dense(256, activation='relu'))
model_cnn_03.add(Dense(1, activation='sigmoid'))
model_cnn_03.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_cnn_03.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_validation), epochs=5, batch_size=32, verbose=2)
```

The best validation accuracy is from the word vectors updated through training, at epoch 3 with validatioan accuracy of 83.25%. By looking at the training loss and accuracy, it seems that word embedding learned from scratch tends to overfit to the trainig data, and by passing pre-trained word vectors as weights initialisation, it somewhat more generalizes and ends up having higher validation accuracy.

But finally! I have a better result than Tf-Idf + logistic regression model! I have tried various different methods with Doc2Vec, Word2Vec in the hope of outperforming a simple logistic regression model with Tf-Idf input. You can take a look at the previous post for detail. Tf-Idf + logistic regression model’s validation accuracy was at 82.91%. And now I’m finally beginning to see a possibility of Word2Vec + neural network outperforming this simple model.

Let’s see if we can do better by defining a bit more elaborate model structure. The CNN architecture I will implement below is inspired by Zhang, Y., & Wallace, B. (2015) “A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification”.

Basically, the above structure is implementing what we have done above with bigram filters, but not only to bigrams but also to trigrams and fourgrams. However this is not linearly stacked layers, but parallel layers. And after convolutional layer and max pooling layer, it simply concatenated max pooled result from each of bigram, trigram, and fourgram, then build one output layer on top of them.

The model I defined below is basically as same as the above picture, but the differences are that I added one fully connected hidden layer with dropout just before the output layer, and also my output layer will have just one output node with Sigmoid activation instead of two.

There is also another famous paper by Y. Kim(2014), “Convolutional Neural Networks for Sentence Classification”. https://arxiv.org/pdf/1408.5882.pdf

In this paper, he implemented more sophisticated approach by making use of “channel” concept. Not only the model go through different n-grams, his model has multi-channels (eg. one channel for static input word vectors, another channel for word vectors input but set them to update during training). But in this post, I will not go through multi-channel approach.

So far I have only used Sequential model API of Keras, and this worked fine with all the previous models I defined above since the structures of the models were only linearly stacked. But as you can see from the above picture, the model I am about to define has parallel layers which take the same input but do their own computation, then the results will be merged. In this kind of neural network structure, we can use Keras functional API.

Keras functional API can handle multi-input, multi-output, shared layers, shared input, etc. It is not impossible to define these types of models with Sequential API, but when you want to save the trained model, functional API enables you to simply save the model and load, but with sequential API it is difficult.

```python
from keras.layers import Input, Dense, concatenate, Activation
from keras.models import Model

tweet_input = Input(shape=(45,), dtype='int32')

tweet_encoder = Embedding(100000, 200, weights=[embedding_matrix], input_length=45, trainable=True)(tweet_input)
bigram_branch = Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1)(tweet_encoder)
bigram_branch = GlobalMaxPooling1D()(bigram_branch)
trigram_branch = Conv1D(filters=100, kernel_size=3, padding='valid', activation='relu', strides=1)(tweet_encoder)
trigram_branch = GlobalMaxPooling1D()(trigram_branch)
fourgram_branch = Conv1D(filters=100, kernel_size=4, padding='valid', activation='relu', strides=1)(tweet_encoder)
fourgram_branch = GlobalMaxPooling1D()(fourgram_branch)
merged = concatenate([bigram_branch, trigram_branch, fourgram_branch], axis=1)

merged = Dense(256, activation='relu')(merged)
merged = Dropout(0.2)(merged)
merged = Dense(1)(merged)
output = Activation('sigmoid')(merged)
model = Model(inputs=[tweet_input], outputs=[output])
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
model.summary()
#Layer (type)                    Output Shape         Param #     Connected to
#==================================================================================================
#input_7 (InputLayer)            (None, 45)           0
#__________________________________________________________________________________________________
#embedding_59 (Embedding)        (None, 45, 200)      20000000    input_7[0][0]
#__________________________________________________________________________________________________
#conv1d_66 (Conv1D)              (None, 44, 100)      40100       embedding_59[0][0]
#__________________________________________________________________________________________________
#conv1d_67 (Conv1D)              (None, 43, 100)      60100       embedding_59[0][0]
#__________________________________________________________________________________________________
#conv1d_68 (Conv1D)              (None, 42, 100)      80100       embedding_59[0][0]
#__________________________________________________________________________________________________
#global_max_pooling1d_47 (Global (None, 100)          0           conv1d_66[0][0]
#__________________________________________________________________________________________________
#global_max_pooling1d_48 (Global (None, 100)          0           conv1d_67[0][0]
#__________________________________________________________________________________________________
#global_max_pooling1d_49 (Global (None, 100)          0           conv1d_68[0][0]
#__________________________________________________________________________________________________
#concatenate_8 (Concatenate)     (None, 300)          0           global_max_pooling1d_47[0][0]
#                                                                 global_max_pooling1d_48[0][0]
#                                                                 global_max_pooling1d_49[0][0]
#__________________________________________________________________________________________________
#dense_52 (Dense)                (None, 256)          77056       concatenate_8[0][0]
#__________________________________________________________________________________________________
#dropout_6 (Dropout)             (None, 256)          0           dense_52[0][0]
#__________________________________________________________________________________________________
#dense_53 (Dense)                (None, 1)            257         dropout_6[0][0]
#__________________________________________________________________________________________________
#activation_4 (Activation)       (None, 1)            0           dense_53[0][0]
#==================================================================================================
#Total params: 20,257,613
#Trainable params: 20,257,613
#Non-trainable params: 0
```

```python
from keras.callbacks import ModelCheckpoint
filepath="CNN_best_weights.{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model.fit(x_train_seq, y_train, batch_size=32, epochs=5, validation_data=(x_val_seq, y_validation), callbacks = [checkpoint])

from keras.models import load_model
loaded_CNN_model = load_model('CNN_best_weights.02-0.8333.hdf5')
loaded_CNN_model.evaluate(x=x_val_seq, y=y_validation)
```

The best validation accuracy is 83.33%, slightly better than the simple CNN model with bigram filters, which yielded 83.25% validation accuracy. I could even define a deeper structure with more hidden layers, or even make use of the multi-channel approach that Yoon Kim(2014) has implemented, or try different pool size to see how the performance differs, but I will stop here for now. However if you happen to try more complex CNN structure, and get the result, I would love to hear about it.

#### Final Model Evaluation with Test Set

So far I have tested the model on the validation set to decide the feature extraction tuning and model comparison. Now I will finally check the final result with the test set. I will compare two different models: 1. Tf-Idf + logistic regression, 2. Word2Vec + CNN. As another measure for comparison, I will also plot ROC curve of both models.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tvec = TfidfVectorizer(max_features=100000,ngram_range=(1, 3))
tvec.fit(x_train)
x_train_tfidf = tvec.transform(x_train)
x_test_tfidf = tvec.transform(x_test)
lr_with_tfidf = LogisticRegression()
lr_with_tfidf.fit(x_train_tfidf,y_train)
lr_with_tfidf.score(x_test_tfidf,y_test)   #0.8329678591566945
yhat_lr = lr_with_tfidf.predict_proba(x_test_tfidf)

sequences_test = tokenizer.texts_to_sequences(x_test)
x_test_seq = pad_sequences(sequences_test, maxlen=45)
loaded_CNN_model.evaluate(x=x_test_seq, y=y_test)   #[0.3629236272231923, 0.8386066035813283]
yhat_cnn = loaded_CNN_model.predict(x_test_seq)

from sklearn.metrics import roc_curve, auc
fpr, tpr, threshold = roc_curve(y_test, yhat_lr[:,1])
roc_auc = auc(fpr, tpr)
fpr_cnn, tpr_cnn, threshold = roc_curve(y_test, yhat_cnn)
roc_auc_nn = auc(fpr_cnn, tpr_cnn)
plt.figure(figsize=(8,7))
plt.plot(fpr, tpr, label='tfidf-logit (area = %0.3f)' % roc_auc, linewidth=2)
plt.plot(fpr_cnn, tpr_cnn, label='w2v-CNN (area = %0.3f)' % roc_auc_nn, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('Receiver operating characteristic: is positive', fontsize=18)
plt.legend(loc="lower right")
plt.show()
```

And the final result is as below.

model | validation set accuracy | test set accuracy | ROC AUC
-- | -- | -- | --
Tf-Idf + logistic regression | 82.91% | 83.30% | 0.91
Word2Vec + CNN | 83.33% | 83.86% | 0.92


References:
