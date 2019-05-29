'''classify_tweet_relevance.py

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Introduction to Text Classification 
NORC Data Science Brown Bag, may29/2019 
https://github.com/lefft/text_classification
leffel-timothy@norc.org
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


This file provides a sample workflow for binary (two-class) 
document classification. The dataset provided is a collection 
of 600 tweets, each hand-annotated with a binary label indicating 
whether the tweet text is relevant to the topic of "Juul" (Juul is 
the most popular nicotine vaping device in the US, as of may/2019). 

The data file containing tweets is called `sample_tweets-600.txt`. 
The first column contains the juul-relevance label, and it is coded as: 
  - '0' ==> not relevant to Juul 
  - '1' ==> relevant to Juul 

This script has the same structure as `classify_utterance_type.py` -- see 
that file's docstring for details. The main difference is that here 
we are using logistic regression instead of naive bayes -- a common (but 
not totally necessary) choice when doing binary classification. 

Notice that even though we are dealing with a binary problem here, 
everything else is the same. This illustrates the beautifully designed 
scikit-learn classification and preprocessing APIs -- all classification 
problems can be thought of as having the same core components: 

  - a categorical vector of labels (with 2 or more values); 
  - a matrix of numeric features used to predict the labels; 
  - a vectorizer to build the matrix from unstructured data (if necessary); and
  - a classification algorithm to instantiate and fit/train. 

In contexts such as e.g. credit risk modeling, the matrix of features might 
just be derived from explicit demographic or survey info about applicants, 
perhaps already stored in a tabular format. In this case, the data is already 
vectorized and the workflow is simply: partition the data, train a model, and 
generate predictions on your holdout/evaluation set. 

What makes text classification different is almost entirely one thing: there is 
an additional step required to transform raw text documents into numerical 
representations that we can compute on. In the case of document classification, 
we start with n unstructured text documents, and then *build* a feature matrix 
with n rows, each of which represents a document. The rows of a feature matrix 
are sometimes called "document vectors," and the columns "feature/term vectors."
This kind of matrix is known as a *document-term matrix* (DTM), and is one of 
the fundamental data structures in NLP. 
'''





### 0. setup/prep ------------------------------------------------------------

### import modules, functions, and classes we'll need 
import re

import pandas as pd

from collections import Counter

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix


### specify paths to input and output files we'll use/create 
tweets_infile = '../data/tweet_samples-600.txt'
model_eval_outfile = '../output/tweet_samples-model_info.txt'




### 1. load, check, + partition data ------------------------------------------
dat = pd.read_csv(tweets_infile, sep='|', encoding='utf-8')

docs, labs = dat['text'].tolist(), dat['label'].tolist()

### explore a bit -- check label + distros, + 25 most freq words 
print(f'\nlabel distro:\n  >> {Counter(labs)}\n\n')   # label distro 
Counter(' '.join(docs))                                    # char freqs 
Counter(sum([doc.split() for doc in docs], [])).most_common(25)   # word freqs 

# (here's the label distribution, just for reference: 
# Counter({1: 424, 0: 176})


### randomly split data into training and evaluation sets 
docs_train, docs_test, labs_train, labs_test = train_test_split(
  docs, labs, test_size=.33, random_state=69)




### 2. define main modeling function ------------------------------------------

def prep_text(doc, toss_re=r'[?.,!$-]'):
  '''simple text preprocessing routine (remove punctuation)'''
  return re.sub(toss_re, '', doc)


def prep_train_evaluate(docs_train, docs_test, labs_train, labs_test, **kwargs):
  '''func to prep text, extract features, train model, predict, evaluate'''

  # instantiate vectorizer + classifier 
  vectorizer = CountVectorizer(token_pattern=r'\b[a-zA-Z0-9_<>]{1,}\b', 
                               **kwargs)
  classifier = LogisticRegression(solver='liblinear')

  # construct feature matrices for train and test sets 
  vectorizer.fit(docs_train)
  X_train = vectorizer.transform(docs_train)
  X_test = vectorizer.transform(docs_test)

  # fit/train classifier using train features and labels 
  classifier.fit(X_train, labs_train)

  # generate test set model predictions from test matrix 
  preds_test = classifier.predict(X_test)

  # measure performance using simple accuracy (proportion correct) 
  accuracy = accuracy_score(labs_test, preds_test)

  # print lil message showing param settings + performance 
  print(f'  >> test set accuracy: {accuracy:.3f}\n({kwargs})\n')

  # return classifier, vectorizer, predictions, and score for inspection 
  return {'clf': classifier, 'vect': vectorizer, 
          'preds': preds_test, 'acc': accuracy}




### 3. call the function with some different argument combos ------------------

''' fit 1: no text preprocessing, just word 1-grams ''' 
fit1_kwargs = {
  'analyzer': 'word', 'ngram_range': (1, 1), 
  'lowercase': False, 'preprocessor': None}

fit1_results = prep_train_evaluate(
  docs_train, docs_test, labs_train, labs_test, **fit1_kwargs)


''' fit 2: light text preprocessing, just word 1-grams ''' 
fit2_kwargs = {
  'analyzer': 'word', 'ngram_range': (1, 1), 
  'lowercase': True, 'preprocessor': prep_text}

fit2_results = prep_train_evaluate(
  docs_train, docs_test, labs_train, labs_test, **fit2_kwargs)


''' fit 3: light text preprocessing, using word {1,2}-grams ''' 
fit3_kwargs = {
  'analyzer': 'word', 'ngram_range': (1, 2), 
  'lowercase': True, 'preprocessor': prep_text}

fit3_results = prep_train_evaluate(
  docs_train, docs_test, labs_train, labs_test, **fit3_kwargs)


''' fit 4: light text preprocessing, character {1,2,3}-grams ''' 
fit4_kwargs = {
  'analyzer': 'char', 'ngram_range': (1, 3), 
  'lowercase': True, 'preprocessor': prep_text}

fit4_results = prep_train_evaluate(
  docs_train, docs_test, labs_train, labs_test, **fit4_kwargs)


''' fit 5: no text preprocessing, character {1,2,3}-grams ''' 
fit5_kwargs = {
  'analyzer': 'char', 'ngram_range': (1, 3), 
  'lowercase': False, 'preprocessor': None}

fit5_results = prep_train_evaluate(
  docs_train, docs_test, labs_train, labs_test, **fit5_kwargs)




### 4. write a text file that has info about each model fit -------------------
def make_outfile_row(fit_idx, fit_kwargs, fit_results):
  '''func to write a single row given prep param dict + results dict''' 
  acc = fit_results['acc']
  text_prep_settings = "|".join(val.__str__() for val in fit_kwargs.values())
  return f'{fit_idx}|{acc:.3f}|{text_prep_settings}'


print(f'writing clf results to file:\n  >> {model_eval_outfile}\n')
with open(model_eval_outfile, 'w') as file:
  # write the header and then a line/row for each fit/configuration 
  file.write('fit_idx|accuracy|analyzer|ngram_range|lowercase|preprocessor\n')
  file.write(make_outfile_row(1, fit1_kwargs, fit1_results)+'\n')
  file.write(make_outfile_row(2, fit2_kwargs, fit2_results)+'\n')
  file.write(make_outfile_row(3, fit3_kwargs, fit3_results)+'\n')
  file.write(make_outfile_row(4, fit4_kwargs, fit4_results)+'\n')
  file.write(make_outfile_row(5, fit5_kwargs, fit5_results)+'\n')


