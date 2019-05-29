'''classify_utterance_type.py

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Introduction to Text Classification 
NORC Data Science Brown Bag, may29/2019 
https://github.com/lefft/text_classification
leffel-timothy@norc.org
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


This file illustrates a sample workflow for multi-class 
text classification. The dataset provided is a collection 
of 551 short movie lines drawn from the Cornell Movie Dialog Corpus. 
The file is called `cmdc_lines-annotated-551.txt`. 

Each line in the sample dataset consists of two 
pipe-separated values -- the first is a category indicating 
"utterance type", and the second is the actual text of the 
corresponding movie line. 

There are four possible utterance types considered here: 
  - 'd' ==> a statement/declarative sentence 
  - 'q' ==> a question/interrogative sentence 
  - 'c' ==> a command/imperative sentence 
  - 'o' ==> something else, e.g. a greeting 



### Outline of this script 
0. setup/prep:
    - import modules, functions, and classes we'll need 
    - specify paths to input and output files we'll use/create 
1. load + partition data: 
    - read in the annotated text data (+ explore a bit) 
    - randomly split it into training and evaluation sets 
2. define main modeling function, which will... 
    - preprocess the raw text documents (optional) 
    - transform each document into a row of a numeric matrix 
    - fit a model using only the training data matrix and labels 
    - generate class predictions over the evaluation set documents 
    - measure performance by comparing predictions to human labels 
3. call the function with different argument combinations, so that... 
    - we get an idea of how certain choices affect accuracy; 
    - we get an idea of what strategies will work well for this dataset; 
    - we can ultimately hone in on a final model. 
4. write a text file that has information about each model fit: 
    - how was the text preprocessed, if at all? 
    - what kind of features were extracted to form the input matrix? 
    - what (non-default) hyper-parameter settings were used, if any? 



### Statement of problem (what's our objective?) 
We want to find a model M such that for any English 
utterance u, M(u) is u's "utterance type". 

Here an "utterance type" is one of the following categories: 
  - (d)eclarative -- declarative sentence (statement, assertion) 
  - (q)uestion -- interrogative sentence (intuitively: a question) 
  - (c)ommand -- imperative sentence (a command or directive) 
  - (o)ther -- anything else (e.g. exclamations, greetings) 

Even though these definitions are not mathematically precise, 
our annotated data points provide many exemplars of each type. 
In supervised classification, this is sometimes the best thing 
you can get (linguistic categories can be ineffable!). 

If defining the classes were easy to do in a mathematically 
rigorous fashion, then we would probably just use our knowledge 
of English and a bit of experimentation to develop an accurate 
rule-based classifier! 

By using a statistical model instead of rules, we are essentially 
outsourcing the task of thinking about the nature of the data 
from the analyst's mind to linear algebra's mind. 



### General approach to the problem 
We will develop an utterance type classifier using a 
statistical model called "Multinomial Naive Bayes" (NB). 

All that's relevant for now is that a fitted NB model can be used 
to predict the class of a new/unseen text document -- call it doc. 
It does so by calculating the conditional probability prob(cls|doc) 
for each of our classes cls. The class with the highest such 
probability -- the argmax -- is selected as the model prediction. 

Here's the steps we'll take: 
  1. partition the data into a training set and an evaluation set 
  2. transform the documents into a numerical matrix (BoW DTM) 
  3. fit ("train") a model using only the training data 
  4. use the model to predict each evaluation doc's class from its DTM row 
  5. compare model predictions to true labels to measure performance 



### Specific preprocessing and feature extraction configurations we'll try 
  1. no text preprocessing, just word 1-grams 
  2. light text preprocessing, just word 1-grams 
  3. light text preprocessing, using word {1,2}-grams 
  4. no text preprocessing, just character 3-grams  
  5. light text preprocessing, just character 3-grams 



### Lessons to be learned from this demo 
  1. text preprocessing choices can have a huge impact on model accuracy 
  2. feature extraction choices can have a huge impact on model accuracy 
  3. these effects can and do interact with one another! 



### A few bonus exercises for the brave and/or interested 
##### 1. hyper-parameter tuning 
  - use sklearn.model_selection.GridSearchCV to optimize model config 
  - retrain using optimized params, and measure performance on eval set 
  - compare baseline results to tuned model results 
  - think about these questions: 
      - what's the relative impact of preprocessing versus model tuning? 
      - how might you integrate text preprocessing into a grid search? 
##### 2. post hoc feature analysis 
  - how might you find the "most important" features for this problem? 
  - what measure might you use to measure "importance"? 
  - in this data, what features are characteristic of each utterance type... 
      - when using bag of words features? 
      - when using bag of character n-grams features? 
  - how might we distill info about important features to improve our model? 
  - how do important features change if stop words are included? 
  - what does the answer to the previous question say about stop word removal? 
##### 3. alternative classification algorithms 
  - try replacing sklearn.naive_bayes.MultinomialNB with any of these: 
      - sklearn.linear_model.LogisticRegression 
      - sklearn.linear_model.SGDClassifier 
      - sklearn.tree.DecisionTreeClassifier 
      - sklearn.svm.SVC 
  - then fiddle around with the modeling function using one of the above 
  - what happens to performance when alternative algorithms are used? 
  - (how) do preprocessing choices interact with algorithm choice? 
  - (how) do feature extraction choices interact with algorithm choice? 
'''



### 0. setup/prep ------------------------------------------------------------

### import modules, functions, and classes we'll need 
import re

from collections import Counter

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score


### specify paths to input and output files we'll use/create 
movie_lines_infile = '../data/cmdc_lines-annotated-551.txt'
model_eval_outfile = '../output/cmdc_lines-model_info.txt'




### 1. load, check, + partition data ------------------------------------------

### read in the annotated text data 
labs, docs = [], []
with open(movie_lines_infile, 'r') as file:
  for line in file:
    lab, doc = line.strip().split('|')
    # what happens if we use `lab=='q'` instead of `lab`? 
    # what about for the other utterance type categories? 
    # labs.append(lab=='q')
    labs.append(lab)
    # note usage of padding characters `<` and `>`! 
    docs.append(f'< {doc} >')

# remove header row 
labs, docs = labs[1:], docs[1:]


### explore a bit -- check label + distros, + 25 most freq words 
print(f'\nlabel distro:\n  >> {Counter(labs)}\n\n')   # label distro 
Counter(' '.join(docs))                                    # char freqs 
Counter(sum([doc.split() for doc in docs], [])).most_common(25)   # word freqs 

# (here's the label distribution, just for reference: 
# Counter({'d': 230, 'q': 192, 'o': 66, 'c': 64})


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
  classifier = MultinomialNB()

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



