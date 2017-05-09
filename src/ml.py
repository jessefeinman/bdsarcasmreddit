import pickle
import warnings
from copy import deepcopy
from datetime import datetime
from itertools import islice, tee
from os import listdir
from random import shuffle

import numpy as np
from json_io import tweet_iterate
from nlp import *
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectPercentile, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit

DEFAULT_CLASSIFIERS = [
    LogisticRegression(n_jobs=-1)
    # LogisticRegression(solver='sag', max_iter=1000, n_jobs=-1, warm_start=True),
    # SGDClassifier(loss='log', penalty='elasticnet', n_jobs=-1),
    # BernoulliNB(alpha=0.2, binarize=0.4),
    # MultinomialNB(alpha=0),
]
DEFAULT_CLASSIFIERS_ARGS = [
    # (SGDClassifier(penalty='elasticnet', n_jobs=-1), {'loss':['log','modified_huber','perceptron'], 'penalty':['none','l1','elasticnet','l2']}),
    # (BernoulliNB(),{'alpha':list(np.arange(0,20,0.1)), 'binarize':list(np.arange(0.1,0.9,0.1))}),
    # (MultinomialNB(),{'alpha':list(np.arange(0,20,0.1))})
]
FILENAME_REGEX = r'[ :<>".*,|\/]+'
PICKLED_FEATS_DIR = 'pickledfeatures/'
JSON_DIR = '../json/'

def trainTest(X, y, classifiers=DEFAULT_CLASSIFIERS, reduce=0, splits=10, trainsize=0.8, testsize=0.2):
    sss = StratifiedShuffleSplit(n_splits=splits, test_size=testsize, train_size=trainsize)
    results = []
    for i, (train_index, test_index) in enumerate(sss.split(X, y)):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        if reduce > 0:
            print("Features before reduction: " + str(X_train.shape))
            reducer = SelectKBest(score_func=f_classif, k=reduce)
            X_train = reducer.fit_transform(X_train, y_train)
            X_test = reducer.transform(X_test)
            print("Features after reduction: " + str(str(X_train.shape)))
            support = reducer.get_support()
        
        for classifier in classifiers:
            print("Starting to train %s"%str(type(classifier)))
            s = datetime.now()
            classifier.fit(X_train, y_train)
            traintime = (datetime.now() - s).total_seconds()
            score = classifier.score(X_test, y_test)
            if reduce > 0:
                results.append((classifier, traintime, score, support))
            else:
                results.append((classifier, traintime, score))
            print("%s\tTime: %d\tScore:\t%f" %(str(type(classifier)), traintime, score))
    return results

def flattenDict(feature):
    d = {}
    for key, value in feature.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                d[subkey] = subvalue
        else:
            d[key] = value
    return d

def flatten(X,y=None):
    if y:
        return (flattenDict(x) for x in X), y
    else:
        return (flattenDict(x) for x in X)

def saveVectorizer(dv, X=None, y=None, extra=''):
    if X is not None and y is not None:
        pickle.dump((dv, X, y), open(PICKLED_FEATS_DIR + 'Xydv' + extra  + '.pickle', 'wb'))
        pickle.dump(dv, open(PICKLED_FEATS_DIR + 'dv' + extra  + '.pickle', 'wb'))
    else:
        pickle.dump(dv, open(PICKLED_FEATS_DIR + 'dv' + extra  + '.pickle', 'wb'))

def split_feat(gen, n):
    def create_generator(it, n):
        return (item[n] for item in it)
    G = tee(gen, n)
    return [create_generator(g, n) for n, g in enumerate(G)]

def predict(listOfString, classifier, dvp, cleanTokens):
    listOfFeats = [flattenDict(feature(s, cleanTokens)) for s in listOfString]
    X = dvp.transform(listOfFeats)
    prediction = classifier.predict(X)
    invert_op = getattr(c, "predict_proba", None)
    if callable(invert_op):
        preProb = classifier.predict_proba(X)
        return {'classifier':classifier, 'prediction': prediction, 'prediction_probabilities':preProb}
    else:
        return {'classifier':classifier, 'prediction': prediction}
    print(r)
    return r