import pickle
import warnings
from copy import deepcopy
from datetime import datetime
from itertools import islice, tee
from os import listdir
from random import shuffle

import numpy as np
from nlp import *
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectPercentile, SelectKBest, f_classif
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit

def trainTest(X, y, classifiers, reduce=0, splits=10, trainsize=0.8, testsize=0.2):
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

def split_feat(gen, n):
    def create_generator(it, n):
        return (item[n] for item in it)
    G = tee(gen, n)
    return [create_generator(g, n) for n, g in enumerate(G)]

def predict(listOfString, classifier, dvp, cleanTokens):
    listOfFeats = [flattenDict(feature(s, cleanTokens)) for s in listOfString]
    X = dvp.transform(listOfFeats)
    prediction = classifier.predict(X)
    invert_op = getattr(classifier, "predict_proba", None)
    if callable(invert_op):
        preProb = classifier.predict_proba(X)
        return {'classifier':classifier, 'prediction': prediction, 'prediction_probabilities':preProb}
    else:
        return {'classifier':classifier, 'prediction': prediction}
    print(r)
    return r