import pickle
import warnings
from copy import deepcopy
from datetime import datetime
from itertools import islice
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


def reduceFeatures(X_train, X_test, y_train, dv, percent):
    dvc = deepcopy(dv)
    print("\n\nFeatures before reduction: " + str(X_train.shape))
    reducer = SelectPercentile(score_func=f_classif, percentile=percent)
    X_train = reducer.fit_transform(X_train, y_train)
    X_test = reducer.transform(X_test)
    print("\n\nFeatures after reduction: " + str(str(X_train.shape)))
    support = reducer.get_support()
    dvc.restrict(support)
    return X_train, X_test, dvc


def train(i, X_train, y_train, classifiers):
    warnings.filterwarnings("ignore")
    print("\n\nStarting to train...")
    r = {}
    for n, c in enumerate(classifiers):
        s = datetime.now()
        c.fit(X_train, y_train)
        time = (datetime.now() - s).total_seconds()
        r[(i, n, str(type(c)))] = {'trainTime': time}
        print("Trained:\t%d\t%s\tTime: %d" % (n, str(type(c)), time))
    return classifiers, r


def test(i, X_test, y_test, classifiers):
    print("\n\nStarting to test...")
    r = {}
    for n, c in enumerate(classifiers):
        s = datetime.now()
        score = c.score(X_test, y_test)
        time = (datetime.now() - s).total_seconds()
        r[(i, n, str(type(c)))] = {'testTime': time, 'score': score}
        print("Tested:\t%d\t%s\tTime: %d\tScore:\t%f" % (n, str(type(c)), time, score))
    return classifiers, r


def pickleClassifiersDV(i, classifiers, dvc, startTime, voting=False, extra=""):
    if voting:
        pickle.dump((classifiers, dvc),
                    open('pickled/' + sub(FILENAME_REGEX, "", str(startTime)) + " " + str(i) + 'voting' + extra + '.pickle', 'wb'))
    else:
        pickle.dump((classifiers, dvc),
                    open('pickled/' + sub(FILENAME_REGEX, "", str(startTime)) + " " + str(i) + 'voting' + extra + '.pickle', 'wb'))

def loadClassifiersDV(file=None):
    if file:
        classifiers, dvc =  pickle.load(open(file, 'rb'))
        return [(classifiers, dvc)]
    cdv = []
    for file in listdir('pickled'):
        (c, dv) = pickle.load(open('pickled/'+file, 'rb'))
        cdv.append((c, dv))
    return cdv

def trainTest(X, y, classifiers=DEFAULT_CLASSIFIERS, reduce=0, splits=10, trainsize=0.8, testsize=0.2):
    sss = StratifiedShuffleSplit(n_splits=splits, test_size=testsize, train_size=trainsize)
    results = []
    for i, (train_index, test_index) in enumerate(sss.split(X, y)):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        if reduce > 0:
            print("\n\nFeatures before reduction: " + str(X_train.shape))
            reducer = SelectKBest(score_func=f_classif, k=reduce)
            X_train = reducer.fit_transform(X_train, y_train)
            X_test = reducer.transform(X_test)
            print("\n\nFeatures after reduction: " + str(str(X_train.shape)))
            support = reducer.get_support()
        
        for classifier in classifiers:
            s = datetime.now()
            classifier.fit(X_train, y_train)
            trainTime = (datetime.now() - s).total_seconds()
            score = classifier.score(X_test, y_test)
            results.append((classifier, traintime, score))
            print("%d\t%s\tTime: %d\tScore:\t%f" % (n, str(type(classifier)), trainTime, score))
    return results

def testSavedClassifier(X, y, classifier, dv):
    X_test = dv.transform(X)
    y_test = np.array(y)

    score = classifier.score(X_test, y_test)
    results = {'score': score}
    print("Score:\t%f" % score)
    return results

def searchForParameters(classifiers, crossValidation, maxIter, X_train, y_train, X_test, y_test):
    print("\n\nStarting to train & Test...")
    for n, (c, options) in enumerate(classifiers):
        try:
            clf = RandomizedSearchCV(c, options, cv=crossValidation, n_iter=maxIter)
            s = datetime.now()
            clf.fit(X_train, y_train)
        except ValueError:
            clf = GridSearchCV(c, options, cv=crossValidation)
            s = datetime.now()
            clf.fit(X_train, y_train)
        time = (datetime.now() - s).total_seconds()
        y_true, y_pred = y_test, clf.predict(X_test)
        report = classification_report(y_true, y_pred)
        best = clf.best_params_
        print("\t%d\t%s\tTime: %d\tParams::\t%s" % (n, str(type(c)), time, str(clf.best_params_)))
        print(report)


def optimize(X, y, dv, reduce=0, splits=10, trainsize=0.8, classifiers=DEFAULT_CLASSIFIERS_ARGS, crossValidation=10,
             maxIter=20):
    warnings.filterwarnings("ignore")
    sss = StratifiedShuffleSplit(n_splits=splits, train_size=trainsize)
    startTime = datetime.now()
    for i, (train_index, test_index) in enumerate(sss.split(X, y)):
        startIterationTime = datetime.now()
        print("Starting iteration %d: " % i + str(startIterationTime))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if reduce > 0:
            (X_train, X_test, dvc) = reduceFeatures(X_train, X_test, y_train, dv, percent=reduce)

        searchForParameters(classifiers, crossValidation, maxIter, X_train, y_train, X_test, y_test)

        stopIterationTime = datetime.now()
        print("Iteration time:\t%d" % (stopIterationTime - startIterationTime).total_seconds())
        print("Total elapsed time:\t%d" % (stopIterationTime - startTime).total_seconds())


def processTweets(jsonFileName=JSON_DIR + "sarcastic/unique.json", sarcastic=True, save=True, n=None, extra=""):
    if n:
        start = datetime.now()
        tweets = tweet_iterate(jsonFileName, key="text")
        tweets = islice(tweets, n)
        feats = [(feature(tweet), sarcastic) for tweet in tweets]
        print((datetime.now() - start).total_seconds())
        if save:
            saveFeatures(feats, sarcastic, extra)
        return feats
    else:
        start = datetime.now()
        tweets = tweet_iterate(jsonFileName, key="text")
        feats = [(feature(tweet), sarcastic) for tweet in tweets]
        print((datetime.now() - start).total_seconds())
        if save:
            saveFeatures(feats, sarcastic, extra)
        return feats
    
def processReddit(jsonFileName=JSON_DIR + "sarcastic/unique.json", sarcastic=True, save=True, n=None, extra=""):
    if n:
        start = datetime.now()
        tweets = tweet_iterate(jsonFileName, key="text")
        tweets = islice(tweets, n)
        feats = [(featureReddit(tweet), sarcastic) for tweet in tweets]
        print((datetime.now() - start).total_seconds())
        if save:
            saveFeatures(feats, sarcastic, extra)
        return feats
    else:
        start = datetime.now()
        tweets = tweet_iterate(jsonFileName, key="text")
        feats = [(featureReddit(tweet), sarcastic) for tweet in tweets]
        print((datetime.now() - start).total_seconds())
        if save:
            saveFeatures(feats, sarcastic, extra)
        return feats


def saveFeatures(feats, sarcastic, extra=""):
    if sarcastic:
        sarcastic = 'sarcastic'
    else:
        sarcastic = 'serious'
    file = PICKLED_FEATS_DIR + sarcastic + 'Feats' + extra + '.pickle'
    pickle.dump(feats, open(file, 'wb'))


def loadFeatures(sarcastic, extra=""):
    if sarcastic:
        sarcastic = 'sarcastic'
    else:
        sarcastic = 'serious'
    file = PICKLED_FEATS_DIR + sarcastic + 'Feats' + extra + '.pickle'
    feats = pickle.load(open(file, 'rb'))
    return feats

def flattenFeatureDicts(features, leaveOut=None):
    featuresFlattened = []
    for feature in features:
        d = {}
        for key, value in feature.items():
            if key != leaveOut:
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        d[subkey] = subvalue
                else:
                    d[key] = value
        featuresFlattened.append(d)
    return featuresFlattened


def saveVectorizer(dv, X=None, y=None, extra=''):
    if X is not None and y is not None:
        pickle.dump((dv, X, y), open(PICKLED_FEATS_DIR + 'Xydv' + extra  + '.pickle', 'wb'))
        pickle.dump(dv, open(PICKLED_FEATS_DIR + 'dv' + extra  + '.pickle', 'wb'))
    else:
        pickle.dump(dv, open(PICKLED_FEATS_DIR + 'dv' + extra  + '.pickle', 'wb'))


def loadVectorizer(features=True, extra=''):
    if features:
        (dv, X, y) = pickle.load(open(PICKLED_FEATS_DIR + 'Xydv' + extra  + '.pickle', 'rb'))
        return dv, X, y
    else:
        dv = pickle.load(open(PICKLED_FEATS_DIR + 'dv' + extra  + '.pickle', 'rb'))
        return dv


def vectorize(features, sarcasm, fit=True, save=True, extra=''):
    dv = DictVectorizer()
    if fit:
        (X, y) = (dv.fit_transform(features), np.array(sarcasm))
        if save:
            saveVectorizer(dv, X, y, extra)
        return dv, X, y
    else:
        dv.transform(features)
        if save:
            saveVectorizer(dv, extra)
        return dv


def vectorizerTransform(dv, features, sarcasm):
    (X, y) = (dv.transform(features), np.array(sarcasm))
    return X, y


def predict(listOfString, classifierDV):
    listOfFeats = flattenFeatureDicts([feature(s) for s in listOfString])
    r = {}
    dv = classifierDV[1]
    X = dv.transform(listOfFeats)
    for i, c in enumerate(classifierDV[0]):
        prediction = c.predict(X)
        
        invert_op = getattr(c, "predict_proba", None)
        if callable(invert_op):
            preProb = c.predict_proba(X)
            r[(i,str(type(c)))] = {'prediction': prediction, 'prediction_probabilities':preProb}
        else:
            r[(i,str(type(c)))] = {'prediction': time}
    print(r)
    return r