import json
import pickle
from pprint import pprint
import re
from os import listdir, SEEK_END
import datetime
import random
from sklearn.linear_model import LogisticRegression
import nlp
import ml
import numpy as np
from dvs import DictVectorizerPartial
import scipy
import pyspark
from pyspark.sql import SQLContext

def filterComments(generator):
    import nlp
    import ml
    pop = []
    for comment in generator:
        try:
            text = comment['body'].lower()
            if 10 <= len(text) <= 120:
                if sarcasm_re.search(text) is not None:
                    yield (True, ml.flattenDict(nlp.feature(comment['body'], nlp.cleanTokensReddit)))
                else:
                    pop.append(comment['body'])
                    if len(pop) == 1300:
                        yield (False, ml.flattenDict(nlp.feature(random.choice(pop), nlp.cleanTokensReddit)))
                        pop = []
        except:
            pass

def getVocab(gen):
    for sarc, features in gen:
        for key in features:
            yield key

def vectorize(gen, dv):
    blocksize = 100000
    sarclst = []
    featlst = []
    for sarc, features in gen:
        sarclst.append(sarc)
        featlst.append(features)
        if len(sarclst) == blocksize:
            yield (sarclst, dv.transform(featlst))
            sarclst = []
            featlst = []
    yield (sarclst, dv.transform(featlst))

    
if __name__=='__main__':
    list_re = [
        r"(\/sarcasm)",
        r"(&lt;\/?sarcasm&gt)",
        r"(#sarcasm)",
        r"(\s*\/s\s*$)"
              ]
    sarcasm_re = re.compile('|'.join(list_re))

    sc = pyspark.SparkContext('local[*]')
    sqlContext = SQLContext(sc)

    df_rdd = sqlContext.read.format('json').load('/scratch/redditSarcasm/*')
    rdd = df_rdd.rdd
    features = rdd.mapPartitions(filterComments)
    vocab = dict(features.mapPartitions(getVocab).distinct().zipWithIndex().collect())
    dvp = DictVectorizerPartial(vocab=vocab)
    vdvp = lambda gen: vectorize(gen, dvp)
    csrs = features.mapPartitions(vdvp)
    (sarclst, matricies) = zip(*csrs.collect())
    sarclst = list(sarclst)
    matricies = list(matricies)
    y = np.array(reduce(lambda a,b: a+b, sarclst))
    X = scipy.sparse.vstack(matricies, format='csr')
    results = ml.trainTest(X,
                           y,
                           classifiers=[LogisticRegression(n_jobs=-1)],
                           reduce=0,
                           splits=4,
                           trainsize=0.8,
                           testsize=0.2)
    pickle.dump(results, open('/user/jfeinma00/trained-logistic-classifier.pickle', 'wb'))    