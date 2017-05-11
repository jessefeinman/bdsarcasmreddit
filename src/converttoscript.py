
# coding: utf-8

# In[11]:

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


# In[2]:

list_re = [
    r"(\/sarcasm)",
    r"(&lt;\/?sarcasm&gt)",
    r"(#sarcasm)",
    r"(\s*\/s\s*$)"
          ]
sarcasm_re = re.compile('|'.join(list_re))


# In[3]:

import pyspark
sc = pyspark.SparkContext('local[*]')


# In[4]:

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

df_rdd = sqlContext.read.format('json').load('test.json')
rdd = df_rdd.rdd


# In[5]:

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
                    if len(pop) == 1:#set to 1300
                        yield (False, ml.flattenDict(nlp.feature(random.choice(pop), nlp.cleanTokensReddit)))
                        pop = []
        except:
            pass
                        
features = rdd.mapPartitions(filterComments)


# In[6]:

def getVocab(gen):
    for sarc, features in gen:
        for key in features:
            yield key

vocab = dict(features.mapPartitions(getVocab).distinct().zipWithIndex().collect())


# In[7]:

dvp = DictVectorizerPartial(vocab=vocab)
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
vdvp = lambda gen: vectorize(gen, dvp)

csrs = features.mapPartitions(vdvp)


# In[8]:

(sarclst, matricies) = zip(*csrs.collect())
sarclst = list(sarclst)
matricies = list(matricies)


# In[9]:

y = np.array(reduce(lambda a,b: a+b, sarclst))
X = scipy.sparse.vstack(matricies, format='csr')


# In[12]:

results = ml.trainTest(X,
                       y,
                       classifiers=[LogisticRegression(n_jobs=-1)],
                       reduce=0,
                       splits=4,
                       trainsize=0.8,
                       testsize=0.2)
pickle.dump(results, open('trained-logistic-classifier.pickle', 'wb'))    


# In[ ]:



