import json
import pickle
from pprint import pprint
import re
from os import listdir, SEEK_END
import datetime
import random
from scipy.sparse import vstack
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import nlp
import ml
import numpy as np
from dvs import DictVectorizerPartial
import scipy
import pyspark
from pyspark.sql import SQLContext
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, NaiveBayes
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics

def filterComments(generator):
    import nlp
    import ml
    list_re = [
        r"(\/sarcasm)",
        r"(&lt;\/?sarcasm&gt)",
        r"(#sarcasm)",
        r"(\s*\/s\s*$)"
              ]
    sarcasm_re = re.compile('|'.join(list_re))
    pop = []
    for comment in generator:
        try:
            text = comment['body'].lower()
            if 10 <= len(text) <= 120:
                if sarcasm_re.search(text) is not None:
                    yield (True, ml.flattenDict(nlp.feature(comment['body'], nlp.cleanTokensReddit)))
                else:
                    pop.append(comment['body'])
                    if len(pop) == 1800:
                        yield (False, ml.flattenDict(nlp.feature(random.choice(pop), nlp.cleanTokensReddit)))
                        pop = []
        except:
            pass

def getVocab(gen):
    for sarc, features in gen:
        for key in features:
            yield key

def v(gen, dv):
    from pyspark.mllib.regression import LabeledPoint
    for s, f in gen:
        yield LabeledPoint(s, dv.transform([f]).tocsc().transpose())
			
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

def train(gen):
    for sarclst, matrix in gen:
        y = np.array(sarclst)
        X = matrix
        result = ml.trainTest(X, y, classifiers=[LogisticRegression(n_jobs=-1)], reduce=0, splits=4, trainsize=0.8, testsize=0.2)
        print result
        yield result

def gerkin(gen):
    for result in gen:
        yield pickle.dumps(result)

if __name__=='__main__':
    sc = pyspark.SparkContext()
    sqlContext = SQLContext(sc)
    df_rdd = sqlContext.read.format('json').load('/scratch/redditSarcasm/*')
    print "Read df"
    rdd = df_rdd.rdd
    print "made rdd"
    print "Reducing and transforming"
    features = rdd.mapPartitions(filterComments)
    print "Done reducing and transforming"
    vocab = dict(features.mapPartitions(getVocab).distinct().zipWithIndex().collect())
    print "Gathering Vocab"
    dvp = DictVectorizerPartial(vocab=vocab)

    if True:
        vdvp = lambda gen: vectorize(gen, dvp)
        csrs = features.mapPartitions(vdvp)
        train, test = csrs.randomSplit([0.9,0.1])
        mb = MultinomialNB()
        for sarc, mat in train.collect():
            mb.partial_fit(mat,sarc,classes=np.array([True,False]))
        lsm = test.collect()
        ls, lm = zip(*lsm)
        testsarc = [item for sublist in ls for item in sublist]
        testmatrix = vstack(lm)
        score = mb.score(testsarc, testmatrix)
        print "Score:\t"+str(score)
        sc.parallelize([pickle.dumps(mb)]).saveAsTextFile('/user/jfeinma00/mb'+str(score)) 
    
	if False:
		vdvp = lambda gen: vectorize(gen, dvp)
		csrs = features.mapPartitions(vdvp)
		print "Collecting and saving X y"
		trained = csrs.mapPartitions(train)
		dill = trained.mapPartitions(gerkin)
		dill.saveAsTextFile('/user/jfeinma00/logistic')
    
    if False:
        topoints = lambda gen: v(gen, dvp)
        labeledpoints = features.mapPartitions(topoints)
        print "Created labeledpoints"
        training, test = labeledpoints.randomSplit([0.8,0.2])
        print "Split data into test and train"
        model = LogisticRegressionWithLBFGS.train(training, iterations=3)
        #model = NaiveBayes.train(training)
        print "Trained model"
        predictionAndLabels = test.map(lambda lp: (float(model.predict(lp.features)), lp.label))
        print "Got prediction values for test set"
        accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / test.count()
        print('model accuracy {}'.format(accuracy))
        metrics = BinaryClassificationMetrics(predictionAndLabels)
        print("Area under PR = %s" % metrics.areaUnderPR)
        print("Area under ROC = %s" % metrics.areaUnderROC)
        model.save(sc, '/user/jfeinma00/lr%s'%str(accuracy))
    
    sc.p(vocab.items()).saveAsTextFile('/user/jfeinma00/dvp') 
