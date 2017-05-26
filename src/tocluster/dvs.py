from sklearn.feature_extraction import DictVectorizer
from array import array
from collections import Mapping
import numpy as np
import scipy.sparse as sp

class DictVectorizerPartial(DictVectorizer):
    def __init__(self, dtype=np.float32, separator="=", sparse=True, vocab={}, feature_names=[]):
        self.dtype = dtype
        self.separator = separator
        self.sparse = sparse
        self.sort = False
        self.feature_names_ = feature_names
        self.vocabulary_ = vocab
    
    def fit(self, x=[]):
        self.partial_fit(self, x=[], vocab={})

    def partial_fit(self, x=[], vocab=None):
        if not vocab:
            vocab = self.vocabulary_

        for xp in x:
            for key in xp:
                vocab.setdefault(key, len(vocab))
        self.vocabulary_ = vocab
        self.feature_names_ = sorted(vocab, key=vocab.get)
    
    def _fit_transform(self, x, y, fit, vocab):
        indptr = [0]
        indices = []
        X = []
        for xp in x:
            for key, val in xp.items():
                if fit:
                    index = vocab.setdefault(key, len(vocab))
                    indices.append(index)
                    X.append(self.dtype(val))
                elif key in vocab:
                    indices.append(vocab[key])
                    X.append(self.dtype(val))
            indptr.append(len(indices))
        self.vocabulary_ = vocab
        self.feature_names_ = sorted(vocab, key=vocab.get)
        N = len(vocab)-1
        M = len(indptr)-1
        if y:
            return sp.csr_matrix((X, indices, indptr), dtype=self.dtype, shape=(M, N)), y
        else:
            return sp.csr_matrix((X, indices, indptr), dtype=self.dtype, shape=(M, N))
    
    def transform(self, x, y=None):
        return self._fit_transform(x, y, fit=False, vocab=self.vocabulary_)

    def partial_fit_transform(self, x, y=None, vocab={}):
        if vocab:
            return self._fit_transform(x, y, fit=True, vocab=vocab)
        else:
            return self._fit_transform(x, y, fit=True, vocab=self.vocabulary_)
        
    def fit_transform(self, x, y=None):
        return self._fit_transform(x, y, fit=True, vocab={})
