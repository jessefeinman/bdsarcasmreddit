from sklearn.feature_extraction import DictVectorizer
from array import array
from collections import Mapping
import numpy as np
import scipy.sparse as sp

class DictVectorizerStreaming(DictVectorizer):
    def __init__(self, dtype=np.float32, separator="=", sparse=True):
        self.dtype = dtype
        self.separator = separator
        self.sparse = sparse
        self.sort = False
        self.feature_names_ = []
        self.vocabulary_ = {}
    
    def partial_fit_transform(self, X):
        assert array("i").itemsize == 4, (
            "sizeof(int) != 4 on your platform; please report this at"
            " https://github.com/scikit-learn/scikit-learn/issues and"
            " include the output from platform.platform() in your bug report")

        dtype = self.dtype

        feature_names = self.feature_names_
        vocab = self.vocabulary_

        # Process everything as sparse regardless of setting
        X = [X] if isinstance(X, Mapping) else X

        indices = array("i")
        indptr = array("i", [0])
        # XXX we could change values to an array.array as well, but it
        # would require (heuristic) conversion of dtype to typecode...
        values = []

        # collect all the possible feature names and build sparse matrix at
        # same time
        for x in X:
            for f, v in x.items():
                if f in vocab:
                    indices.append(vocab[f])
                    values.append(dtype(v))
                else:
                    feature_names.append(f)
                    
                    vocab[f] = len(vocab)
                    indices.append(vocab[f])
                    values.append(dtype(v))
            indptr.append(len(indices))

        if len(indptr) == 1:
            raise ValueError("Sample sequence X is empty.")

        indices = np.frombuffer(indices, dtype=np.intc)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        shape = (len(indptr) - 1, len(vocab))

        result_matrix = sp.csr_matrix((values, indices, indptr),
                                      shape=shape, dtype=dtype)

        if self.sparse:
            result_matrix.sort_indices()
        else:
            result_matrix = result_matrix.toarray()

        self.feature_names_ = feature_names
        self.vocabulary_ = vocab

        return result_matrix