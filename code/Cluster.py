import sys
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from time import time
from sklearn.svm import LinearSVC
from scipy import *
from sklearn.cross_validation import train_test_split
import random
import gc
import cPickle
# from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier, LogisticRegression

from sklearn.cluster import MiniBatchKMeans, KMeans, Birch, AffinityPropagation, DBSCAN


def NewLineTokenizer (Str):
    #you may use your own tokenizer
    return Str.split('\n')


###############################################################################
# main
###############################################################################
@profile
def main():
    if sys.argv[2] == 'kmeans':
        Cl = MiniBatchKMeans(init='k-means++', n_clusters=40, batch_size=100,
                      n_init=10, max_no_improvement=10, verbose=0,
                      random_state=0)
    elif sys.argv[2] == 'birch':
        Cl = Birch (n_clusters=40)
    elif sys.argv[2] == 'ap':
        Cl = AffinityPropagation()
    else:
        Cl = DBSCAN()

    PosFolder = '/home/annamalai/Mahin/malware'
    print 'Cl: {}, Pos: {}, ngram: {}'.format(Cl, PosFolder, sys.argv[1])


    PosSamples = [os.path.join (PosFolder, f) for f in os.listdir(PosFolder)]#[:100]
    X = PosSamples

    print 'performing CVectorizer'
    CVectorizer = CountVectorizer(input = u'filename',
                                  lowercase = False,
                                  token_pattern = None,
                                  tokenizer = NewLineTokenizer,
                                  ngram_range=(1, int(sys.argv[1])),
                                  dtype=np.float64,
                                  decode_error = 'ignore')
    print 'performing TfidfTransformer and Normalizer'
    TFIDFTransformer = TfidfTransformer()
    normalizer = Normalizer()
    print 'creating FVs'
    T0 = time()
    TrainFVs = CVectorizer.fit_transform(X)
    print 'feat ext time', time() - T0

    TrainFVs = TFIDFTransformer.fit_transform(TrainFVs)

    print TrainFVs.shape
    raw_input('hit any key...')

    print 'clustering with samples shape:', TrainFVs.shape
    T0 = time()
    # memory_dump('before_train_mem.txt')
    print '*'*100
    Model = Cl.fit (TrainFVs) # re-train on current training set (daily)
    print 'cluster labels: ', Cl.labels_
    print 'batch fitted'
    print 'clustering time', time() - T0
    print '*'*100

if __name__ == '__main__':
    main()