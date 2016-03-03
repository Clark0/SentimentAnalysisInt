import sys
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from time import time
from sklearn.svm import LinearSVC, SVC
from scipy import *
from sklearn.cross_validation import train_test_split
import random
import gc
import cPickle
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier, LogisticRegression
import re
from itertools import chain, combinations
import copy
from nltk.util import ngrams
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import scipy.sparse as ssp
from sklearn.grid_search import GridSearchCV
from termcolor import colored
from sklearn.feature_selection import SelectKBest, chi2, f_classif



def Tokenizer (Str):
    Str = [re.sub('[^a-zA-Z0-9]', '', S) for S in Str.split()]
    wnl = WordNetLemmatizer()
    return [wnl.lemmatize(S) for S in Str]



###############################################################################
# main
###############################################################################

def main():
    Clf = GridSearchCV(LogisticRegression(max_iter=1000,n_jobs=8,class_weight='balanced'), cv=5,
                   param_grid={"C": [0.001,0.01,0.1,1,10,100]},n_jobs=8)
    # Clf = GridSearchCV(LinearSVC(C = 0.1, class_weight = 'balanced',max_iter=1000), cv=3,
    #                param_grid={"C": [0.001,0.01,0.1,1,10,100]},n_jobs=8)

    File = sys.argv[1]
    if 'imdb' == File:
        File = '/home/annamalai/Senti/UCI/imdb_labelled.txt'
    elif 'amazon' == File:
        File = '/home/annamalai/Senti/UCI/amazon_cells_labelled.txt'
    else:
        File = '/home/annamalai/Senti/UCI/yelp_labelled.txt'
    print 'Clf: {}, Src File: {}'.format(Clf, File)


    PosSamples = [l.split('\t')[0].strip() for l in open (File).xreadlines() if l.strip().endswith('1')]#[:100]
    NegSamples = [l.split('\t')[0].strip() for l in open (File).xreadlines() if l.strip().endswith('0')]#[:100]
    print 'loaded {} pos and {} neg samples'.format(len(PosSamples), len(NegSamples))
    raw_input('hit any key...')
    X = PosSamples + NegSamples
    y = [1 for _ in xrange(len(PosSamples))] + [-1 for _ in xrange (len(NegSamples))]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.1,
                                                        random_state=random.randint(0,100))

    print 'performing count vectorizing'
    CVectorizer = CountVectorizer(lowercase = True,
                                  stop_words='english',
                                  tokenizer = Tokenizer,
                                  ngram_range=(1,2),
                                  dtype=np.float64,
                                  decode_error = 'ignore',
                                  max_df=0.8)
    print 'performing Normalization'
    # TFIDFTransformer = TfidfTransformer()
    normalizer = Normalizer()
    print 'creating Train and Test FVs'
    T0 = time()
    TrainFVs = CVectorizer.fit(X_train)
    FromDocVocab = CVectorizer.get_feature_names()
    wnl = WordNetLemmatizer()
    LexVocab = [wnl.lemmatize(l.strip()) for l in open('/home/annamalai/Senti/HuLiuLexicon/positive-words.txt').xreadlines()] +\
               [wnl.lemmatize(l.strip()) for l in open('/home/annamalai/Senti/HuLiuLexicon/negative-words.txt').xreadlines()]
    FinalVocab = set(FromDocVocab + LexVocab);FinalVocab = list(FinalVocab);FinalVocab.sort()

    CVectorizer = CountVectorizer(lowercase = True,
                                  stop_words='english',
                                  vocabulary=FinalVocab,
                                  tokenizer = Tokenizer,
                                  ngram_range=(1,2),
                                  dtype=np.float64,
                                  decode_error = 'ignore',
                                  max_df=0.8,
                                  min_df=0.1)

    TrainFVs = CVectorizer.transform(X_train)
    TestFVs = CVectorizer.transform(X_test)
    print 'feat ext time', time() - T0

    # TrainFVs = TFIDFTransformer.fit_transform(TrainFVs)
    # TestFVs = TFIDFTransformer.transform(TestFVs)

    TrainFVs = normalizer.transform(TrainFVs)
    TestFVs = normalizer.transform(TestFVs)

    print 'Trai/test split'
    print TrainFVs.shape
    print TestFVs.shape
    # raw_input('hit any key...')

    print 'training classifier with train samples shape:', TrainFVs.shape
    T0 = time()
    Clf.fit (TrainFVs, y_train)
    print 'best model: {}'.format(Clf.best_estimator_)
    print 'training time', time() - T0

    print 'testing classifier with test samples shape:', TestFVs.shape
    T0 = time()
    PredictedLabels = Clf.predict(TestFVs)
    print 'testing time', time() - T0

    print '*'*100
    print 'classification report'
    print '-'*20
    Accuracy = np.mean(PredictedLabels == y_test)
    print "Test Set Accuracy = ", Accuracy

    print(metrics.classification_report(y_test,
                PredictedLabels, target_names=['Neg', 'Pos']))

    print "Accuracy classification score:", metrics.accuracy_score(y_test, PredictedLabels)
    print "Hamming loss:", metrics.hamming_loss(y_test, PredictedLabels)
    print "Average hinge loss:", metrics.hinge_loss(y_test, PredictedLabels)
    print "Log loss:", metrics.log_loss(y_test, PredictedLabels)
    print "F1 Score:", metrics.f1_score(y_test, PredictedLabels)
    print "Zero-one classification loss:", metrics.zero_one_loss(y_test, PredictedLabels)
    print '*'*100

    Vocab = CVectorizer.get_feature_names()
    try:
        FeatureImportances = Clf.coef_[0]
    except:
        FeatureImportances = Clf.best_estimator_.coef_[0]

    print FeatureImportances.shape
    raw_input()
    PosTopFeatureIndices = FeatureImportances.argsort()[-30:][::-1]
    NegTopFeatureIndices = FeatureImportances.argsort()[:30][::-1]
    for PosFIndex, NegFIndex in zip(PosTopFeatureIndices, NegTopFeatureIndices):
                print Vocab[PosFIndex], '+-', Vocab[NegFIndex]


    FeatureImportancesSparseArray = ssp.lil_matrix((TestFVs.shape[1],TestFVs.shape[1]))
    FeatureImportancesSparseArray.setdiag(FeatureImportances)

    AllFVsTimesW = TestFVs*FeatureImportancesSparseArray
    print AllFVsTimesW.shape

    Ind = 0
    for TestFV in TestFVs:
        if PredictedLabels[Ind] != y_test[Ind]:
            Ind += 1
            continue
        if len(X_test[Ind].split()) < 5:
            Ind += 1
            continue
        print 'Sample: {}, actual label: {}'.format(X_test[Ind], y_test[Ind])
        CurTestFV = np.array(AllFVsTimesW[Ind].toarray())
        CurTestFV = CurTestFV.transpose()
        CurTestFV = CurTestFV.reshape(CurTestFV.shape[0],)
        PosTopFeatureIndices = CurTestFV.argsort()[-2:][::-1]
        NegTopFeatureIndices = CurTestFV.argsort()[:2][::-1]
        PosFeatImps= CurTestFV.argsort()[-2:]
        NegFeatImps = CurTestFV.argsort()[:2]
        Tmp = AllFVsTimesW[Ind].todense()
        Tmp = np.sort(Tmp)
        if y_test[Ind] == 1:
            print 'top postive feats:', colored(', '.join(['['+Vocab[PosFIndex]+']' for PosFIndex in PosTopFeatureIndices]), 'green')

        else:
            print 'top negative feats: ', colored(', '.join (['['+Vocab[NegFIndex]+']' for NegFIndex in NegTopFeatureIndices]), 'red')
        Ind += 1
        raw_input()

if __name__ == '__main__':
    main()
