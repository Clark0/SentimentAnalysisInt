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

def pad_sequence(sequence, n, pad_left=False, pad_right=False, pad_symbol=None):
    if pad_left:
        sequence = chain((pad_symbol,) * (n-1), sequence)
    if pad_right:
        sequence = chain(sequence, (pad_symbol,) * (n-1))
    return sequence


def skipgrams(sequence, n, k, pad_left=False, pad_right=False, pad_symbol=None):
    sequence_length = len(sequence)
    sequence = iter(sequence)
    sequence = pad_sequence(sequence, n, pad_left, pad_right, pad_symbol)

    if sequence_length + pad_left + pad_right < k:
        # raise Exception("The length of sentence + padding(s) < skip")
        return
    if n < k:
        raise Exception("Degree of Ngrams (n) needs to be bigger than skip (k)")

    history = []
    nk = n+k

    # Return point for recursion.
    if nk < 1:
        return
    # If n+k longer than sequence, reduce k by 1 and recur
    elif nk > sequence_length:
        for ng in skipgrams(list(sequence), n, k-1):
            yield ng

    while nk > 1: # Collects the first instance of n+k length history
        history.append(next(sequence))
        nk -= 1

    # Iterative drop first item in history and picks up the next
    # while yielding skipgrams for each iteration.
    for item in sequence:
        history.append(item)
        current_token = history.pop(0)
        # Iterates through the rest of the history and
        # pick out all combinations the n-1grams
        for idx in list(combinations(range(len(history)), n-1)):
            ng = [current_token]
            for _id in idx:
                ng.append(history[_id])
            yield tuple(ng)

    # Recursively yield the skigrams for the rest of seqeunce where
    # len(sequence) < n+k
    for ng in list(skipgrams(history, n, k-1)):
        yield ng


def Tokenizer (Str):
    Str = [re.sub('[^a-zA-Z0-9]', '', S) for S in Str.split()]
    wnl = WordNetLemmatizer()
    # return [wnl.lemmatize((re.compile("[^\w']|_").sub("",S.strip()))) for S in Str.split()]
    return [wnl.lemmatize(S) for S in Str]

def SGTokenizer (Str):
    wnl = WordNetLemmatizer()
    Seq = [re.compile("[^\w']|_").sub("",S.strip()) for S in Str.split()]
    S = [wnl.lemmatize(t) for tup in skipgrams(Seq, 3, 1) for t in tup]
    # S = ' '.join(S)
    # print S
    # raw_input()
    return S


###############################################################################
# main
###############################################################################

def main():
    # if sys.argv[2] == 'svm':
    #     Clf = LinearSVC(C = 0.1, class_weight = 'balanced',max_iter=100)
    # elif sys.argv[2] == 'lr':
    #     Clf = LogisticRegression (C=0.1,max_iter=100,n_jobs=8)
    # elif sys.argv[2] == 'pa':
    #     Clf = PassiveAggressiveClassifier(C=0.1,n_iter=1,n_jobs=8,class_weight='balanced')
    # else:
    #     Clf = SGDClassifier(n_iter=1,n_jobs=8,class_weight='balanced')

    Clf = LinearSVC(C = 0.1, class_weight = 'balanced',max_iter=100)
    Clf = LogisticRegression (C=0.1,max_iter=1000,n_jobs=8,class_weight='balanced')
    Clf = GridSearchCV(LogisticRegression(max_iter=1000,n_jobs=8,class_weight='balanced'), cv=5,
                   param_grid={"C": [0.001,0.01,0.1,1,10,100]},n_jobs=8)
    # Clf = GridSearchCV(LinearSVC(C = 0.1, class_weight = 'balanced',max_iter=1000), cv=3,
    #                param_grid={"C": [0.001,0.01,0.1,1,10,100]},n_jobs=8)

    File = '/home/annamalai/Senti/UCI/amazon_cells_labelled.txt'
    Ngram = 2

    print 'Clf: {}, File: {}, ngram: {}'.format(Clf, File, Ngram)


    PosSamples = [l.split('\t')[0].strip() for l in open (File).xreadlines() if l.strip().endswith('1')]#[:100]
    NegSamples = [l.split('\t')[0].strip() for l in open (File).xreadlines() if l.strip().endswith('0')]#[:100]
    print 'loaded {} pos and {} neg samples'.format(len(PosSamples), len(NegSamples))
    X = PosSamples + NegSamples
    y = [1 for _ in xrange(len(PosSamples))] + [-1 for _ in xrange (len(NegSamples))]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.1,
                                                        random_state=random.randint(0,100))
    print '# TrainLabels', len(y_train)
    print '# TestLabels', len(y_test)

    print 'performing CVectorizer'
    CVectorizer = CountVectorizer(lowercase = True,
                                  stop_words='english',
                                  # token_pattern='(?u)\b\w\w+\b',
                                  # tokenizer = SGTokenizer,
                                  tokenizer = Tokenizer,
                                  ngram_range=(1,2),
                                  dtype=np.float64,
                                  decode_error = 'ignore',
                                  max_df=0.8)
    print 'performing TfidfTransformer and Normalizer'
    # TFIDFTransformer = TfidfTransformer()
    normalizer = Normalizer()
    print 'creating Train and Test FVs'
    T0 = time()
    TrainFVs = CVectorizer.fit_transform(X_train)
    TestFVs = CVectorizer.transform(X_test)
    print 'feat ext time', time() - T0

    # TrainFVs = TFIDFTransformer.fit_transform(TrainFVs)
    # TestFVs = TFIDFTransformer.transform(TestFVs)

    TrainFVs = normalizer.fit_transform(TrainFVs)
    TestFVs = normalizer.transform(TestFVs)

    print 'Trai/test split'
    print TrainFVs.shape
    print TestFVs.shape
    # raw_input('hit any key...')

    print 'training classifier with train samples shape:', TrainFVs.shape
    T0 = time()
    # memory_dump('before_train_mem.txt')
    Model = Clf.fit (TrainFVs, y_train) # re-train on current training set (daily)
    print 'batch fitted'
    print 'training time', time() - T0
    # memory_dump('after_train_mem.txt')

    print 'testing classifier with test samples shape:', TestFVs.shape
    T0 = time()
    # memory_dump('before_test_mem.txt')
    PredictedLabels = Clf.predict(TestFVs)
    print 'testing time', time() - T0
    # memory_dump('after_test_mem.txt')

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
    # print Vocab[:100]
    # raw_input()
    try:
        FeatureImportances = Clf.coef_[0]
    except:
        FeatureImportances = Clf.best_estimator_.coef_[0]

    print FeatureImportances.shape
    raw_input()
    PosTopFeatureIndices = FeatureImportances.argsort()[-100:][::-1]
    NegTopFeatureIndices = FeatureImportances.argsort()[:100][::-1]
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
        # print TestFV
        # print TestFV.shape
        CurTestFV = np.array(AllFVsTimesW[Ind].toarray())
        CurTestFV = CurTestFV.transpose()
        CurTestFV = CurTestFV.reshape(CurTestFV.shape[0],)
        # print CurTestFV.shape
        # raw_input()
        PosTopFeatureIndices = CurTestFV.argsort()[-2:][::-1]
        NegTopFeatureIndices = CurTestFV.argsort()[:2][::-1]
        PosFeatImps= CurTestFV.argsort()[-2:]
        NegFeatImps = CurTestFV.argsort()[:2]
        Tmp = AllFVsTimesW[Ind].todense()
        Tmp = np.sort(Tmp)
        # print PosTopFeatureIndices, AllFVsTimesW[Ind].todense().argsort(), Tmp
        # print NegTopFeatureIndices, NegFeatImps
        if y_test[Ind] == 1:
            print 'top postive feats:', colored(', '.join(['['+Vocab[PosFIndex]+']' for PosFIndex in PosTopFeatureIndices]), 'green')

        else:
            print 'top negative feats: ', colored(', '.join (['['+Vocab[NegFIndex]+']' for NegFIndex in NegTopFeatureIndices]), 'red')
        Ind += 1
        raw_input()
    # AvgFV = AllFVsTimesW.mean(axis=0)
    # AvgFV = AvgFV.view(dtype=np.float64).reshape(AvgFV.shape[1],-1)
    # AvgFV = np.array(AvgFV).reshape(-1,)
    # TopFeatureIndices = AvgFV.argsort()[-100:][::-1]
    # for FIndex in TopFeatureIndices:
    #     print Vocab[FIndex]

if __name__ == '__main__':
    main()
