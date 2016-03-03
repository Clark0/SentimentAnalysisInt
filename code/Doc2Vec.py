# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from sklearn import metrics

import numpy as np
import os
from pprint import pprint
from random import shuffle
from sklearn.linear_model import LogisticRegression
import time, re

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

class LabeledLineSentence(object):
    def __init__(self):
        self.files = ['/home/annamalai/Senti/UCI/imdb_labelled.txt',
                      '/home/annamalai/Senti/UCI/yelp_labelled.txt']#,
                      # '/home/annamalai/Senti/UCI/yelp_labelled.txt']
        self.sentences = []
        self.labels = []

    def to_array(self):
        self.sentences = []
        for file_num, f in enumerate(self.files):
            for ind, line in enumerate(open(f).xreadlines()):
                label = line.split('\t')[-1].strip()
                words = [re.sub('[^a-zA-Z0-9]', '', S).lower() for S in line.split('\t')[0].strip().split()]
                # print words
                # raw_input()
                self.sentences.append(LabeledSentence(words,[str(ind)+'_'+label]))
                self.labels.append(str(ind)+'_'+label)

        print 'loaded a total of {} sentences'.format(len(self.sentences))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences



# print 'loading sentences'
# sentences = LabeledLineSentence()
# print 'loaded sentences'
#
# model = Doc2Vec(min_count=1, window=4, size=50, sample=1e-4, negative=5, workers=8)
# print 'doc2vec done'
#
# # for s in sentences.to_array():
# #     print s
# #     raw_input()
#
# model.build_vocab(sentences.to_array())
# print 'built vocab done', model
#
# for epoch in range(10):
#     T0 = time.time()
#     model.train(sentences.sentences_perm())
#     print 'epoch {}, time {}'.format(epoch, time.time()-T0)
#
# print 'model trained'
#
# model.save('./vocab.d2v')
# with open ('doc_labels.txt', 'w') as fh:
#     for l in sentences.labels:
#         print>>fh, l
# print 'saved labels for docs'

model = Doc2Vec.load('./vocab.d2v')
print 'model dumped onto file system and loaded'

# try:
#     print model.docvecs['0_0']
#     raw_input()
# except:
#     pass

doc_labels = [l.strip() for l in open('doc_labels.txt').xreadlines()]
PosFvs = []
NegFvs = []
for index, fv in enumerate(model.docvecs):
    if index == 1000:
        break
    cls = int(doc_labels[index].split('_')[1])
    if cls:
        PosFvs.append(fv)
    else:
        NegFvs.append(fv)

TrainPos = PosFvs[:400]; TestPos = PosFvs[400:]
TrainNeg = NegFvs[:400]; TestNeg = NegFvs[400:]

TrainFvs = TrainPos + TrainNeg
TestFvs = TestPos + TestNeg
TrainLabels = [1 for i in xrange(400)] + [0 for i in xrange(400)]
TestLabels = [1 for i in xrange(100)] + [0 for i in xrange(100)]

Clf = GridSearchCV(LogisticRegression(max_iter=1000,n_jobs=8,class_weight='balanced'), cv=5,
                   param_grid={"C": [0.001,0.01,0.1,1,10,100]},n_jobs=8)

Clf.fit(TrainFvs,TrainLabels)
PredictedLabels = Clf.predict(TestFvs)
print '*'*100
print 'classification report'
print '-'*20
Accuracy = np.mean(PredictedLabels == TestLabels)
print "Test Set Accuracy = ", Accuracy

print(metrics.classification_report(TestLabels,
            PredictedLabels, target_names=['Neg', 'Pos']))

print "Accuracy classification score:", metrics.accuracy_score(TestLabels, PredictedLabels)
print "Hamming loss:", metrics.hamming_loss(TestLabels, PredictedLabels)
print "Average hinge loss:", metrics.hinge_loss(TestLabels, PredictedLabels)
print "Log loss:", metrics.log_loss(TestLabels, PredictedLabels)
print "F1 Score:", metrics.f1_score(TestLabels, PredictedLabels)
print "Zero-one classification loss:", metrics.zero_one_loss(TestLabels, PredictedLabels)
print '*'*100




print 'total vocab size: {} '.format(len(model.vocab.keys()))

# for k,v in model.vocab.iteritems():
#         print k
#         print v
#         raw_input()

tgt_word = ['good','bad', 'great', 'like']

for t in tgt_word:
    print 'most similar to ',t
    pprint(model.most_similar(t))


# print model.n_similarity(['android.telephony.gsm.SmsManager:sendTextMessage'],
#                     # 'android.location.Location:getLatitude'],
#                    ['android.telephony.gsm.SmsManager:sendMultipartTextMessage'])


