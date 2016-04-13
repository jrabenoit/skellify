#!/usr/bin/env python 

import itertools
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
from sklearn.feature_selection import SelectKBest

#outer loop 5-fold CV. X = data, y = label
def oSkfCv(labels, data):        
    oX_train = []
    oX_test = []
    oy_train = []
    oy_test = []
    skf = StratifiedKFold(labels, n_folds=5)
    for train_index, test_index in skf:
        oX_train.append(data[train_index])
        oX_test.append(data[test_index])
        oy_train.append(labels[train_index])
        oy_test.append(labels[test_index])

    return oX_train, oX_test, oy_train, oy_test

#Do 5-fold CV in inner loop
def iSkfCv(oy_train, oX_train):
    '''Set up as a flat structure of 25 lists'''
    iX_train = []
    iX_test = []
    iy_train = []
    iy_test = []
    for oX_train_, oy_train_ in zip(oX_train, oy_train):
        iskf = StratifiedKFold(oy_train_, n_folds=5)
        for train_index, test_index in iskf:      
            iX_train.append(oX_train_[train_index])
            iX_test.append(oX_train_[test_index])
            iy_train.append(oy_train_[train_index])
            iy_test.append(oy_train_[test_index])

    return iX_train, iX_test, iy_train, iy_test
