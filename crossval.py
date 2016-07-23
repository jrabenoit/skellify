#!/usr/bin/env python 

from collections import defaultdict
import itertools
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
from sklearn.feature_selection import SelectKBest

#outer loop 5-fold CV. X = data, y = label
def oSkfCv(group_label_dict, znorm_dict, iter_n):        
    '''defaultdict creates dict with empty list inside so append works'''
    oX_train = defaultdict(list)
    oX_test = defaultdict(list)
    oy_train = defaultdict(list)
    oy_test= defaultdict(list)
    
    labels = group_label_dict
    data = znorm_dict
    
    for i in range(iter_n):
        skf = StratifiedKFold(labels[i], n_folds=5)
        for train_index, test_index in skf:
            oX_train[i].append(data[i][train_index])
            oX_test[i].append(data[i][test_index])
            oy_train[i].append(labels[i][train_index])
            oy_test[i].append(labels[i][test_index])

    return oX_train, oX_test, oy_train, oy_test

#Do 5-fold CV in inner loop
def iSkfCv(oy_train, oX_train, iter_n):
    '''Set up as a flat structure of 25 lists'''
    iX_train = defaultdict(list)
    iX_test = defaultdict(list)
    iy_train = defaultdict(list)
    iy_test= defaultdict(list)

    print(len(oy_train), len(oX_train))
    print(len(oy_train[1]), len(oX_train[1]))
    print(oy_train[1], oX_train[1])

#July 22: Matt, this is where the problem starts:
    for i in range(iter_n):
        for oX_train_, oy_train_ in zip(oX_train[i], oy_train[i]):
            iskf = StratifiedKFold(oy_train_[i], n_folds=5)
            for train_index, test_index in iskf:      
                iX_train[i].append(oX_train_[i][train_index])
                iX_test[i].append(oX_train_[i][test_index])
                iy_train[i].append(oy_train_[i][train_index])
                iy_test[i].append(oy_train_[i][test_index])
    return iX_train, iX_test, iy_train, iy_test

    
