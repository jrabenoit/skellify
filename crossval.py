#!/usr/bin/env python 

from collections import defaultdict
import pprint
import itertools
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
from sklearn.feature_selection import SelectKBest

#outer loop 5-fold CV. X = data, y = label
def oSkfCv(group_label_dict, znorm_dict, iter_n, concat_subjects_dict):        
    '''defaultdict creates dict with empty list inside so append works'''
    oX_train = defaultdict(list)
    oX_test = defaultdict(list)
    oy_train = defaultdict(list)
    oy_test= defaultdict(list)
    
    train_index_outer = defaultdict(list)
    test_index_outer = defaultdict(list)
    train_index_files= defaultdict(list)
    test_index_files= defaultdict(list)
    
    
    labels = group_label_dict
    data = znorm_dict
    files = concat_subjects_dict
    
    for i in range(iter_n):
        skf = StratifiedKFold(labels[i], n_folds=5)
        for train_index, test_index in skf:
            train_index_outer[i].append(train_index)
            test_index_outer[i].append(test_index)
            oX_train[i].append(data[i][train_index])
            oX_test[i].append(data[i][test_index])
            oy_train[i].append(labels[i][train_index])
            oy_test[i].append(labels[i][test_index])          

    for i in range(iter_n):
        for j in range(5):
            list_files_fold_train = []
            list_files_fold_test = []
            for k in range(len(train_index_outer[i][j])):
                list_files_fold_train.append(files[i][train_index_outer[i][j][k]])
            train_index_files[i] += [list_files_fold_train]   
            for k in range(len(test_index_outer[i][j])):
                list_files_fold_test.append(files[i][test_index_outer[i][j][k]])
            test_index_files[i] += [list_files_fold_test]    
    
#    print('CONCAT_SUBJECTS_DICT')
#    pprint.pprint(concat_subjects_dict[0])
#    print('TRAIN_INDEX')
#    pprint.pprint(train_index_outer[0])        
#    print('TRAIN_INDEX_FILES')
#    pprint.pprint(train_index_files[0])
    return oX_train, oX_test, oy_train, oy_test, train_index_outer, test_index_outer, train_index_files, test_index_files

#Do 5-fold CV in inner loop
def iSkfCv(oy_train, oX_train, iter_n):
    '''Set up as a flat structure of 25 lists'''
    iX_train = defaultdict(list)
    iX_test = defaultdict(list)
    iy_train = defaultdict(list)
    iy_test= defaultdict(list)

    train_index_inner = defaultdict(list)
    test_index_inner = defaultdict(list)
#Some subjects will not be listed here because they are in the holdout set.
    for i in range(iter_n):
        for oX_train_, oy_train_ in zip(oX_train[i], oy_train[i]):
            iskf = StratifiedKFold(oy_train_, n_folds=5)
            for train_index, test_index in iskf:      
                train_index_inner[i].append(train_index)
                test_index_inner[i].append(test_index)
                iX_train[i].append(oX_train_[train_index])
                iX_test[i].append(oX_train_[test_index])
                iy_train[i].append(oy_train_[train_index])
                iy_test[i].append(oy_train_[test_index])
    return iX_train, iX_test, iy_train, iy_test, train_index_inner, test_index_inner

    
