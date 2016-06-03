#!/usr/bin/env python 

from sklearn.feature_selection import SelectKBest, f_classif
import copy
import numpy as np

#Run feature selection. Data here need to be transformed because they'll be used in the ML step.
def SelKBest1(X_train, X_test, y_train, y_test):
    fX_train = copy.copy(X_train)
    fX_test = copy.copy(X_test)
    fy_train = copy.copy(y_train)
    fy_test = copy.copy(y_test)
    skb = SelectKBest(f_classif, k=3)
    for i in range(0,len(X_train)):
        fX_train[i] = skb.fit_transform(fX_train[i], fy_train[i])
        fX_test[i] = skb.transform(fX_test[i])    
    return fX_train, fX_test, fy_train, fy_test

feat_sel_dict = {}#'SelKBest1':SelKBest1}

def SelKBest_maker(f_classif, k):
    return lambda X_train, X_test, y_train, y_test: SelKBest_base(X_train, X_test, y_train, y_test, k)
    
for k in range(10,101,10):
    feat_sel_dict['SelKBest'+str(k)] = SelKBest_maker(f_classif, k)
    
def SelKBest_base(X_train, X_test, y_train, y_test, k=10):
    '''Leave the copying alone here'''
    fX_train = copy.copy(X_train)
    fX_test = copy.copy(X_test)
    fy_train = copy.copy(y_train)
    fy_test = copy.copy(y_test)
    skb = SelectKBest(f_classif, k=k)
    for i in range(0,len(X_train)):
        fX_train[i] = skb.fit_transform(fX_train[i], fy_train[i])
        fX_test[i] = skb.transform(fX_test[i])  
    return fX_train, fX_test, fy_train, fy_test

################################################################################

def SelKBest1Final(X_train, X_test, y_train, y_test):
    fX_train = copy.copy(X_train)
    fX_test = copy.copy(X_test)
    fy_train = copy.copy(y_train)
    fy_test = copy.copy(y_test)
    skb = SelectKBest(f_classif, k=3)
    skb.fit(fX_train, fy_train)
    fX_train = skb.transform(fX_train)
    fX_test = skb.transform(fX_test)    
    return fX_train, fX_test, fy_train, fy_test
    
feat_sel_dict_final = {}#'SelKBest1':SelKBest1Final}

def SelKBest_maker_final(f_classif, k):
    return lambda X_train, X_test, y_train, y_test: SelKBest_base_final(X_train, X_test, y_train, y_test, k)
    
for k in range(10,101,10):
    feat_sel_dict_final['SelKBest'+str(k)] = SelKBest_maker_final(f_classif, k)
    
def SelKBest_base_final(X_train, X_test, y_train, y_test, k=10):
    '''Leave the copying alone here'''
    fX_train = copy.copy(X_train)
    fX_test = copy.copy(X_test)
    fy_train = copy.copy(y_train)
    fy_test = copy.copy(y_test)
    skb = SelectKBest(f_classif, k=k)
    skb.fit(fX_train, fy_train)
    fX_train = skb.transform(fX_train)
    fX_test = skb.transform(fX_test)  
    return fX_train, fX_test, fy_train, fy_test
    
