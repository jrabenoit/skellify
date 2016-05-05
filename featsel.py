#!/usr/bin/env python 

from sklearn.feature_selection import SelectKBest
import copy
import numpy as np

#Run feature selection. Data here need to be transformed because they'll be used in the ML step.
def SelKBest1(X_train, X_test, y_train, y_test):
    fX_train = copy.copy(X_train)
    fX_test = copy.copy(X_test)
    fy_train = copy.copy(y_train)
    fy_test = copy.copy(y_test)
    skb = SelectKBest(k=10)
    for i in range(0,len(X_train)):
        fX_train[i] = skb.fit_transform(fX_train[i], fy_train[i])
        fX_test[i] = skb.transform(fX_test[i])    
    return fX_train, fX_test, fy_train, fy_test

def SelKBest2(X_train, X_test, y_train, y_test):
    fX_train = copy.copy(X_train)
    fX_test = copy.copy(X_test)
    fy_train = copy.copy(y_train)
    fy_test = copy.copy(y_test)
    skb = SelectKBest(k=20)
    for i in range(0,len(X_train)):
        fX_train[i] = skb.fit_transform(fX_train[i], fy_train[i])
        fX_test[i] = skb.transform(fX_test[i])    
    return fX_train, fX_test, fy_train, fy_test

def SelKBest3(X_train, X_test, y_train, y_test):
    fX_train = copy.copy(X_train)
    fX_test = copy.copy(X_test)
    fy_train = copy.copy(y_train)
    fy_test = copy.copy(y_test)
    skb = SelectKBest(k=100)
    for i in range(0,len(X_train)):
        fX_train[i] = skb.fit_transform(fX_train[i], fy_train[i])
        fX_test[i] = skb.transform(fX_test[i])    
    return fX_train, fX_test, fy_train, fy_test

feat_sel_dict = {
               #  'SelKBest1':SelKBest1,
               #  'SelKBest2':SelKBest2,
                 'SelKBest3':SelKBest3 
                }
'''
def SelKBest_maker(k):
    return lambda fX_train, fX_test, fy_train, fy_test: LSvm_base(X_train, X_test, y_train, y_test, k)
    
for k in [10,50,10]:
    ml_func_dict['LSvmL1_C'+str(C).replace(".","p")] = LSvm_maker(penalty='l1',loss=None,dual=False,C=C)
    ml_func_dict['LSvmL2_C'+str(C).replace(".","p")] = LSvm_maker(penalty='l2',loss='hinge',dual=True,C=C)
    
def SelKBest_base(fX_train, fX_test, fy_train, fy_test, penalty='l1', loss=None, dual=False, C=1.0):
    # Note: c argument not actually used yet
    lX_train = copy.copy(fX_train)
    lX_test = copy.copy(fX_test)
    ly_train = copy.copy(fy_train)
    ly_test = copy.copy(fy_test)
    for i in range(0,len(fX_train)):
        if not loss:
            lsvm = svm.LinearSVC(penalty=penalty, dual=dual, C=C)
        else:
            lsvm = svm.LinearSVC(penalty=penalty, dual=dual, loss=loss)
        lsvm.fit(lX_train[i], ly_train[i])
        lX_train[i] = lsvm.score(lX_train[i], ly_train[i])
        lX_test[i] = lsvm.score(lX_test[i], ly_test[i])
    return lX_train, lX_test
'''
################################################################################

def SelKBest1Final(X_train, X_test, y_train, y_test):
    fX_train = copy.copy(X_train)
    fX_test = copy.copy(X_test)
    fy_train = copy.copy(y_train)
    fy_test = copy.copy(y_test)
    skb = SelectKBest(k=10)
    skb.fit(fX_train, fy_train)
    fX_train = skb.transform(fX_train)
    fX_test = skb.transform(fX_test)    
    return fX_train, fX_test, fy_train, fy_test

def SelKBest2Final(X_train, X_test, y_train, y_test):
    fX_train = copy.copy(X_train)
    fX_test = copy.copy(X_test)
    fy_train = copy.copy(y_train)
    fy_test = copy.copy(y_test)
    skb = SelectKBest(k=20)
    fX_train = skb.fit_transform(fX_train, fy_train)
    fX_test = skb.transform(fX_test)    
    return fX_train, fX_test, fy_train, fy_test
    
def SelKBest3Final(X_train, X_test, y_train, y_test):
    fX_train = copy.copy(X_train)
    fX_test = copy.copy(X_test)
    fy_train = copy.copy(y_train)
    fy_test = copy.copy(y_test)
    skb = SelectKBest(k=100)
    skb.fit(fX_train, fy_train)
    fX_train = skb.transform(fX_train)
    fX_test = skb.transform(fX_test)    
    return fX_train, fX_test, fy_train, fy_test
    
feat_sel_dict_final = {
                   #    'SelKBest1Final':SelKBest1Final,
                   #    'SelKBest2Final':SelKBest2Final,
                       'SelKBest3':SelKBest3Final 
                      }
