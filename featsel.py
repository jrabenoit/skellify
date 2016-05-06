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
                 'SelKBest1':SelKBest1,
                 'SelKBest2':SelKBest2,
                 'SelKBest3':SelKBest3 
                }

def SelKBest_maker(k):
    return lambda X_train, X_test, y_train, y_test: SelKBest_base(X_train, X_test, y_train, y_test, k)
    
for k in range(10,100,10):
    feat_sel_dict['SelKBest'+str(k)+'feature'] = SelKBest_maker(k)
    
def SelKBest_base(X_train, X_test, y_train, y_test, k=10):
    fX_train = copy.copy(X_train)
    fX_test = copy.copy(X_test)
    fy_train = copy.copy(y_train)
    fy_test = copy.copy(y_test)
    for i in range(0,len(X_train)):
        skb = SelectKBest(k=k)
        fX_train[i] = skb.fit_transform(fX_train[i], fy_train[i])
        fX_test[i] = skb.transform(fX_test[i])  
    return fX_train, fX_test, fy_train, fy_test

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
                       'SelKBest1Final':SelKBest1Final,
                       'SelKBest2Final':SelKBest2Final,
                       'SelKBest3':SelKBest3Final 
                      }

def SelKBest_maker_final(k):
    return lambda X_train, X_test, y_train, y_test: SelKBest_base_final(X_train, X_test, y_train, y_test, k)
    
for k in range(10,100,10):
    feat_sel_dict_final['SelKBest'+str(k)+'feature'] = SelKBest_maker_final(k)
    
def SelKBest_base_final(X_train, X_test, y_train, y_test, k=10):
    fX_train = copy.copy(X_train)
    fX_test = copy.copy(X_test)
    fy_train = copy.copy(y_train)
    fy_test = copy.copy(y_test)
    skb = SelectKBest(k=k)
    skb.fit(fX_train, fy_train)
    fX_train = skb.transform(fX_train)
    fX_test = skb.transform(fX_test)  
    return fX_train, fX_test, fy_train, fy_test
