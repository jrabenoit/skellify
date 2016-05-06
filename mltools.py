#!/usr/bin/env python 

from sklearn import svm
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import ensemble
from sklearn import linear_model
import copy

#.fit fits the model to the dataset in brackets. Score tests the fitted model on data.

def GauNaiBay(fX_train, fX_test, fy_train, fy_test):
    lX_train = copy.copy(fX_train)
    lX_test = copy.copy(fX_test)
    ly_train = copy.copy(fy_train)
    ly_test = copy.copy(fy_test)
    for i in range(0,len(fX_train)):
        gnb = naive_bayes.GaussianNB()
        gnb.fit(lX_train[i], ly_train[i])
        lX_train[i] = gnb.score(lX_train[i], ly_train[i])
        lX_test[i] = gnb.score(lX_test[i], ly_test[i])
    return lX_train, lX_test

def KNeighbors(fX_train, fX_test, fy_train, fy_test):
    lX_train = copy.copy(fX_train)
    lX_test = copy.copy(fX_test)
    ly_train = copy.copy(fy_train)
    ly_test = copy.copy(fy_test)
    for i in range(0,len(fX_train)):
        knc = neighbors.KNeighborsClassifier()
        knc.fit(lX_train[i], ly_train[i])
        lX_train[i] = knc.score(lX_train[i], ly_train[i])
        lX_test[i] = knc.score(lX_test[i], ly_test[i])
    return lX_train, lX_test

def CSupSvc(fX_train, fX_test, fy_train, fy_test):
    lX_train = copy.copy(fX_train)
    lX_test = copy.copy(fX_test)
    ly_train = copy.copy(fy_train)
    ly_test = copy.copy(fy_test)
    for i in range(len(fX_train)):
        csvm = svm.SVC()
        csvm.fit(lX_train[i], ly_train[i])
        lX_train[i] = csvm.score(lX_train[i], ly_train[i])
        lX_test[i] = csvm.score(lX_test[i], ly_test[i])
    return lX_train, lX_test
    
def RandomForest(fX_train, fX_test, fy_train, fy_test):
    lX_train = copy.copy(fX_train)
    lX_test = copy.copy(fX_test)
    ly_train = copy.copy(fy_train)
    ly_test = copy.copy(fy_test)
    for i in range(len(fX_train)):
        rf = ensemble.RandomForestClassifier()
        rf.fit(lX_train[i], ly_train[i])
        lX_train[i] = rf.score(lX_train[i], ly_train[i])
        lX_test[i] = rf.score(lX_test[i], ly_test[i])
    return lX_train, lX_test

def LinearSgd(fX_train, fX_test, fy_train, fy_test):
    lX_train = copy.copy(fX_train)
    lX_test = copy.copy(fX_test)
    ly_train = copy.copy(fy_train)
    ly_test = copy.copy(fy_test)
    for i in range(len(fX_train)):
        sgd = linear_model.SGDClassifier()
        sgd.fit(lX_train[i], ly_train[i])
        lX_train[i] = sgd.score(lX_train[i], ly_train[i])
        lX_test[i] = sgd.score(lX_test[i], ly_test[i])
    return lX_train, lX_test

ml_func_dict = {
            #    'GauNaiBay':GauNaiBay,
            #    'KNeighbors':KNeighbors,
            #    'CSupSvc':CSupSvc,
            #    'RandomForest':RandomForest,
            #    'LinearSgd':LinearSgd
               } 

def LSvm_maker(penalty,loss,dual,C):
    return lambda fX_train, fX_test, fy_train, fy_test: LSvm_base(fX_train, fX_test, fy_train, fy_test, penalty,loss,dual,C)
    
for C in [0.01,0.03,0.1,0.3,1.0,3.0,10.0,30.0]:
    ml_func_dict['LSvmL1_C'+str(C).replace(".","p")] = LSvm_maker(penalty='l1',loss=None,dual=False,C=C)
    ml_func_dict['LSvmL2_C'+str(C).replace(".","p")] = LSvm_maker(penalty='l2',loss='hinge',dual=True,C=C)
    
def LSvm_base(fX_train, fX_test, fy_train, fy_test, penalty='l1', loss=None, dual=False, C=1.0):
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

################################################################################

def GauNaiBayFinal(fX_train, fX_test, fy_train, fy_test):
    lX_train = copy.copy(fX_train)
    lX_test = copy.copy(fX_test)
    ly_train = copy.copy(fy_train)
    ly_test = copy.copy(fy_test)
    gnb = naive_bayes.GaussianNB()
    gnb.fit(lX_train, ly_train)
    lX_train = gnb.score(lX_train, ly_train)
    lX_test = gnb.score(lX_test, ly_test)
    return lX_train, lX_test
    
def KNeighborsFinal(fX_train, fX_test, fy_train, fy_test):
    lX_train = copy.copy(fX_train)
    lX_test = copy.copy(fX_test)
    ly_train = copy.copy(fy_train)
    ly_test = copy.copy(fy_test)
    knc = neighbors.KNeighborsClassifier()
    knc.fit(lX_train, ly_train)
    lX_train = knc.score(lX_train, ly_train)
    lX_test = knc.score(lX_test, ly_test)
    return lX_train, lX_test
    
def CSupSvcFinal(fX_train, fX_test, fy_train, fy_test):
    lX_train = copy.copy(fX_train)
    lX_test = copy.copy(fX_test)
    ly_train = copy.copy(fy_train)
    ly_test = copy.copy(fy_test)
    csvm = svm.SVC()
    csvm.fit(lX_train, ly_train)
    lX_train = csvm.score(lX_train, ly_train)
    lX_test = csvm.score(lX_test, ly_test)
    return lX_train, lX_test
    
def RandomForestFinal(fX_train, fX_test, fy_train, fy_test):
    lX_train = copy.copy(fX_train)
    lX_test = copy.copy(fX_test)
    ly_train = copy.copy(fy_train)
    ly_test = copy.copy(fy_test)
    rf = ensemble.RandomForestClassifier()
    rf.fit(lX_train, ly_train)
    lX_train = rf.score(lX_train, ly_train)
    lX_test = rf.score(lX_test, ly_test)
    return lX_train, lX_test

def LinearSgdFinal(fX_train, fX_test, fy_train, fy_test):
    lX_train = copy.copy(fX_train)
    lX_test = copy.copy(fX_test)
    ly_train = copy.copy(fy_train)
    ly_test = copy.copy(fy_test)
    sgd = linear_model.SGDClassifier()
    sgd.fit(lX_train, ly_train)
    lX_train = sgd.score(lX_train, ly_train)
    lX_test = sgd.score(lX_test, ly_test)
    return lX_train, lX_test
    
ml_func_dict_final = {
               #       'GauNaiBayFinal':GauNaiBayFinal,
               #       'KNeighborsFinal':KNeighborsFinal,
               #       'CSupSvcFinal':CSupSvcFinal,
               #       'RandomForestFinal':RandomForestFinal,
               #       'LinearSgdFinal':LinearSgdFinal
                     } 

def LSvm_maker_final(penalty,loss,dual,C):
    return lambda fX_train, fX_test, fy_train, fy_test: LSvm_base_final(fX_train, fX_test, fy_train, fy_test, penalty,loss,dual,C)
    
for C in [0.01,0.03,0.1,0.3,1.0,3.0,10.0,30.0]:
    ml_func_dict_final['LSvmL1_C'+str(C).replace(".","p")] = LSvm_maker_final(penalty='l1',loss=None,dual=False,C=C)
    ml_func_dict_final['LSvmL2_C'+str(C).replace(".","p")] = LSvm_maker_final(penalty='l2',loss='hinge',dual=True,C=C)
    
def LSvm_base_final(fX_train, fX_test, fy_train, fy_test, penalty='l1', loss=None, dual=False, C=1.0):
    # Note: c argument not actually used yet
    lX_train = copy.copy(fX_train)
    lX_test = copy.copy(fX_test)
    ly_train = copy.copy(fy_train)
    ly_test = copy.copy(fy_test)
    lsvm = svm.LinearSVC(penalty=penalty, dual=dual, C=C)
    lsvm.fit(lX_train, ly_train)
    lX_train = lsvm.score(lX_train, ly_train)
    lX_test = lsvm.score(lX_test, ly_test)
    return lX_train, lX_test

