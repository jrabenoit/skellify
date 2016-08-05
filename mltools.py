#!/usr/bin/env python 

from sklearn import svm
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import ensemble
from sklearn import linear_model
import copy

#.fit fits the model to the dataset in brackets. Score tests the fitted model on data.

def GauNaiBay(X_train, X_test, y_train, y_test):
    for i in range(0,len(X_train)):
        gnb = naive_bayes.GaussianNB()
        gnb.fit(X_train[i], y_train[i])
        X_train[i] = gnb.score(X_train[i], y_train[i])
        X_test[i] = gnb.score(X_test[i], y_test[i]) 
    return X_train, X_test

def KNeighbors(X_train, X_test, y_train, y_test):
    for i in range(0,len(X_train)):
        knc = neighbors.KNeighborsClassifier()
        knc.fit(X_train[i], y_train[i])
        X_train[i] = knc.score(X_train[i], y_train[i])
        X_test[i] = knc.score(X_test[i], y_test[i])
    return X_train, X_test

def CSupSvc(X_train, X_test, y_train, y_test):
    for i in range(len(X_train)):
        csvm = svm.SVC()
        csvm.fit(X_train[i], y_train[i])
        X_train[i] = csvm.score(X_train[i], y_train[i])
        X_test[i] = csvm.score(X_test[i], y_test[i])
    return X_train, X_test
    
def RandomForest(X_train, X_test, y_train, y_test):
    for i in range(len(X_train)):
        rf = ensemble.RandomForestClassifier()
        rf.fit(X_train[i], y_train[i])
        X_train[i] = rf.score(X_train[i], y_train[i])
        X_test[i] = rf.score(X_test[i], y_test[i])
    return X_train, X_test

def ExtraTrees(X_train, X_test, y_train, y_test):
    for i in range(len(X_train)):
        rf = ensemble.ExtraTreesClassifier()
        rf.fit(X_train[i], y_train[i])
        X_train[i] = rf.score(X_train[i], y_train[i])
        X_test[i] = rf.score(X_test[i], y_test[i])
    return X_train, X_test

def LinearSgd(X_train, X_test, y_train, y_test):
    for i in range(len(X_train)):
        sgd = linear_model.SGDClassifier()
        sgd.fit(X_train[i], y_train[i])
        X_train[i] = sgd.score(X_train[i], y_train[i])
        X_test[i] = sgd.score(X_test[i], y_test[i])
    return X_train, X_test

ml_func_dict = {
#                'GauNaiBay':GauNaiBay,
#                'KNeighbors':KNeighbors,
#                'CSupSvc':CSupSvc,
                'RandomForest':RandomForest
#                 'ExtraTrees':ExtraTrees
#                'LinearSgd':LinearSgd
               } 
'''
def LSvm_maker(penalty,loss,dual,C):
    return lambda fX_train, fX_test, fy_train, fy_test: LSvm_base(fX_train, fX_test, fy_train, fy_test, penalty,loss,dual,C)
 
#remember to change the final run values of C as well, if you change this one   
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
'''    
################################################################################
################################################################################

def GauNaiBayFinal(X_train, X_test, y_train, y_test):
    gnb = naive_bayes.GaussianNB()
    gnb.fit(X_train, y_train)
    lX_train = gnb.score(X_train, y_train)
    lX_test = gnb.score(X_test, y_test)
    lX_train_predict= gnb.predict(X_train)
    lX_test_predict= gnb.predict(X_test)
    ly_train_labels= y_train
    ly_test_labels= y_test
    return lX_train, lX_test, lX_train_predict, lX_test_predict, ly_train_labels, ly_test_labels
    
def KNeighborsFinal(X_train, X_test, y_train, y_test):
    knc = neighbors.KNeighborsClassifier()
    knc.fit(X_train, y_train)
    lX_train = knc.score(X_train, y_train)
    lX_test = knc.score(X_test, y_test)
    lX_train_predict= knc.predict(X_train)
    lX_test_predict= knc.predict(X_test)
    ly_train_labels= y_train
    ly_test_labels= y_test
    return lX_train, lX_test, lX_train_predict, lX_test_predict, ly_train_labels, ly_test_labels
    
def CSupSvcFinal(X_train, X_test, y_train, y_test):
    csvm = svm.SVC()
    csvm.fit(X_train, y_train)
    lX_train = csvm.score(X_train, y_train)
    lX_test = csvm.score(X_test, y_test)
    lX_train_predict= csvm.predict(X_train)
    lX_test_predict= csvm.predict(X_test)
    ly_train_labels= y_train
    ly_test_labels= y_test
    return lX_train, lX_test, lX_train_predict, lX_test_predict, ly_train_labels, ly_test_labels
    
def RandomForestFinal(X_train, X_test, y_train, y_test):
    rf = ensemble.RandomForestClassifier()
    rf.fit(X_train, y_train)
    lX_train = rf.score(X_train, y_train)
    lX_test = rf.score(X_test, y_test)
    lX_train_predict= rf.predict(X_train)
    lX_test_predict= rf.predict(X_test)
    ly_train_labels= y_train
    ly_test_labels= y_test
    return lX_train, lX_test, lX_train_predict, lX_test_predict, ly_train_labels, ly_test_labels

def ExtraTreesFinal(X_train, X_test, y_train, y_test):
    rf = ensemble.ExtraTreesClassifier()
    rf.fit(X_train, y_train)
    lX_train = rf.score(X_train, y_train)
    lX_test = rf.score(X_test, y_test)
    lX_train_predict= rf.predict(X_train)
    lX_test_predict= rf.predict(X_test)
    ly_train_labels= y_train
    ly_test_labels= y_test
    return lX_train, lX_test, lX_train_predict, lX_test_predict, ly_train_labels, ly_test_labels

def LinearSgdFinal(X_train, X_test, y_train, y_test):
    sgd = linear_model.SGDClassifier()
    sgd.fit(X_train, y_train)
    lX_train = sgd.score(X_train, y_train)
    lX_test = sgd.score(X_test, y_test)
    lX_train_predict= sgd.predict(X_train)
    lX_test_predict= sgd.predict(X_test)
    ly_train_labels= y_train
    ly_test_labels= y_test
    return lX_train, lX_test, lX_train_predict, lX_test_predict, ly_train_labels, ly_test_labels
    
ml_func_dict_final = {
#                      'GauNaiBay':GauNaiBayFinal,
#                      'KNeighbors':KNeighborsFinal,
#                      'CSupSvc':CSupSvcFinal,
                      'RandomForest':RandomForestFinal
#                       'ExtraTrees':ExtraTreesFinal
#                      'LinearSgd':LinearSgdFinal
                     } 
'''
def LSvm_maker_final(penalty,loss,dual,C):
    return lambda fX_train, fX_test, fy_train, fy_test: LSvm_base_final(fX_train, fX_test, fy_train, fy_test, penalty,loss,dual,C)
    
# Expanded C: [0.01,0.03,0.1,0.3,1.0,3.0,10.0,30.0]
for C in [0.01,0.03,0.1,0.3,1.0,3.0,10.0,30.0]:
    ml_func_dict_final['LSvmL1_C'+str(C).replace(".","p")] = LSvm_maker_final(penalty='l1',loss=None,dual=False,C=C)
    ml_func_dict_final['LSvmL2_C'+str(C).replace(".","p")] = LSvm_maker_final(penalty='l2',loss='hinge',dual=True,C=C)
    
def LSvm_base_final(fX_train, fX_test, fy_train, fy_test, penalty='l1', loss=None, dual=False, C=1.0):
    # Note: c argument not actually used yet
    for i in range(0,len(fX_train)):
        if not loss:
            lsvm = svm.LinearSVC(penalty=penalty, dual=dual, C=C)
        else:
            lsvm = svm.LinearSVC(penalty=penalty, dual=dual, loss=loss)
    lsvm.fit(fX_train, fy_train)
    lX_train = lsvm.score(fX_train, fy_train)
    lX_test = lsvm.score(fX_test, fy_test)
    lX_train_predict= lsvm.predict(fX_train)
    lX_test_predict= lsvm.predict(fX_test)
    ly_train_labels= fy_train
    ly_test_labels= fy_test
    return lX_train, lX_test, lX_train_predict, lX_test_predict, ly_train_labels, ly_test_labels
'''
