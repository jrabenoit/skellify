#!/usr/bin/env python 

from sklearn import svm
import copy

#Run inner loop Linear SVM (l = linear SVM, s = score)
    
def LSvmL1(fX_train, fX_test, fy_train, fy_test):
    lX_train = copy.copy(fX_train)
    lX_test = copy.copy(fX_test)
    ly_train = copy.copy(fy_train)
    ly_test = copy.copy(fy_test)
    for i in range(len(fX_train)):
        #Values of C<=0.56 don't work
        lsvm = svm.LinearSVC(penalty='l1', dual=False)
        lsvm.fit_transform(lX_train[i], ly_train[i])
        lsvm.transform(lX_test[i])
        lX_train[i] = lsvm.score(lX_train[i], ly_train[i])
        lX_test[i] = lsvm.score(lX_test[i], ly_test[i])

    return lX_train, lX_test

def LSvmL2(fX_train, fX_test, fy_train, fy_test):
    lX_train = copy.copy(fX_train)
    lX_test = copy.copy(fX_test)
    ly_train = copy.copy(fy_train)
    ly_test = copy.copy(fy_test)
    for i in range(0,len(fX_train)):
        lsvm = svm.LinearSVC(C = 0.3, penalty='l2', loss='hinge', dual=True)
        lsvm.fit_transform(lX_train[i], ly_train[i])
        lsvm.transform(lX_test[i])
        lX_train[i] = lsvm.score(lX_train[i], ly_train[i])
        lX_test[i] = lsvm.score(lX_test[i], ly_test[i])
        
    return lX_train, lX_test
'''
def LSvmL1_alternate(fX_train, fX_test, fy_train, fy_test):
    return LSvm_base(fX_train, fX_test, fy_train, fy_test,
              penalty='l1', loss=None, dual=False)
              
def LSvmL2_alternate(fX_train, fX_test, fy_train, fy_test):
    return LSvm_base(fX_train, fX_test, fy_train, fy_test,
              penalty='l2', loss='hinge', dual=True)

ml_func_dict = {}
for c in [0.01,0.03,0.1,0.3,1.0,3.0,10.0,30.0]:
    ml_func_dict['LSvmL1_c'+str(c)] = LSvm_maker(penalty='l1',loss=None,
                                                 dual=False,c=c)

def LSvm_maker(penalty,loss,dual,c):
    return lambda fX_train, fX_test, fy_train, fy_test :
        LSvm_base(fX_train, fX_test, fy_train, fy_test,
                  penalty,loss,dual,c)

def LSvm_base(fX_train, fX_test, fy_train, fy_test,
              penalty='l1', loss=None, dual=False, c=1.0):
    # Note: c argument not actually used yet
    lX_train = copy.copy(fX_train)
    lX_test = copy.copy(fX_test)
    ly_train = copy.copy(fy_train)
    ly_test = copy.copy(fy_test)
    for i in range(0,len(fX_train)):
        if not loss:
            lsvm = svm.LinearSVC(penalty=penalty, dual=dual, )
        else:
            lsvm = svm.LinearSVC(penalty=penalty, dual=dual, loss=loss)
        lsvm.fit_transform(lX_train[i], ly_train[i])
        lsvm.transform(lX_test[i])
        lX_train[i] = lsvm.score(lX_train[i], ly_train[i])
        lX_test[i] = lsvm.score(lX_test[i], ly_test[i])
    return lX_train, lX_test
'''
    
def LSvmL1Final(fX_train, fX_test, fy_train, fy_test):
    lX_train = copy.copy(fX_train)
    lX_test = copy.copy(fX_test)
    ly_train = copy.copy(fy_train)
    ly_test = copy.copy(fy_test)
    lsvm = svm.LinearSVC(penalty='l1', dual=False)
    lsvm.fit_transform(lX_train, ly_train)
    lsvm.transform(lX_test)
    lX_train = lsvm.score(lX_train, ly_train)
    lX_test = lsvm.score(lX_test, ly_test)

    return lX_train, lX_test

def LSvmL2Final(fX_train, fX_test, fy_train, fy_test):
    lX_train = copy.copy(fX_train)
    lX_test = copy.copy(fX_test)
    ly_train = copy.copy(fy_train)
    ly_test = copy.copy(fy_test)
    lsvm = svm.LinearSVC(penalty='l2', loss='hinge', dual=True)
    lsvm.fit_transform(lX_train, ly_train)
    lsvm.transform(lX_test)
    lX_train = lsvm.score(lX_train, ly_train)
    lX_test = lsvm.score(lX_test, ly_test)
        
    return lX_train, lX_test

