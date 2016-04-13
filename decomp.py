#!/usr/bin/env python 

from sklearn import decomposition
import copy


def RPca1(iX_train, iX_test, iy_train, iy_test):
    dX_train = copy.copy(iX_train)
    dX_test = copy.copy(iX_test)
    dy_train = copy.copy(iy_train)
    dy_test = copy.copy(iy_test)
    for i in range(0,len(iX_train)):
        pca = decomposition.RandomizedPCA()
        pca.fit(dX_train[i])
        dX_train[i] = pca.transform(dX_train[i])
        dX_test[i] = pca.transform(dX_test[i])        

    return dX_train, dX_test, dy_train, dy_test

    
def RPca2(iX_train, iX_test, iy_train, iy_test):
    dX_train = copy.copy(iX_train)
    dX_test = copy.copy(iX_test)
    dy_train = copy.copy(iy_train)
    dy_test = copy.copy(iy_test)
    for i in range(0,len(iX_train)):
        pca = decomposition.RandomizedPCA(n_components=3)
        pca.fit(dX_train[i])
        dX_train[i] = pca.transform(dX_train[i])
        dX_test[i] = pca.transform(dX_test[i])        

    return dX_train, dX_test, dy_train, dy_test

    
def NullDecomp(iX_train, iX_test, iy_train, iy_test):
    dX_train = copy.copy(iX_train)
    dX_test = copy.copy(iX_test)
    dy_train = copy.copy(iy_train)
    dy_test = copy.copy(iy_test)

    return dX_train, dX_test, dy_train, dy_test    

    
def RPca1Final(iX_train, iX_test, iy_train, iy_test):
    dX_train = copy.copy(iX_train)
    dX_test = copy.copy(iX_test)
    dy_train = copy.copy(iy_train)
    dy_test = copy.copy(iy_test)
    pca = decomposition.RandomizedPCA()
    pca.fit(dX_train)
    dX_train = pca.transform(dX_train)
    dX_test = pca.transform(dX_test)        

    return dX_train, dX_test, dy_train, dy_test

   
def RPca2Final(iX_train, iX_test, iy_train, iy_test):
    dX_train = copy.copy(iX_train)
    dX_test = copy.copy(iX_test)
    dy_train = copy.copy(iy_train)
    dy_test = copy.copy(iy_test)
    pca = decomposition.RandomizedPCA(n_components=3)
    pca.fit(dX_train)
    dX_train = pca.transform(dX_train)
    dX_test = pca.transform(dX_test)        

    return dX_train, dX_test, dy_train, dy_test

    
def NullDecompFinal(iX_train, iX_test, iy_train, iy_test):
    dX_train = copy.copy(iX_train)
    dX_test = copy.copy(iX_test)
    dy_train = copy.copy(iy_train)
    dy_test = copy.copy(iy_test)

    return dX_train, dX_test, dy_train, dy_test 
