#!/usr/bin/env python 

from sklearn import decomposition
import copy
    
def NullDecomp(X_train, X_test, y_train, y_test):
    return X_train, X_test, y_train, y_test    

decomp_dict = {'NullDecomp':NullDecomp}

def RPca_maker(n_components):
    return lambda iX_train, iX_test, iy_train, iy_test: RPca_base(iX_train, iX_test, iy_train, iy_test, n_components)
    
for n_components in range(1,21,1):
    decomp_dict['RPca'+str(n_components)] = RPca_maker(n_components)
    
def RPca_base(iX_train, iX_test, iy_train, iy_test, n_components=3):
    dX_train = copy.copy(iX_train)
    dX_test = copy.copy(iX_test)
    dy_train = copy.copy(iy_train)
    dy_test = copy.copy(iy_test)
    for i in range(0,len(iX_train)):
        pca = decomposition.RandomizedPCA(n_components=n_components)
        pca.fit(dX_train[i])
        dX_train[i] = pca.transform(dX_train[i])
        dX_test[i] = pca.transform(dX_test[i])
    return dX_train, dX_test, dy_train, dy_test

################################################################################
    
def NullDecompFinal(iX_train, iX_test, iy_train, iy_test):
    return iX_train, iX_test, iy_train, iy_test 
    
decomp_dict_final = {'NullDecomp':NullDecompFinal}
               
def RPca_maker_final(n_components):
    return lambda iX_train, iX_test, iy_train, iy_test: RPca_base_final(iX_train, iX_test, iy_train, iy_test, n_components)
    
for n_components in range(1,21,1):
    decomp_dict_final['RPca'+str(n_components)] = RPca_maker_final(n_components)
    
def RPca_base_final(iX_train, iX_test, iy_train, iy_test, n_components=3):
    dX_train = copy.copy(iX_train)
    dX_test = copy.copy(iX_test)
    dy_train = copy.copy(iy_train)
    dy_test = copy.copy(iy_test)
    pca = decomposition.RandomizedPCA(n_components=n_components)
    pca.fit(dX_train)
    dX_train = pca.transform(dX_train)
    dX_test = pca.transform(dX_test) 
    return dX_train, dX_test, dy_train, dy_test

