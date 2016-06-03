#!/usr/bin/env python 

from operator import itemgetter

#Gives average accuracy of each param set tested on inner loop data
def InnerAverages(lX_train, lX_test, param_set):
    ps = str(param_set)
    ir_train = {
    'train_1 '+ps : sum(itemgetter(0,1,2,3,4)(lX_train))*100/5,
    'train_2 '+ps : sum(itemgetter(5,6,7,8,9)(lX_train))*100/5,
    'train_3 '+ps : sum(itemgetter(10,11,12,13,14)(lX_train))*100/5,
    'train_4 '+ps : sum(itemgetter(15,16,17,18,19)(lX_train))*100/5,
    'train_5 '+ps : sum(itemgetter(20,21,22,23,24)(lX_train))*100/5
    }
    ir_test = {
    'test_1 '+ps : sum(itemgetter(0,1,2,3,4)(lX_test))*100/5,
    'test_2 '+ps : sum(itemgetter(5,6,7,8,9)(lX_test))*100/5,
    'test_3 '+ps : sum(itemgetter(10,11,12,13,14)(lX_test))*100/5,
    'test_4 '+ps : sum(itemgetter(15,16,17,18,19)(lX_test))*100/5,
    'test_5 '+ps : sum(itemgetter(20,21,22,23,24)(lX_test))*100/5
    }
    
    return ir_train, ir_test
   
#Gives average accuracy of the best param set for each fold, on outer loop data
def OuterAverages(lX_train, lX_test, param_set):
    ps = str(param_set)
    oR_train = {ps : lX_train*100}
    oR_test = {ps : lX_test*100}
    
    return oR_train, oR_test
    
