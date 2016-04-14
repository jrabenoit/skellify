#!/usr/bin/env python 

import itertools
from pprint import pprint

'''
def helper(d,tag):
    d2 = {k:v for (k,v) in d.items() if tag in k}
    inverse = [(value,key) for key,value in d2.items()]
    return max(inverse)[1]
'''

def PickBest(test_results):
#Create new dict for each outer fold, compare all param sets for fold, return key
    folds = {}
    fold_1 = max({k:v for (k,v) in test_results.items() if 'test_1' in k}, key=test_results.get)
    fold_2 = max({k:v for (k,v) in test_results.items() if 'test_2' in k}, key=test_results.get)
    fold_3 = max({k:v for (k,v) in test_results.items() if 'test_3' in k}, key=test_results.get)
    fold_4 = max({k:v for (k,v) in test_results.items() if 'test_4' in k}, key=test_results.get)
    fold_5 = max({k:v for (k,v) in test_results.items() if 'test_5' in k}, key=test_results.get)
    
    #fold_1 = helper(test_results,'test_1')
    
    fold_index= [fold_1, fold_2, fold_3, fold_4, fold_5]
    
    folds[fold_1] = test_results[fold_1]
    folds[fold_2] = test_results[fold_2]
    folds[fold_3] = test_results[fold_3]
    folds[fold_4] = test_results[fold_4]
    folds[fold_5] = test_results[fold_5]    
    
    pprint(folds)

    return fold_index, folds

def PrintFinal(final_train_results, final_test_results, n_1, n_2):   

    pprint(final_train_results)    
    final_average_train = round(sum(final_train_results.values())/5, 2)
    print('\n>>> Average of best tool across training sets: {}% <<<\n'.format(round(final_average_train, 2)))

    pprint(final_test_results)
    final_average_test = round(sum(final_test_results.values())/5, 2)
    print('\n>>> Expected generalization accuracy if given new data: {}% <<<\n'.format(round(final_average_test, 2)))
    
    n_max = max(n_1, n_2)
    n_min = min(n_1, n_2)
    expected_accuracy = (n_max/(n_min + n_max))*100
    print('>>> Expected accuracy for group sizes n_1 = {} and n_2 = {}: {}% <<<\n'.format(n_1, n_2, round(expected_accuracy, 2)))

    acc_diff = final_average_test - expected_accuracy
    if acc_diff > 0:
        acc_direction = str('above')
    elif acc_diff < 0:
        acc_direction = str('below')
    else: 
        acc_direction = str('equal')
    print('>>> Test accuracy was {} expected accuracy by {}% <<<\n'.format(acc_direction, round(abs(acc_diff), 2)))
        
    return final_average_train, final_average_test
