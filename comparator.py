#!/usr/bin/env python 

import itertools
import copy
import operator
import numpy as np
from scipy.stats import chisquare
from pprint import pprint

def PickBest(test_results):
    '''returns key for fold with min variance of the 3 sets of test parameters giving the highest accuracy on the inner loop holdout set'''
    test_results_copy = copy.copy(test_results)

#Get mean for each fold, remove two lowest accuracy items, change accuracy to variance, pick combination of parameters giving minimum variance for each fold.
    fold_1_mean = sum(value for key, value in test_results_copy.items() if 'test_1' in key.lower())/5
    fold_2_mean = sum(value for key, value in test_results_copy.items() if 'test_2' in key.lower())/5
    fold_3_mean = sum(value for key, value in test_results_copy.items() if 'test_3' in key.lower())/5
    fold_4_mean = sum(value for key, value in test_results_copy.items() if 'test_4' in key.lower())/5
    fold_5_mean = sum(value for key, value in test_results_copy.items() if 'test_5' in key.lower())/5
#Get dict elements (param sets) with highest 3 accuracies for each fold
    top_3_accs={}
    top_3_accs.update(dict(sorted(((k, v) for k, v in test_results_copy.items() if 'test_1' in k), key=operator.itemgetter(1), reverse=True)[:3]))
    top_3_accs.update(dict(sorted(((k, v) for k, v in test_results_copy.items() if 'test_2' in k), key=operator.itemgetter(1), reverse=True)[:3]))
    top_3_accs.update(dict(sorted(((k, v) for k, v in test_results_copy.items() if 'test_3' in k), key=operator.itemgetter(1), reverse=True)[:3]))
    top_3_accs.update(dict(sorted(((k, v) for k, v in test_results_copy.items() if 'test_4' in k), key=operator.itemgetter(1), reverse=True)[:3]))
    top_3_accs.update(dict(sorted(((k, v) for k, v in test_results_copy.items() if 'test_5' in k), key=operator.itemgetter(1), reverse=True)[:3]))
#Change accuracy to variance
    top_3_accs.update((k, (v-fold_1_mean)**2) for k,v in top_3_accs.items() if 'test_1' in k)
    top_3_accs.update((k, (v-fold_2_mean)**2) for k,v in top_3_accs.items() if 'test_2' in k)
    top_3_accs.update((k, (v-fold_3_mean)**2) for k,v in top_3_accs.items() if 'test_3' in k)
    top_3_accs.update((k, (v-fold_4_mean)**2) for k,v in top_3_accs.items() if 'test_4' in k)
    top_3_accs.update((k, (v-fold_5_mean)**2) for k,v in top_3_accs.items() if 'test_5' in k)
#Select key with minimum variance from remaining 3 elements per fold with highest accuracy
    folds = {}
    fold_1 = min({k:v for (k,v) in top_3_accs.items() if 'test_1' in k}, key=top_3_accs.get)
    fold_2 = min({k:v for (k,v) in top_3_accs.items() if 'test_2' in k}, key=top_3_accs.get)
    fold_3 = min({k:v for (k,v) in top_3_accs.items() if 'test_3' in k}, key=top_3_accs.get)
    fold_4 = min({k:v for (k,v) in top_3_accs.items() if 'test_4' in k}, key=top_3_accs.get)
    fold_5 = min({k:v for (k,v) in top_3_accs.items() if 'test_5' in k}, key=top_3_accs.get)
    
    fold_index= [fold_1, fold_2, fold_3, fold_4, fold_5]

    folds[fold_1] = top_3_accs[fold_1]
    folds[fold_2] = top_3_accs[fold_2]
    folds[fold_3] = top_3_accs[fold_3]
    folds[fold_4] = top_3_accs[fold_4]
    folds[fold_5] = top_3_accs[fold_5]    
    
    pprint(folds)
    return fold_index, folds

################################################################################

def PrintFinal(final_train_results, final_test_results, n_1, n_2, final_train_predictions, final_test_predictions, final_train_labels, final_test_labels, iter_n):   

    pprint(final_train_results)    
    final_average_train = round(sum(final_train_results.values())/5, 3)
    print('\n>>> Average of best tool across training sets: {}% <<<\n'.format(round(final_average_train, 2)))

    pprint(final_test_results)
    final_average_test = round(sum(final_test_results.values())/5, 3)
    print('\n>>> Average Tested Accuracy: {}% <<<\n'.format(round(final_average_test, 3)))
    
    n_max = max(n_1, n_2)
    n_min = min(n_1, n_2)
    expected_accuracy = (n_max/(n_min + n_max))*100
    print('>>> Chance Accuracy For n_1= {} and n_2= {}: {}% <<<\n'.format(n_1, n_2, round(expected_accuracy, 3)))
    acc_diff = final_average_test - expected_accuracy
    if acc_diff > 0:
        acc_direction = str('above')
    elif acc_diff < 0:
        acc_direction = str('below')
    else: 
        acc_direction = str('equals')
    print('>>> Tested accuracy {} chance accuracy by {}% <<<\n'.format(acc_direction, round(abs(acc_diff), 3)))
    
# Significance testing for 3 multiple comparisons, 1 df
    for i in range(iter_n):
        n_test_incorrect = 0
        n_test_correct = 0
        for pred,actual in zip(final_test_predictions[i],final_test_labels[i]):
            n_test_incorrect += sum(pred!=actual)
            n_test_correct += sum(pred==actual)
        print('n_test_incorrect for {} = {}'.format([i], n_test_incorrect))
        print('n_test_correct for {} = {}'.format([i], n_test_correct))
    count_obs_test= np.array([n_test_incorrect, n_test_correct])
    count_exp_test= np.array([n_min, n_max])

    return final_average_train, final_average_test

#Removed chi-square test; inappropriate for new stats
'''
    #This is a one-way chi square test
    #We are using chi square because of testing a single categorical variable for testing whether our sample data distribution is consistent with a theoretical data distribution (is it generalizable?)
    chi_square_test = chisquare(count_obs_test, f_exp= count_exp_test, ddof=0)
    print('>>> (chi square, raw p-value): {}\n <<<'.format(str(chi_square_test)))
'''
def BootstrapDistribution(concat_dict, iter_n, train_index_outer, test_index_outer, train_index_inner, test_index_inner):

    for i in range(iter_n):
        for subj,result in zip(list_of_train_subjects,list_of_train_results):
            if subj in subject_results_train:
                subject_results_train[subj] += [result]
            else:
                subject_results_train[subj] = [result]

        for subj,result in zip(list_of_test_subjects,list_of_test_results):
            if subj in subject_results_test:
                subject_results_test[subj] += [result]
            else:
                subject_results_test[subj] = [result]
                
    return 
