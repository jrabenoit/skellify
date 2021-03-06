#!/usr/bin/env python 

#Data prep modules, processing modules, and result reporting modules
from collections import defaultdict
import pprint, itertools
import data, prep, crossval
import iterator, comparator
import bootstrap
import visualize

#Select a group of scans to use 
print('Step 1/10: Select Two Groups')
n_1, n_2, dir_1, dir_2, mask = data.SelectGroup()

concatenated_test = defaultdict(list)
concatenated_train = defaultdict(list)

number_of_iterations = 10
for i in range(number_of_iterations):
    print('>>>ITERATION {} OF {}'.format(i+1,number_of_iterations))  
    print('Step 2/10: Sort Data')
#Load two sets of scans into a dataset and mask the data
    concat_dict, concat_subjects_dict, iter_n = prep.Sort(dir_1, dir_2, n_1, n_2)
#Flatten the data, z-normalize (center the data (remove the mean), scale to unit variance)
    print('Step 3/10: Flatten 4D Images to 2D Matrices, Z-normalization')
    masked_dict = prep.MaskFlatten(concat_dict, mask, iter_n)
    znorm_dict = prep.ZNormalize(masked_dict, iter_n)
    print('Step 4/10: Label Groups')
    group_label_dict = prep.GroupLabels(n_1, n_2, iter_n)
#Do 5-fold CV setup for outer fold (X=data, y=labels)
#NOTE: THIS STEP AND THE NEXT STEP DO NOT RUN THE CV. THESE STEPS SET UP 25 DIFFERENT GROUPS OF DATA, AS DETERMINED BY A 5 OUTER FOLD CV, AND A 5 INNER-FOLD CV ON EACH OUTER FOLD.
    print('Step 5/10: Set Up Outer CV')
    oX_train, oX_test, oy_train, oy_test, train_index_outer, test_index_outer, train_index_files, test_index_files = crossval.oSkfCv(group_label_dict, znorm_dict, iter_n, concat_subjects_dict)
#Do 5-fold CV setup for each outer fold, creating 25 inner folds total
    print('Step 6/10: Set Up Inner CV')
    iX_train, iX_test, iy_train, iy_test, train_index_inner, test_index_inner = crossval.iSkfCv(oy_train, oX_train, iter_n)
#Run the data from iSkfCv through an iterative tool that tries all learning combinations
    print('Step 7/10: Try all featsel/decomp/mltool Combos')
    test_results, param_set_list = iterator.ParameterSets(iX_train, iX_test, iy_train, iy_test, iter_n)
#Compare all accuracy results to pick best combination of parameters
    print('Step 8/10: Pick Best featsel/decomp/mltool Combo')
    fold_index, folds = comparator.PickBest(test_results)
    print('Step 9/10: Run Best Combo on Outer CV Holdout')
#Run best param set for each inner fold on corresponding outer fold
    final_train_results, final_test_results, final_train_predictions, final_test_predictions, final_train_labels, final_test_labels = iterator.TestHoldout(oX_train, oX_test, oy_train, oy_test, fold_index, iter_n) 
    print('Step 10/10: Print Test vs. Chance Results')
#Print the results
    final_train_correct, final_test_correct = comparator.PrintFinal(final_train_results, final_test_results, n_1, n_2, final_train_predictions, final_test_predictions, final_train_labels, final_test_labels, iter_n, train_index_files, test_index_files)

#Build accuracy profile for each subject
    subject_results_test, subject_results_train = comparator.SubjectAccuracy(iter_n, final_train_correct, final_test_correct, train_index_files, test_index_files, concat_subjects_dict)

#Concatenate classification attempts for each subject
    for key, value in subject_results_test.items():
        concatenated_test[key].append(value)

    for key, value in subject_results_train.items():
        concatenated_train[key].append(value)

#print('>>>RAW TRAIN CLASSIFICATIONS')
#print(concatenated_train)

#print('>>>RAW TEST CLASSIFICATIONS')
#print(concatenated_test)

print('>>>CHAINING TRAINING RESULTS TOGETHER')
concatenated_train_chained = defaultdict(list)
for key, value in concatenated_train.items():
    concatenated_train_chained[key] = list(itertools.chain.from_iterable(value))
    
#print('>>>CHAINING TEST RESULTS TOGETHER')
concatenated_test_chained = defaultdict(list)
for key, value in concatenated_test.items():
    concatenated_test_chained[key] = list(itertools.chain.from_iterable(value))

print('>>>TRAIN SUBJECT ACCURACY SCORES')
per_subject_train_acc = defaultdict(list)
for key, value in concatenated_train_chained.items():
    per_subject_train_acc[key] = round((sum(value)/len(value))*100,4)
#pprint.pprint(per_subject_train_acc)

print('>>>TEST SUBJECT ACCURACY SCORES')
per_subject_test_acc = defaultdict(list)
for key, value in concatenated_test_chained.items():
    per_subject_test_acc[key] = round((sum(value)/len(value))*100,4)
#pprint.pprint(per_subject_test_acc)

final_acc = sum(list(per_subject_test_acc.values()))/len(list(per_subject_test_acc.values()))
print('\n>>>AVERAGE ACCURACY: {}%'.format(round(final_acc,2)))
p_value = bootstrap.EmpiricalDistro(n_1, n_2, per_subject_test_acc)
print('>>>P VALUE UNCORRECTED: {}'.format(p_value))
print('>>>P VALUE CORRECTED: {}'.format(p_value*3))

print('>>>SAMPLE SIZE: {}'.format(n_1 + n_2))

