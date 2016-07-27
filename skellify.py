#!/usr/bin/env python 

#Temporary test message for git setup

#Data prep modules, processing modules, and result reporting modules
import data, prep, crossval, pprint 
import iterator, comparator
import visualize

#Select a group of scans to use 
print('Running Step 1/10: Select Groups for Classification')
n_1, n_2, dir_1, dir_2, mask = data.SelectGroup()

for i in range(10):
    print('>>> ITERATION {} OF 10'.format(i+1))  
    print('Running Step 2/10: Sort & Mask Data')
#Load two sets of scans into a dataset and mask the data
    concat_dict, concat_subjects_dict, iter_n = prep.Sort(dir_1, dir_2, n_1, n_2)
#Flatten the data, z-normalize (center the data (remove the mean), scale to unit variance)
    print('Running Step 3/10: Flatten the 4D Files to 2D Matrices, Z-normalization')
    masked_dict = prep.MaskFlatten(concat_dict, mask, iter_n)
    znorm_dict = prep.ZNormalize(masked_dict, iter_n)
    print('Running Step 4/10: Label Groups')
    group_label_dict = prep.GroupLabels(n_1, n_2, iter_n)
#Do 5-fold CV setup for outer fold (X=data, y=labels)
#NOTE: THIS STEP AND THE NEXT STEP DO NOT RUN THE CV. THESE STEPS SET UP 25 DIFFERENT GROUPS OF DATA, AS DETERMINED BY A 5 OUTER FOLD CV, AND A 5 INNER-FOLD CV ON EACH OUTER FOLD.
    print('Running Step 5/10: Set Up Outer CV Loop')
    oX_train, oX_test, oy_train, oy_test, train_index_outer, test_index_outer, train_index_files, test_index_files = crossval.oSkfCv(group_label_dict, znorm_dict, iter_n, concat_subjects_dict)
#Do 5-fold CV setup for each outer fold, creating 25 inner folds total
    print('Running Step 6/10: Set Up Inner CV Loop')
    iX_train, iX_test, iy_train, iy_test, train_index_inner, test_index_inner = crossval.iSkfCv(oy_train, oX_train, iter_n)
#Run the data from iSkfCv through an iterative tool that tries all learning combinations
    print('Running Step 7/10: List featsel/decomp/mltool Combos')
    test_results, param_set_list = iterator.ParameterSets(iX_train, iX_test, iy_train, iy_test, iter_n)
#Compare all accuracy results to pick best combination of parameters
    print('Running Step 8/10: Pick Best featsel/decomp/mltool Combo')
    fold_index, folds = comparator.PickBest(test_results)
    print('Running Step 9/10: Run Winning Combo on Outer Loop Holdout')
#Run best param set for each inner fold on corresponding outer fold
    final_train_results, final_test_results, final_train_predictions, final_test_predictions, final_train_labels, final_test_labels = iterator.TestHoldout(oX_train, oX_test, oy_train, oy_test, fold_index, iter_n) 
    print('Running Step 10/10: Print Test vs. Chance Results')
#Print the results
    final_average_train, final_average_test = comparator.PrintFinal(final_train_results, final_test_results, n_1, n_2, final_train_predictions, final_test_predictions, final_train_labels, final_test_labels, iter_n, train_index_files, test_index_files)

#Build bootstrap distribution
    comparator.BootstrapDistribution(concat_subjects_dict, iter_n, train_index_outer, test_index_outer, train_index_files, test_index_files)
