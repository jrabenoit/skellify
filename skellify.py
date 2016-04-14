#!/usr/bin/env python 

#Temporary test message for git setup

#Data prep modules, processing modules, and result reporting modules
import data, prep, crossval 
import iterator, comparator
import visualize

#Select a group of scans to use 
print('Running Step 1/10: Select Groups for Classification')
n_1, n_2, dir_1, dir_2, mask = data.SelectGroup()

#Load two sets of scans into a dataset and mask the data
print('Running Step 2/10: Sort Data Into Groups and Mask Sorted Data')
sorted_files = prep.Sort(dir_1, dir_2)

print('Running Step 3/10: Flatten the 4D Files to 2D Matrices, z-normalization')
masked_data = prep.MaskFlatten(sorted_files, mask)
znorm_data = prep.ZNormalize(masked_data)

print('Running Step 4/10: Apply a Label to Each Group')
labels = prep.GroupLabels(n_1, n_2)

#Do 5-fold CV setup for outer fold (X=data, y=labels)
#NOTE: THIS STEP AND THE NEXT STEP DO NOT RUN THE CV. THESE STEPS SET UP 25 DIFFERENT GROUPS OF DATA, AS DETERMINED BY A 5 OUTER FOLD CV, AND A 5 INNER-FOLD CV ON EACH OUTER FOLD.
print('Running Step 5/10: Set Up Outer CV Loop: oSkfCv')
oX_train, oX_test, oy_train, oy_test = crossval.oSkfCv(labels, znorm_data)

#Do 5-fold CV setup for each outer fold, creating 25 inner folds total
print('Running Step 6/10: Set Up Inner CV Loop: iSkfCv')
iX_train, iX_test, iy_train, iy_test = crossval.iSkfCv(oy_train, oX_train)

#Run the data from iSkfCv through an iterative tool that tries all learning combinations
print('Running Step 7/10: Define Sets of Feature Transformation/Reduction & ML Classifier')
test_results, param_set_list = iterator.ParameterSets(iX_train, iX_test, iy_train, iy_test)

#Compare all accuracy results to pick best combination of parameters
print('Running Step 8/10: Pick the Best Feature Transformation/Reduction & ML Classifier')
fold_index, folds = comparator.PickBest(test_results)

print('Running Step 9/10: Test Inner Loop Best Combination on Outer Loop Holdout')
#Run best param set for each inner fold on corresponding outer fold
final_train_results, final_test_results = iterator.TestHoldout(oX_train, oX_test, oy_train, oy_test, fold_index) 

print('Running Step 10/10: PrintFinal')
#Print the results all purdy-like
final_average_train, final_average_test = comparator.PrintFinal(final_train_results, final_test_results)

