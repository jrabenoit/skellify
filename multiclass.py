#!/usr/bin/env python 

#Data prep modules, processing modules, and result reporting modules
import data, prep, crossval 
import iterator, comparator
import visualize

#Select a group of scans to use 
print('Running Step 1/10: SelectGroup')
group_1, size_1, files_1, group_2, size_2, files_2 = data.SelectGroup()

#Load two sets of scans into a dataset, mask data
print('Running Step 2/10: Sort')
sorted_files = prep.Sort(files_1, files_2)
print('Running Step 3/10: MaskFlatten')
data = prep.MaskFlatten(sorted_files)
print('Running Step 4/10: GroupLabels')
labels = prep.GroupLabels(size_1, size_2)

#Do 5-fold CV setup for outer fold (X=data, y=labels)
print('Running Step 5/10: oSkfCv')
oX_train, oX_test, oy_train, oy_test = crossval.oSkfCv(labels, data)

#Do 5-fold CV for each outer fold, creating 25 inner folds total
print('Running Step 6/10: iSkfCv')
iX_train, iX_test, iy_train, iy_test = crossval.iSkfCv(oy_train, oX_train)

#Run the data from iSkfCv through an iterative tool that tries all learning combinations
print('Running Step 7/10: ParameterSets')
test_results, param_set_list = iterator.ParameterSets(iX_train, iX_test, iy_train, iy_test)

#Compare all accuracy results to pick best combination of parameters
print('Running Step 8/10: PickBest')
fold_index, folds = comparator.PickBest(test_results)

print('Running Step 9/10: TestHoldout')
#Run best param set for each inner fold on corresponding outer fold
final_train_results, final_test_results = iterator.TestHoldout(oX_train, oX_test, oy_train, oy_test, fold_index) 

print('Running Step 10/10: PrintFinal')
#Print the results all purdy-like
final_average_train, final_average_test = comparator.PrintFinal(final_train_results, final_test_results)

