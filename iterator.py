#!/usr/bin/env python 

import itertools
import numpy as np
import copy
import featsel, decomp, mltools, results
import re

def ParameterSets(iX_train, iX_test, iy_train, iy_test):
#    param_set_list = list(itertools.product(featsel.feat_sel_dict.keys(),decomp.decomp_dict.keys(),mltools.ml_func_dict.keys()))    
    param_set_list = list(mltools.ml_func_dict.keys())
    train_results = {}
    test_results = {}
    for i in range(0,len(param_set_list)):
        print('  running iterator param set {0}/{1}: {2}'.format(i+1, len(param_set_list), param_set_list[i]))
        param_set = param_set_list[i]
####Random forest only, no featsel or decomp
        lX_train, lX_test= eval("mltools.ml_func_dict['{}'](iX_train, iX_test, iy_train, iy_test)".format(param_set))
        ir_train, ir_test = results.InnerAverages(lX_train, lX_test, param_set)
        train_results.update(ir_train)
        test_results.update(ir_test)  

    return test_results, param_set_list

'''
        fX_train, fX_test, fy_train, fy_test = eval("featsel.feat_sel_dict['{}'](iX_train, iX_test, iy_train, iy_test)".format(param_set[0]))
        dX_train, dX_test, dy_train, dy_test = eval("decomp.decomp_dict['{}'](fX_train, fX_test, fy_train, fy_test)".format(param_set[1]))
        lX_train, lX_test= eval("mltools.ml_func_dict['{}'](dX_train, dX_test, dy_train, dy_test)".format(param_set[2]))
        ir_train, ir_test = results.InnerAverages(lX_train, lX_test, param_set)
        train_results.update(ir_train)
        test_results.update(ir_test)          
'''
   

   
def TestHoldout(oX_train, oX_test, oy_train, oy_test, fold_index):
    hX_train, hX_test = copy.copy(oX_train), copy.copy(oX_test)
    hy_train, hy_test = copy.copy(oy_train), copy.copy(oy_test)

    params_fold_1 = list(re.compile('\w+').findall(fold_index[0]))
    params_fold_2 = list(re.compile('\w+').findall(fold_index[1]))
    params_fold_3 = list(re.compile('\w+').findall(fold_index[2]))
    params_fold_4 = list(re.compile('\w+').findall(fold_index[3]))
    params_fold_5 = list(re.compile('\w+').findall(fold_index[4]))
    
    final_params = [params_fold_1, params_fold_2, params_fold_3, params_fold_4, params_fold_5]

    final_train_results= {}
    final_test_results= {}
    #Using list of lists so I know that the labels remain ordered
    final_train_predictions= []
    final_test_predictions= []
    final_train_labels=[]
    final_test_labels=[]
    
    for i in range(0,5):
        print('  Final params, fold {0}/{1}: {2}'.format(i+1, len(final_params), final_params[i][1:4]))
        param_set = final_params[i]
####Random Forest Only, no featsel or decomp
        lX_train, lX_test, lX_train_predict, lX_test_predict, ly_train_labels, ly_test_labels = eval("mltools.ml_func_dict_final['{}'](hX_train[i], hX_test[i], hy_train[i], hy_test[i])".format(str(param_set[1])))
        
        oR_train, oR_test = results.OuterAverages(lX_train, lX_test, param_set)
        final_train_results.update(oR_train)
        final_test_results.update(oR_test)
        final_train_predictions.append(lX_train_predict)
        final_test_predictions.append(lX_test_predict)
        final_train_labels.append(ly_train_labels)
        final_test_labels.append(ly_test_labels)        

    return final_train_results, final_test_results, final_train_predictions, final_test_predictions, final_train_labels, final_test_labels
'''       
        fX_train, fX_test, fy_train, fy_test = eval("featsel.feat_sel_dict_final['{}'](hX_train[i], hX_test[i], hy_train[i], hy_test[i])".format(str(param_set[1])))
        dX_train, dX_test, dy_train, dy_test = eval("decomp.decomp_dict_final['{}'](fX_train, fX_test, fy_train, fy_test)".format(str(param_set[2])))
        lX_train, lX_test, lX_train_predict, lX_test_predict, ly_train_labels, ly_test_labels = eval("mltools.ml_func_dict_final['{}'](dX_train, dX_test, dy_train, dy_test)".format(str(param_set[3])))
'''
