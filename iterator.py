#!/usr/bin/env python 

import itertools
import numpy as np
import copy
import featsel, decomp, mltools, results
import re

def ParameterSets(iX_train, iX_test, iy_train, iy_test):
    sX_train = copy.copy(iX_train) 
    sX_test = copy.copy(iX_test)
    sy_train =  copy.copy(iy_train)
    sy_test = copy.copy(iy_test)

#    param_set_list = list(itertools.product(('SelKBest1','SelKBest2', 'SelKBest3'),('RPca1','RPca2','NullDecomp'),mltools.ml_func_dict.keys()))
    param_set_list = list(itertools.product(featsel.feat_sel_dict.keys(),decomp.decomp_dict.keys(),mltools.ml_func_dict.keys()))
 
    
    train_results = {}
    test_results = {}
    for i in range(0,len(param_set_list)):
        print('  running iterator param set {0}/{1}: {2}'.format(i+1, len(param_set_list), param_set_list[i]))
        param_set = param_set_list[i]

        fX_train, fX_test, fy_train, fy_test = eval("featsel.feat_sel_dict['{}'](sX_train, sX_test, sy_train, sy_test)".format(param_set[0]))
        dX_train, dX_test, dy_train, dy_test = eval("decomp.decomp_dict['{}'](fX_train, fX_test, fy_train, fy_test)".format(param_set[1]))
        lX_train, lX_test = eval("mltools.ml_func_dict['{}'](dX_train, dX_test, dy_train, dy_test)".format(param_set[2]))
        
        ir_train, ir_test = results.InnerAverages(lX_train, lX_test, param_set)
        train_results.update(ir_train)
        test_results.update(ir_test)    
        
    return test_results, param_set_list

   
def TestHoldout(oX_train, oX_test, oy_train, oy_test, fold_index):
    hX_train, hX_test = copy.copy(oX_train), copy.copy(oX_test)
    hy_train, hy_test = copy.copy(oy_train), copy.copy(oy_test)

    params_fold_1 = list(re.compile('\w+').findall(fold_index[0]))
    params_fold_2 = list(re.compile('\w+').findall(fold_index[1]))
    params_fold_3 = list(re.compile('\w+').findall(fold_index[2]))
    params_fold_4 = list(re.compile('\w+').findall(fold_index[3]))
    params_fold_5 = list(re.compile('\w+').findall(fold_index[4]))
    
    final_params = [params_fold_1, params_fold_2, params_fold_3, params_fold_4, params_fold_5]

    final_train_results = {}
    final_test_results = {}
    for i in range(0,5):
        print('  running final param set, fold {0}/{1}: {2}'.format(i+1, len(final_params), final_params[i][1:4]))
        
        param_set = final_params[i]
        
        fX_train, fX_test, fy_train, fy_test = eval("featsel.feat_sel_dict_final['{}'](hX_train[i], hX_test[i], hy_train[i], hy_test[i])".format(str(param_set[1])))
        dX_train, dX_test, dy_train, dy_test = eval("decomp.decomp_dict_final['{}'](fX_train, fX_test, fy_train, fy_test)".format(str(param_set[2])))
        lX_train, lX_test = eval("mltools.ml_func_dict_final['{}'](dX_train, dX_test, dy_train, dy_test)".format(str(param_set[3])))
        
        or_train, or_test = results.OuterAverages(lX_train, lX_test, param_set)
        final_train_results.update(or_train)
        final_test_results.update(or_test)

    return final_train_results, final_test_results
