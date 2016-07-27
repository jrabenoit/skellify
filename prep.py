#!/usr/bin/env python 

import pprint
import random
import glob
import copy
import numpy as np
from nilearn.input_data import NiftiMasker
from sklearn import preprocessing

def Sort(dir_1, dir_2, n_1, n_2):
    '''Create dict of subject file list of both groups'''
    files_1 = glob.glob(dir_1)
    files_2 = glob.glob(dir_2)
    random.shuffle(files_1)
    random.shuffle(files_2)
#From this point on the dict is set in order, changed only by if statement below    
    n_list = [n_1, n_2]
    iter_n = max(n_list)
    concat_dict = {}
    g_1 = files_1
    g_2 = files_2
   
    for i in range(iter_n):  
        if n_1 > n_2:
            files_1.insert(0, g_1.pop()) 
            g_1 = files_1[:n_2]
            g_2 = files_2
            g_concat = g_1 + g_2
            concat_dict[i] = g_concat 
        elif n_1 < n_2:
            files_2.insert(0, g_2.pop()) 
            g_1 = files_1
            g_2 = files_2[:n_1]
            g_concat = g_1 + g_2
            concat_dict[i] = g_concat                    
        else:
            files_1.insert(0, g_1.pop())
            files_2.insert(0, g_2.pop())
            g_1 = files_1
            g_2 = files_2
            g_concat = g_1 + g_2
            concat_dict[i] = g_concat

    concat_subjects_dict = concat_dict

    return concat_dict, concat_subjects_dict, iter_n
'''
    dataset_1 = sorted(glob.glob(dir_1))
    dataset_2 = sorted(glob.glob(dir_2))
    sorted_files = dataset_1 + dataset_2
    return concat_files
'''

def MaskFlatten(concat_dict, mask, iter_n):
    '''Mask image data, convert to 2D feature matrix'''
    nifti_masker = NiftiMasker(mask_img=mask)
    masked_dict = {}
    for i in range(iter_n):
        masked_dict[i] = nifti_masker.fit_transform(concat_dict[i])
    return masked_dict

def ZNormalize(masked_dict, iter_n):
    '''Will this work with sparse data? See 4.3.1.2'''
    stdscaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
    znorm_dict = {}
    for i in range(iter_n):
        znorm_dict[i] = stdscaler.fit_transform(masked_dict[i])
    return znorm_dict

def GroupLabels(n_1, n_2, iter_n):
    '''Create two-class label set matching data input'''
    group_size = min(n_1, n_2)
    group_label_dict = {}
    for i in range(iter_n):
        array_1 = np.zeros(group_size, dtype=np.int8)
        array_2 = np.ones(group_size, dtype=np.int8)
        labels = np.append(array_1, array_2)
        group_label_dict[i] = labels
    return group_label_dict
    
