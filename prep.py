#!/usr/bin/env python 

import glob
import copy
import numpy as np
from nilearn.input_data import NiftiMasker
from sklearn import preprocessing

def Sort(dir_1, dir_2):
    '''Sorting with output: alphabetized set 1, alphabetized set 2'''
    dataset_1 = sorted(glob.glob(copy.copy(dir_1)))
    dataset_2 = sorted(glob.glob(copy.copy(dir_2)))
    sorted_files = dataset_1 + dataset_2
    return sorted_files

def MaskFlatten(sorted_files, mask):
    '''Mask image data, convert to 2D feature matrix'''
    nifti_masker = NiftiMasker(mask_img=mask)
    masked_data = nifti_masker.fit_transform(sorted_files)
    return masked_data

def GroupLabels(size_1, size_2):
    '''Create two-class label set matching data input'''
    array_1 = np.zeros(int(size_1), dtype=np.int8)
    array_2 = np.ones(int(size_2), dtype=np.int8)
    labels = np.append(array_1, array_2)
    return labels
