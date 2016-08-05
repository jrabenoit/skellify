#!/usr/bin/env python 

import numpy as np
from collections import defaultdict
import pprint, itertools
import random

def EmpiricalDistro(n_1, n_2, per_subject_test_acc):
    sample_size = n_1 + n_2
    sample_distro = []
    n_iterations = 10000
    for i in range(n_iterations):
        sample_list = []
        sample_list = np.random.choice(list(per_subject_test_acc.values()), sample_size,replace=True) 
        sample_mean = sum(sample_list)/len(sample_list)
        sample_distro.append(sample_mean)
    
    p_value = sum(i <= 50.0 for i in sample_distro)/n_iterations
    
    return p_value
