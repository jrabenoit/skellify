#!/usr/bin/env python 

import os, os.path

# Dict for each group of form: group_keys={key:['description', 'directory location']}
group_keys={
    1: ['Healthy Controls', '/home/james/Desktop/depression/10_Skellify_data/healthy_controls/'],
    2: ['Low-Moderate MDD Group', '/home/james/Desktop/depression/10_Skellify_data/low_HAMD/'],
    3: ['Severe MDD Group', '/home/james/Desktop/depression/10_Skellify_data/med_HAMD/'],
    4: ['Very Severe MDD Group', '/home/james/Desktop/depression/10_Skellify_data/high_HAMD/'],
    5: ['Patients with >=50% Reduction in HAM-D Score', '/home/james/Desktop/depression/10_Skellify_data/responder/'],
    6: ['Patients with <7 HAM-D Score', '/home/james/Desktop/depression/10_Skellify_data/remitter/'],
    7: ['Patients with <50% Reduction in HAM-D Score', '/home/james/Desktop/depression/10_Skellify_data/nonresponder/'],
    8: ['Patients with La/La type 5-HT TLPR', '/home/james/Desktop/depression/10_Skellify_data/lala_tlpr/'],
    9: ['Patients with Sa/Sa + Sa/Lg type 5-HT TLPR', '/home/james/Desktop/depression/10_Skellify_data/sasa_salg_tlpr/'],
    10: ['Patients with La/Sa + La/Lg type 5-HT TLPR', '/home/james/Desktop/depression/10_Skellify_data/lasa_lalg_tlpr/']
    }

# Mask file for skeletonized DTI scans
mask = '/home/james/Desktop/depression/07_Machine_Learning/02_FA_Skeletonized/mean_FA_skeleton_mask.nii.gz'

def SelectGroup():
    print('Groups:')
    for key, value in group_keys.items():
        print("    {}) {}".format(key, value[0]))    

    choice_1 = int(input('Choose first group, then press Enter: '))
    choice_2 = int(input('Choose second group, then press Enter: '))

    dir_1 = str(group_keys[choice_1][1]) + '*.nii.gz'
    dir_2 = str(group_keys[choice_2][1]) + '*.nii.gz'

    n_1 = len([name for name in os.listdir(str(group_keys[choice_1][1])) if os.path.isfile(os.path.join(str(group_keys[choice_1][1]), name))])
    n_2 = len([name for name in os.listdir(str(group_keys[choice_2][1])) if os.path.isfile(os.path.join(str(group_keys[choice_2][1]), name))])
    
    print('\nThe two groups selected are: {} and {}.\n'.format(group_keys[choice_1][0], group_keys[choice_2][0]))

    return n_1, n_2, dir_1, dir_2, mask       
