#!/usr/bin/env python 

# Dict for each group of form: group_keys={key:['description', n, 'directory location']}
group_keys={
    1: ['Healthy Controls', 18, '/home/james/Desktop/depression/10_Skellify_data/healthy_controls/*.nii.gz'],
    2: ['Low-Moderate MDD Group', 16, '/home/james/Desktop/depression/10_Skellify_data/low_HAMD/*.nii.gz'],
    3: ['Severe MDD Group', 19, '/home/james/Desktop/depression/10_Skellify_data/med_HAMD/*.nii.gz'],
    4: ['Very Severe MDD Group', 17, '/home/james/Desktop/depression/10_Skellify_data/high_HAMD/*.nii.gz'],
    5: ['Patients with >=50% Reduction in HAM-D Score', 14, '/home/james/Desktop/depression/10_Skellify_data/responder/*.nii.gz'],
    6: ['Patients with <7 HAM-D Score', 19, '/home/james/Desktop/depression/10_Skellify_data/remitter/*.nii.gz'],
    7: ['Patients with <50% Reduction in HAM-D Score', 12, '/home/james/Desktop/depression/10_Skellify_data/nonresponder/*.nii.gz'],
    8: ['Patients with La/La type 5-HT TLPR', 9, '/home/james/Desktop/depression/10_Skellify_data/lala_tlpr/*.nii.gz'],
    9: ['Patients with Sa/Sa + Sa/Lg type 5-HT TLPR', 14, '/home/james/Desktop/depression/10_Skellify_data/sasa_salg_tlpr/*.nii.gz'],
    10: ['Patients with La/Sa + La/Lg type 5-HT TLPR', 20, '/home/james/Desktop/depression/10_Skellify_data/lasa_lalg_tlpr/*.nii.gz']
    }

# Mask file for skeletonized DTI scans
mask = '/home/james/Desktop/depression/07_Machine_Learning/02_FA_Skeletonized/mean_FA_skeleton_mask.nii.gz'

def SelectGroup():
    print('Groups:')
    for key, value in group_keys.items():
        print("    {}) {}".format(key, value[0]))    

    choice_1 = int(input('Choose first group: '))
    choice_2 = int(input('Choose second group: '))
    
    n_1 = group_keys[choice_1][1] 
    n_2 = group_keys[choice_2][1]

    dir_1 = group_keys[choice_1][2]
    dir_2 = group_keys[choice_2][2]

    print('\nThe two groups selected are: {} and {}.\n'.format(group_keys[choice_1][0], group_keys[choice_2][0]))

    return n_1, n_2, dir_1, dir_2, mask       
