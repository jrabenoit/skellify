#!/usr/bin/env python 
'''
####Visualization
coef = ML.coef_
coef = feature_selection.inverse_transform(coef)
weight_img = nifti_masker.inverse_transform(coef)
plot_stat_map(weight_img, title="ML weights")
nibabel.save(weight_img, '/home/james/Desktop/skellify_{0}_weights.nii'.format(groupSelected))
plt.show()
'''
