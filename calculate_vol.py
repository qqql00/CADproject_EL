import nibabel as nib
from matplotlib import pyplot as plt
import numpy as np
import os

L_path = 'C:\\Users\\20211070\\Desktop\\code_mark\\code\\CDproject\\ovdata\\M67\\MASKS\\M67_LM.nii.gz'
R_path = 'C:\\Users\\20211070\\Desktop\\code_mark\\code\\CDproject\\ovdata\\M67\\MASKS\\M67_RM.nii.gz'

def get_volume(l_path, r_path):
    l_nii = nib.load(l_path)
    l_nii1 =l_nii.get_fdata()
    # count the number of voxels 
    nonzero_voxel_count_l = np.count_nonzero(l_nii1[:,:,:])
    # calculate the volume of a single voxel ---> lx * ly * lz
    lx, ly, lz = l_nii.header.get_zooms()
    # get the total volume of mask
    l_vol = lx * ly * lz * nonzero_voxel_count_l

    r_nii = nib.load(r_path)
    r_nii1 = r_nii.get_fdata()
    nonzero_voxel_count_r = np.count_nonzero(r_nii1[:,:,:])
    rx, ry, rz = r_nii.header.get_zooms()
    r_vol = rx * ry * rz * nonzero_voxel_count_r

    if l_vol > r_vol:
        max_name = os.path.basename(l_path)
        return max_name, l_vol, r_vol
    else:
        max_name = os.path.basename(r_path)
        return max_name, l_vol, r_vol


def get2sli(l_path, r_path):
    l_nii = nib.load(l_path).get_fdata()
    l_sli = l_nii[:,:,40]
    r_nii = nib.load(r_path).get_fdata()
    r_sli =r_nii[:,:,40]
    return l_sli, r_sli

max_p, vl_12, vr_12 = get_volume(L_path, R_path)
print(max_p)
print(vl_12)
print(vr_12)
# L_sli, R_sli = get2sli(L_path, R_path)
# plt.imshow(R_sli)
# plt.show()