from ast import Tuple
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
import os
import torch
from torch.nn.functional import normalize
import cv2
from pathlib import Path, PosixPath

PATH = os.path.dirname(__file__)
ORI_PATH = os.path.join(PATH,'ovdata')
DATA_PATH = os.path.join(PATH,'slices')
NEW_IMG_PATH = os.path.join(DATA_PATH,'IMAGES')
NEW_M_PATH = os.path.join(DATA_PATH,'MASKS')


def get_mpath(file_name):

    tumor_name = os.path.splitext(file_name)[0]
    tumor_id = os.path.splitext(tumor_name)[0]
    m_path = os.path.join(ORI_PATH, tumor_id, 'MASKS', file_name)
    return m_path

def get_mid_m(mask_path):

    my_nii = nib.load(mask_path).get_fdata()
    fron_proj = my_nii[:,:,:].sum(axis=0)
    z_proj = fron_proj[:,:].sum(axis=0)
    a = np.nonzero(z_proj)
    b = np.count_nonzero(z_proj)
    z_m = a[0][0] + int(b/2) 
    return z_m

def get_slices(path, output_path, file, z_m): 
    img_out0 = os.path.join(output_path, file)
    img_out1 = os.path.splitext(img_out0)[0]
    img_out2 =os.path.splitext(img_out1)[0]
    img_out = img_out2 + '.png'

    nii = nib.load(path).get_fdata()
    nii_slices = nii[:,:,z_m]
    # np.save(img_out, nii_slices)
    # img_out = os.path.join(img_out2, '.png')
    nii_slices = nii_slices * 255
    cv2.imwrite(img_out, nii_slices)
    print(img_out)
    return nii_slices


def n_stack(i_sli, m_sli):

    i_sli_t = torch.from_numpy(i_sli)
    m_sli_t = torch.from_numpy(m_sli)
    i_sli_n = torch.nn.functional.normalize(i_sli_t, p=2.0, dim=1, eps=1e-12, out=None)
    m_sli_n = torch.nn.functional.normalize(m_sli_t, p=2.0, dim=1, eps=1e-12, out=None)
    ten_combi = torch.stack(Tuple[i_sli_n, m_sli_n], dim=2)

    return ten_combi

for dir in os.listdir(ORI_PATH):
    # child_dir : B01, B04, B05, ..., M67, M68
    child_dir = os.path.join(ORI_PATH, dir)
    for d in os.listdir(child_dir):
        # CONTENTS ： IMAGES, MASKS
        # print(d)
        if d == 'IMAGES':
            ddd_img= os.path.join(ORI_PATH, dir ,d)
            for file in os.listdir(ddd_img):
                
                img_data_path = os.path.join(ORI_PATH, dir, d, file)
                # fileName = os.path.splitext(file)[0] + '\n'
                m_p = get_mpath(file)
                zm_i = get_mid_m(m_p)
                get_slices(img_data_path, NEW_IMG_PATH, file, zm_i)
                print(img_data_path)
        else:
            ddd_m = os.path.join(ORI_PATH, dir, d)
            for file in os.listdir(ddd_m):
                
                m_data_path = os.path.join(ORI_PATH, dir, d, file)
                # fileName = os.path.splitext(file)[0] + '\n'
                zm_m = get_mid_m(m_data_path)
                get_slices(m_data_path, NEW_M_PATH, file, zm_m)
                # print(img_data_path)