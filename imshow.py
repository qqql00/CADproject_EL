import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D

file = 'C:\\Users\\20211070\\Desktop\\code_mark\\code\\ovdata\\B01\\MASKS\\B01.nii.gz'    #the path of nii.gz file
path_p = 'C:\\Users\\20211070\\Desktop\\code_mark\\code\\CDproject\\dataset\\MASKS\\B01.npy'
# file1 = 'C:\\Users\\20211070\\Desktop\\test_json\\dataset\\image\\M63.nii.gz'
file1 = 'C:\\Users\\20211070\\Desktop\\code_mark\\code\\ovdata\\M68\\IMAGES\\M68.nii.gz'
""" 
img = nib.load(file)    

print(img)
print(img.header['db_name'])    # print header


width, height, queue = img.dataobj.shape

OrthoSlicer3D(img.dataobj).show() 
"""

'''''
num = 1
for i in range(0, queue, 10):
    img_arr = img.dataobj[:,:,i]
    plt.subplot(5,4,num)
    plt.imshow(img_arr, cmap='gray')
    num += 1
'''''
# plt.show()

def show3d(raw_path):
    img = nib.load(file)    
    i_3d = OrthoSlicer3D(img.dataobj)
    i_s = img.dataobj.shape
    return i_3d, i_s


""" def show_sli(sli_p):
    sli = np.load(sli_p)
    return sli """

# show3d(file).show()
img = nib.load(file)
print(img)
print(img.header['db_name']) 
test1 = OrthoSlicer3D(img.dataobj)
print(img.dataobj.shape)
print(img.dataobj.slope, img.dataobj.inter)
# plt.imshow(sli_t)
plt.show()


