import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2

file_pp = 'C:\\Users\\20211070\\Desktop\\code_mark\\code\\ovdata\\B01\\IMAGES\\B01.nii.gz'
sli_img = 'C:\\Users\\20211070\\Desktop\\code_mark\\code\\CDproject\\data\\IMAGES\\B\\B01.png'
center = 60
width = 40

# rectangular window : center, width ---- soft tissue range: (-160, 240)
def windowing(path, center, width):
    min = (2*center - width)/2.0 +0.5
    max = (2*center + width)/2.0 +0.5
    dfactor = 255.0/(max - min)

    my_nii = nib.load(path).get_fdata()
    slice = my_nii[:,:,60]
    slice1 = slice - min
    slice1 = np.trunc(slice1*dfactor)
    slice1[slice1<0.0] = 0
    slice1[slice1 > 255.0] = 255

    return slice,slice1

test_slice, test_slice1 = windowing(file_pp, center, width)
slice0 = test_slice* 255
print(test_slice.dtype)
print(test_slice.min())
print(test_slice.max())
print(test_slice1.dtype)
print(test_slice1.min())
print(test_slice1.max())
print(slice0.dtype)
print(slice0.min())
print(slice0.max())
#plt.imshow(test_slice1)
# plt.savefig('test_slice_B01.png', dpi=300)
#plt.show()
test_img = cv2.imread(sli_img)
print(test_img.dtype)
print(test_img.min())
print(test_img.max())
plt.imshow(test_slice1)
plt.show()


