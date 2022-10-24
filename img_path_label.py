import os
import json
import pickle
import random
import matplotlib.pyplot as plt

def get_imgpath(root):
    tumor_class = ['B', 'M']

    class_index = dict((k, v) for v, k in enumerate(tumor_class))
    json_str = json.dumps(dict((val, key) for key, val in class_index.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    images_path = []
    images_label = [] 
    every_class_num = []  # store sample total numbers of each class
        # traverse folders
    for cla in tumor_class:
        cla_path = os.path.join(root, cla)
        # traverse and get image paths for each class
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)]
        # get the index for this class 
        image_class = class_index[cla]
        # record the sample numbers of this class 
        every_class_num.append(len(images))
        for img_path in images:  
                images_path.append(img_path)
                images_label.append(image_class)
    return images_path, images_label

if __name__ == '__main__':
    # the directory of dataset after slices combination
    image_root = 'C:\\Users\\20211070\\Desktop\\code_mark\\code\\CDproject\\data_combi'   
    i_path,i_label = get_imgpath(image_root)
    print(i_path)
    print(i_label)