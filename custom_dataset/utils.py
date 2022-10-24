import os
import json
import pickle
import random

import matplotlib.pyplot as plt


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # ensure random results are reproducible
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # traverse folders, one folder corrensponds to one category
    tumor_class = ['B', 'M']
    
    # tumor_class.sort()
    # generate cateogry name and corresponding numerical index
    class_indices = dict((k, v) for v, k in enumerate(tumor_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # store all image paths of the train set
    train_images_label = []  # store the index of images in train set
    val_images_path = []  # store all image paths of the validation set
    val_images_label = []  # store the index of images in validation set
    every_class_num = []  # store the total number of samples for each class
    # 
    # traverse folders
    for cla in tumor_class:
        cla_path = os.path.join(root, cla)
        # tracerse and get all image paths
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)]
        # get the index for this class
        image_class = class_indices[cla]
        # record the sample number of this class
        every_class_num.append(len(images))
        # sampling val set according to the val_rate
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))



    return train_images_path, train_images_label, val_images_path, val_images_label

