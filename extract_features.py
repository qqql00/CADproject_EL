from testResNet50 import resnet50
from custom_dataset.sli_dataset import SliDataSet
from custom_dataset.utils import read_split_data
from img_path_label import get_imgpath
from torch.utils.data import DataLoader
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

data_transform = {
        "train": transforms.Compose([transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456], [0.229, 0.224])]),
        "val": transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456], [0.229, 0.224])])}
image_root = 'C:\\Users\\20211070\\Desktop\\code_mark\\code\\CDproject\\data_combi'  # the path of combination
images_path, images_label = get_imgpath(image_root)
w_dataset = SliDataSet(images_path=images_path,
                               images_class=images_label,
                               transform=data_transform["train"])
train_num = len(w_dataset)
batch_size = 1
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

w_loader = DataLoader(w_dataset,
                     batch_size=batch_size,
                     shuffle=True,
                     num_workers=nw,
                     collate_fn=w_dataset.collate_fn)


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model = resnet50(num_classes=2)
# load the pre-trained weights
model_weight_path = "./ResNet50.pth"
assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))


model.avgpool.register_forward_hook(get_activation('avgpool'))
train_bar = tqdm(w_loader, file=sys.stdout)
features_np = []
for step, data in enumerate(train_bar):
    # image: batch_size*2*512*512
    images, labels = data
    # x: 1*2
    x = model(images)
    # print(x.shape)
    features_t = torch.flatten(activation['avgpool'])
    # print(features_t.shape)
    fea_np = features_t.numpy() #convert to Numpy array
    # print(np.shape(fea_np))
    features_np.append(fea_np)
    # print(np.shape(features_np))
    # df = pd.DataFrame(fea_np) #convert to a dataframe
    # df.to_csv("testfile",index=False) #save to file

df = pd.DataFrame(features_np)
df.to_csv("features_ResNet50.csv", index= False)
df_label = pd.DataFrame(images_label)
df_label.to_csv("label_ResNet50.csv", index=False)
    


# x = torch.randn(3, 2, 512, 512)
# output = model(x)
# out_test = torch.flatten(activation['avgpool'])