import os

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sli_dataset import SliDataSet
from utils import read_split_data


root = 'C:\\Users\\20211070\\Desktop\\code_mark\\code\\CDproject\\data_combi'  # dataset root directory


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)

    data_transform = {
        "train": transforms.Compose([
                                     
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456], [0.229, 0.224])]),
        "val": transforms.Compose([
                                   
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456], [0.229, 0.224])])}

    train_data_set = SliDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])

    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers'.format(nw))
    train_loader = DataLoader(train_data_set,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=nw,
                                collate_fn=train_data_set.collate_fn)

    # plot_data_loader_image(train_loader)

    for step, data in enumerate(train_loader):
        images, labels = data


if __name__ == '__main__':
    main()