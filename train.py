from testResNet50 import resnet50
from custom_dataset.sli_dataset import SliDataSet
from custom_dataset.utils import read_split_data
from torch.utils.data import DataLoader
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456], [0.229, 0.224])]),
        "val": transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456], [0.229, 0.224])])}


    image_path = 'C:\\Users\\20211070\\Desktop\\code_mark\\code\\CDproject\\data_combi'
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(image_path)
    train_dataset = SliDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'B':0, 'M':1}
    tumor_class = ['B', 'M']
    class_indices = dict((k, v) for v, k in enumerate(tumor_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 3
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=nw,
                                collate_fn=train_dataset.collate_fn)

    val_dataset = SliDataSet(images_path=val_images_path,
                               images_class=val_images_label,
                               transform=data_transform["val"])
    val_num = len(val_dataset)
    validate_loader = DataLoader(val_dataset,
                                batch_size=batch_size, 
                                shuffle=False,
                                num_workers=nw,
                                )

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    
    net = resnet50()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./resnet50-19c8e357.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # 加载预训练模型并且把不需要的层去掉
    pre_state_dict = torch.load(model_weight_path)
    print("original_model", pre_state_dict.keys())
    new_state_dict = {}
    for k, v in net.state_dict().items():          # 遍历修改模型的各个层
        print("new_model", k)
        if k in pre_state_dict.keys() and k!= 'conv1.weight':
            new_state_dict[k] = pre_state_dict[k]  # 如果原模型的层也在新模型的层里面， 那新模型就加载原先训练好的权重
    net.load_state_dict(new_state_dict, False)

    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 2)   # change the out features 
    
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 100
    best_acc = 0.0
    save_path = './ResNet50.pth'
    train_steps = len(train_loader)
    train_loss = []
    validation_loss =[]
    validation_acc =[]
    epoch_pic = list(range(1, epochs+1))
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)
                # validation loss
                loss = loss_function(outputs, val_labels.to(device))
                val_loss0 = loss.item() * val_images.size(0)
                val_loss = val_loss0/len(validate_loader)

        validation_loss.append(val_loss)
        t_loss = running_loss / train_steps
        train_loss.append(t_loss)       
        val_accurate = acc / val_num
        validation_acc.append(val_accurate)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, t_loss, val_accurate))
        print('val_loss: %.3f' % (val_loss))
        
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    # features = net.forward_features(images)
    # print(features.shape)
    # print(validation_loss)
    # print(train_loss)
    print('Finished Training')
    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(epoch_pic , train_loss, label ='train_loss')
    plt.plot(epoch_pic, validation_loss, label = 'val_loss')
    plt.legend()
    plt.savefig('loss.png', dpi=300)

    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Val_acc")
    plt.plot(epoch_pic, validation_acc)
    plt.savefig('val_acc.png', dpi=300)


if __name__ == '__main__':
    main()
    