import torch
import os
from torchvision import transforms
from PIL import Image

PATH = os.path.dirname(__file__)
pre_ipath = os.path.join(PATH,'data','IMAGES')
pre_mpath = os.path.join(PATH,'data','MASKS')
od_path = os.path.join(PATH,'data_combi') # output path
trans = transforms.ToTensor()
toPIL = transforms.ToPILImage()

def get_combi(i_path, m_path, o_path):
    for cla in os.listdir(i_path):
        if cla == 'B' :
            img_path = os.path.join(i_path, cla)
            for image_name in os.listdir(img_path) :
                image_path = os.path.join(img_path, image_name)
                mask_path = os.path.join(m_path, cla, image_name)
                out_path = os.path.join(o_path, cla, image_name)

                image_t = trans(Image.open(image_path))
                mask_t = trans(Image.open(mask_path))
                combi_t = torch.cat((image_t, mask_t), 0)
                

                combi_img = toPIL(combi_t)
                combi_img.save(out_path)
        if cla == 'BL':
            img_path = os.path.join(i_path, cla)
            for image_name in os.listdir(img_path) :
                image_path = os.path.join(img_path, image_name)
                mask_path = os.path.join(m_path, cla, image_name)
                out_path = os.path.join(o_path, cla, image_name)

                image_t = trans(Image.open(image_path))
                mask_t = trans(Image.open(mask_path))
                print(image_t.shape)
                print(mask_t.shape)
                combi_t = torch.cat((image_t, mask_t), 0)

                combi_img = toPIL(combi_t)
                combi_img.save(out_path)
        else:
            img_path = os.path.join(i_path, cla)
            for image_name in os.listdir(img_path) :
                image_path = os.path.join(img_path, image_name)
                mask_path = os.path.join(m_path, cla, image_name)
                out_path = os.path.join(o_path, cla, image_name)

                image_t = trans(Image.open(image_path))
                mask_t = trans(Image.open(mask_path))
                combi_t = torch.cat((image_t, mask_t), 0)

                combi_img = toPIL(combi_t)
                combi_img.save(out_path)


    return combi_t

if __name__ == '__main__':
    get_combi(pre_ipath,pre_mpath,od_path)
