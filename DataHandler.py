from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import os
import numpy as np
import argparse
from PIL import Image
from torch import Tensor
from torchvision import transforms
import cv2

class CLAHE:
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def __call__(self, im):
        img_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output

class DataHandler(Dataset):

    def __init__(self,gt_file_path,path,transform=None,scale_factor=1):
        self.path=path
        self.gt_path = gt_file_path
        self.scale_factor=scale_factor
        #load the names of all files directory X and directory Y's files are named the same but are at different resolutions
        self.names=[os.path.join(path,f) for  f in os.listdir(self.path) if os.path.isfile(os.path.join(path,f))]
        self.transform=transform
        self.bbox_locations,label=self.process_gt()
        self.loaded_images=[]

    def __len__(self):
        return len(self.names)

    def process_gt(self):
        label=0
        bbox_locations={}
        with open(self.gt_path,'r') as f:
            lines=f.readlines()
            for line in lines:
                columns=line.split(";")
                if columns[0] not in bbox_locations:
                    bbox_locations[columns[0]]=[[float(columns[1])//self.scale_factor,float(columns[2])//self.scale_factor,float(columns[3])//self.scale_factor,
                                                 float(columns[4])//self.scale_factor]]
                    label=1
                else:
                    bbox_locations[columns[0]].append([float(columns[1])//self.scale_factor,float(columns[2])//self.scale_factor,float(columns[3])//self.scale_factor,
                                                       float(columns[4])//self.scale_factor])

        return bbox_locations,label


    def __getitem__(self, ind):
        img=Image.open(self.names[ind])
        img=img.resize(img.width//self.scale_factor,img.height//self.scale_factor)
        if self.transform is not None:
            img=self.transform(img)
        if self.names[ind].split("\\")[1] in self.bbox_locations:
            return img,self.bbox_locations[self.names[ind].split("\\")[1]]
        else:
            return img,[]

#function to create a mini-batch loader instance
def read_all_images(path,batch_size=32,num_workers=0):
    #the mean and std were calculated using numpy and then scaled down to 0-1 range
    dataset_mean = [0.4884048, 0.4982816, 0.50658032]
    dataset_std = [0.0909427*255, 0.0954222*255, 0.01157272*255]
    transform = transforms.Compose([
        CLAHE(),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std),
    ])
    data_handler = DataHandler("data/groundtruth/gt.txt", "data",transform)


    #set pin memory to true to cache the batches imporves speed.
    loader=DataLoader(data_handler,batch_size,True,num_workers=num_workers,pin_memory=True)
    for img,label in loader:
        print(img)
        print(label)
    return loader

read_all_images("data")