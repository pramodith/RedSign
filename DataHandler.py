from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import os
import numpy as np
import argparse
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
import torch

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
        self.scale_factor1=1360/416
        self.scale_factor2=800/416
        #load the names of all files directory X and directory Y's files are named the same but are at different resolutions
        self.bbox_locations, self.bbox_labels = self.process_gt()
        #self.names=[os.path.join(path,f) for  f in os.listdir(self.path) if os.path.isfile(os.path.join(path,f))]
        self.names=list(self.bbox_locations.keys())[:500]
        self.names = [os.path.join(path, self.names[ind]) for ind in range(len(self.names))]
        self.transform=transform
        self.loaded_images=[]

    def __len__(self):
        return len(self.names)

    def process_gt(self):
        bbox_locations={}
        bbox_labels={}
        with open(self.gt_path,'r') as f:
            lines=f.readlines()
            for line in lines:
                columns=line.split(";")
                if columns[0] not in bbox_locations:
                    bbox_locations[columns[0]]=[[float(columns[1])//self.scale_factor1,float(columns[2])//self.scale_factor2,float(columns[3])//self.scale_factor1,
                                                 float(columns[4])//self.scale_factor2]]
                    bbox_labels[columns[0]]=1
                else:
                    bbox_locations[columns[0]].append([float(columns[1])//self.scale_factor1,float(columns[2])//self.scale_factor2,float(columns[3])//self.scale_factor1,
                                                       float(columns[4])//self.scale_factor2])

        return bbox_locations,bbox_labels


    def __getitem__(self, ind):
        img=Image.open(self.names[ind])
        #img=img.resize(img.width//self.scale_factor,img.height//self.scale_factor)
        if self.transform is not None:
            img=self.transform(img)
        if self.names[ind].split("\\")[1] in self.bbox_locations:
            return img,self.bbox_locations[self.names[ind].split("\\")[1]],self.bbox_labels[self.names[ind].split("\\")[1]]
        else:
            return img,[],0

def pad_uneven_number_bbox(batch):
    img=np.asarray([x[0].numpy() for x in batch])
    bbox=[x[1] for x in batch]
    label=np.asarray([x[2] for x in batch])
    bbox=np.asarray(bbox)
    x=[len(bbox[i]) if len(bbox)>0 else 0 for i in range(len(bbox))]
    longest=np.max(x)
    bboxes=np.zeros((len(batch),longest,4))
    for i in range(len(bbox)):
        for j in range(x[i]):
            bboxes[i,j]=bbox[i][j]
    label = torch.LongTensor(label)
    img=torch.FloatTensor(img)
    bboxes=torch.FloatTensor(bboxes)
    return img,bboxes,label

#function to create a mini-batch loader instance
def read_all_images(path,batch_size=4,num_workers=0):
    #the mean and std were calculated using numpy and then scaled down to 0-1 range
    dataset_mean = [0.4884048, 0.4982816, 0.50658032]
    dataset_std = [0.0909427*255, 0.0954222*255, 0.01157272*255]
    transform = transforms.Compose([
        #CLAHE(),
        transforms.Resize((416,416)),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std),
    ])
    data_handler = DataHandler("data/groundtruth/gt.txt", "data",transform)

    #set pin memory to true to cache the batches imporves speed.
    loader=DataLoader(data_handler,batch_size,True,num_workers=num_workers,pin_memory=True,collate_fn=pad_uneven_number_bbox)
    return loader

#read_all_images("data")