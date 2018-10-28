import torch
import cv2
from PIL import Image
import numpy as np

class DataProcessing:

    def __init__(self,data_path="/data"):
        self.data_path=data_path


    def load_image(self,path):
        im=Image.open(path)
        im=im.resize((im.width//4,im.height//4))
        img=np.asarray(im)
        return img


d=DataProcessing()
d.load_image("data/00885.ppm")