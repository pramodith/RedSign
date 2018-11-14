from __future__ import division
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils import *
from DataHandler import read_all_images


def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """

    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  # store the lines in a list
    lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']  # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # This marks the start of a new block
            if len(block) != 0:  # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)  # add it the blocks list
                block = {}  # re-init the block
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def create_modules(blocks):
    net_info = blocks[0]  # Captures the information about the input and pre-processing
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        # check the type of block
        # create a new module for the block
        # append to module_list

        # If it's a convolutional layer
        if (x["type"] == "convolutional"):
            # Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            # Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # Check the activation.
            # It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

            # If it's an upsampling layer
            # We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            module.add_module("upsample_{}".format(index), upsample)

        # If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            # Start  of a route
            start = int(x["layers"][0])
            # end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            # Positive anotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        elif x["type"] == "maxpool":
            stride = int(x["stride"])
            size = int(x["size"])
            if stride != 1:
                maxpool = nn.MaxPool2d(size, stride)
            else:
                maxpool = MaxPoolStride1(size)

            module.add_module("maxpool_{}".format(index), maxpool)

        # shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        # Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = YOLO(anchors)
            module.add_module("YOLO_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)

class YOLO(nn.Module):

    def __init__(self,anchors):
        super(YOLO,self).__init__()
        self.anchors=anchors

    def forward(self,input):
        prediction, inp_dim, anchors, num_classes,_=input
        CUDA=True
        batch_size = prediction.size(0)
        stride = inp_dim // prediction.size(2)
        grid_size = inp_dim // stride
        bbox_attrs = 5 + num_classes
        num_anchors = len(anchors)

        prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
        prediction = prediction.transpose(1, 2).contiguous()
        prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
        anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

        # Sigmoid the  centre_X, centre_Y. and object confidencce
        prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
        prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
        prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

        # Add the center offsets
        grid = np.arange(grid_size)
        a, b = np.meshgrid(grid, grid)

        x_offset = torch.FloatTensor(a).view(-1, 1)
        y_offset = torch.FloatTensor(b).view(-1, 1)

        if CUDA:
            x_offset = x_offset.cuda()
            y_offset = y_offset.cuda()

        x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

        prediction[:, :, :2] += x_y_offset

        # log space transform height and the width
        anchors = torch.FloatTensor(anchors)

        if CUDA:
            anchors = anchors.cuda()

        anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
        prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

        prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

        prediction[:, :, :4] *= stride

        return prediction

class MaxPoolStride1(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x):
        padded_x = F.pad(x, (0, self.pad, 0, self.pad), mode="replicate")
        pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        return pooled_x

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x,CUDA):
        modules = self.blocks[1:]
        outputs = {}  # We cache the outputs for the route layer

        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample" or module_type=='maxpool':
                x = self.module_list[i](x)

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i+layers[1]]
                    x = torch.cat((map1, map2), 1)


            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                # Get the input dimensions
                inp_dim = int(self.net_info["height"])

                # Get the number of classes
                num_classes = int(module["classes"])

                # Transform

                x = self.module_list[i]([x, inp_dim, anchors, num_classes, CUDA])
                if not write:  # if no collector has been intialised.
                    detections = x
                    write = 1

                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x
        bboxes=write_results(detections, 0.4, 1, nms_conf=0.4)
        return bboxes


def train(epochs=100):
    num_classes=1
    total_loss=0
    loss_coordinates=nn.MSELoss()
    loss_confidence=nn.BCELoss()
    model=Darknet('cfg/yolov3-tiny.cfg')
    optim = torch.optim.Adam(model.parameters())
    if torch.cuda.is_available():
        model=model.cuda()
    loader=read_all_images("data")
    for i in range(epochs):
        print(i)
        for img,target_bbox,label in loader:
            if torch.cuda.is_available():
                img=img.cuda()
                target_bbox=target_bbox.cuda()
                label=label.cuda()
            detections=model(img,False)
            detections1=detections.clone()
            loss_val=0
            for bn in range(1):
                mask = []
                detections1=detections1[detections1[:,0]==np.float(bn)]
                mask2=torch.sum(target_bbox[bn],1)>0
                for k in range(len(mask2)):
                    if mask2[k]==1:
                        ious = bbox_iou(target_bbox[bn][k].view(1, -1), detections1[:, 1:5])
                        mask.append(torch.argmax(ious))

                for j in range(4):
                    loss_val+=loss_coordinates(detections1[mask,1+j],target_bbox[bn,mask2,j],)
                for j in range(num_classes+1):
                    loss_val+=loss_confidence(detections1[mask,5+j],torch.ones(len(mask)).cuda())
            optim.zero_grad()
            loss_val.backward()
            total_loss+=loss_val.item()
            optim.step()
        print(total_loss)
        torch.save(model.state_dict(), "saved_weights.pkl")


train()