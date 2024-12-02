import os
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus,AblationCAM, \
                            XGradCAM, EigenCAM, EigenGradCAM,LayerCAM,FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import cv2

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from torchvision.models.resnet import Bottleneck, ResNet
from sklearn import metrics
import timm


class ClipAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(ClipAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        x = self.fc1(x)
        y = self.fc2(x)
        return x, y


class CLIP_Inplanted(nn.Module):
    def __init__(self, model, features=[8, 16, 24]):
        super().__init__()
        self.image_encoder = model
        self.features = features
        self.det_adapters = nn.ModuleList([ClipAdapter(1024, bottleneck=2048) for i in range(len(features))])  # 768

        self.fc = nn.Sequential(
            # nn.Linear(2048, 1024),
            # nn.ReLU(),
            # nn.Dropout(0.25),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Dropout(0.25),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1))

    def forward(self, x):

        B = x.shape[0]
        x = self.image_encoder.patch_embed(x).reshape(B, -1, 1024)
        cls_tokens = self.image_encoder.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.image_encoder.pos_embed[:, :x.size(1), :]
        x = self.image_encoder.pos_drop(x)

        # for i in range(24):

        for i, layer in enumerate(list(self.image_encoder.blocks)):

            x = layer(x)

            if (i + 1) in self.features:
                det_adapt_med, det_adapt_out = self.det_adapters[self.features.index(i + 1)](x)
                x = 0.8 * x + 0.2 * det_adapt_out

                # pooled, tokens = self.image_encoder._global_pool(x)
        x = self.image_encoder.norm(x)
        # x = x[:, 1:].mean(dim=1)# take tokens,ecept cls token

        x = x[:, 0]
        x = self.fc(x)

        return x

image_transforms = transforms.Compose([
        transforms.Resize(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    
model = torch.load("PRETRAINED MODEL PATH")



model.eval()
model.cuda()

#print(model.image_encoder.blocks)
#print(model.image_encoder.blocks[-1].norm1)

use_cuda = torch.cuda.is_available()



def reshape_transform(tensor, height=14, width=14):

    result = tensor[:, 1:, :].reshape(tensor.size(0),
    height, width, tensor.size(2))

    result = result.transpose(2, 3).transpose(1, 2)
    return result
    

cam = GradCAM(model=model,
            target_layers=[model.image_encoder.blocks[-1].norm1],
            use_cuda=use_cuda,
            reshape_transform=reshape_transform)



directory_path = "DATASET PATH"
seve_path = "SAVE PATH"
files = os.listdir(directory_path)




for i, filename in enumerate(files):

    image_path = os.path.join(directory_path, filename)
    print(image_path)
    
    new_file_path = os.path.join(seve_path, filename)
    print(new_file_path)


    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    
    
    input_tensor = preprocess_image(rgb_img,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    
    
    if use_cuda:
        input_tensor = input_tensor.cuda()
    
    # grad-cam
    target_category = None
    grayscale_cam = cam(input_tensor=input_tensor, targets=target_category)
    grayscale_cam = grayscale_cam[0, :]
    grayscale_cam = 1- grayscale_cam
    
    visualization = show_cam_on_image(rgb_img.astype(dtype=np.float32)/255,grayscale_cam)
    
    cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
    
    cv2.imwrite(new_file_path, visualization)
