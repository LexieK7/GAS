import os

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision.models.resnet import Bottleneck, ResNet
from sklearn import metrics
import timm
from sklearn.metrics import roc_curve, roc_auc_score,accuracy_score
import torch.nn.functional as F



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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



model = torch.load("PRETRAINED MODEL PATH")
model.eval()
model.to(device)

dataset = "# DATASET PATH"

testset = datasets.ImageFolder(root=dataset, transform=image_transforms)
valid_data = DataLoader(testset, batch_size=1, shuffle=False, num_workers=8)

print("Data Length:", len(testset))

pred_list = []
label_list = []
output_list = []

with torch.no_grad():
    model.eval()

    for j, (inputs, labels) in enumerate(tqdm(valid_data)):

        inputs = inputs.to(device)

        outputs = model(inputs)
        prob = F.softmax(outputs, dim = 1)

        ret, predictions = torch.max(outputs.data, 1)
        result = prob[0][1].cpu().numpy().tolist()



        save_predictions = predictions.cpu().numpy().tolist()
        pred_list.extend(save_predictions)
        label_list.extend(labels.cpu().numpy().tolist())
        output_list.append(result)


print(pred_list)
print(label_list)


num_zeros = pred_list.count(0)
num_ones = pred_list.count(1)

print(f"Num of poor sample (Pred = 0): {num_zeros}")
print(f"Num of good sample (Pred = 0): {num_ones}")

