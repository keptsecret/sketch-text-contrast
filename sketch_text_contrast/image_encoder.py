import torch as th
import torch.nn as nn

import torchvision.models as models

vgg16 = models.vgg16_bn(pretrained=True)
conv_layers = vgg16.features

for params in conv_layers.parameters():
    params.requires_grad = False

linear_layers = nn.Sequential(
    nn.Linear(in_features=25088, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(in_features=25088, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(in_features=4096, out_features=768, bias=True)
)
