import torch as th
import torch.nn as nn

import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        vgg16 = models.vgg16_bn(pretrained=True)
        self.conv_layers = vgg16.features

        for params in self.conv_layers.parameters():
            params.requires_grad = False

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=2048, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=512, bias=True)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        output = self.linear_layers(x)

        return output
