import torch as th
import torch.nn as nn

import torchvision.models as models

class SketchEncoder(nn.Module):

    def __init__(self, resnet=False, trainable=False):

        super(SketchEncoder,self).__init__()
        self.mode = resnet

	if self.mode:
            self.conv = models.resnet34(pretrained=True)
        else:
            self.conv = models.vgg19(pretrained=True)

        if trainable:
            for params in self.conv.parameters():
                params.requires_grad = False

        model = [nn.Linear(49,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU(), nn.Linear(64,128)]
        self.sketch_mapper = nn.Sequential(*model)

    def forward(self,inputs):

        batch_size = inputs.shape[0]

        conv_feats = self.conv.features(inputs)
        avg_pool_feats = self.conv.avgpool(conv_feats).view(batch_size, 512, -1)

        return self.sketch_mapper(avg_pool_feats)

class ImageEncoder(nn.Module):
    def __init__(self, xf_width, text_ctx):
        super().__init__()

        vgg16 = models.vgg16_bn(pretrained=True)
        # self.conv_layers = vgg16.features
        # self.avg_pool = vgg16.avgpool

        for params in vgg16.features.parameters():
            params.requires_grad = False

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=25088, out_features=8192, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=8192, out_features=16384, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=16384, out_features=32768, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=32768, out_features=xf_width*text_ctx, bias=True)
        )
        # self.linear_layers = vgg16.classifier

        vgg16.classifier = self.linear_layers
        self.vgg16 = vgg16
        self.xf_width = xf_width
        self.text_ctx = text_ctx

    def forward(self, x):
        # x = self.conv_layers(x)
        # x = self.avg_pool(x)
        # print(x.shape)
        # output = self.linear_layers(x)
        x = self.vgg16(x)
        output = x.reshape(1, self.xf_width, self.text_ctx)

        return output
