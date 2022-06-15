import torch as th
import torch.nn as nn

import torchvision.models as models
from .vqvae import Encoder, VectorQuantizer

class SketchEncoder(nn.Module):

    def __init__(self, resnet=False, all_trainable=False):

        super(SketchEncoder,self).__init__()
        self.mode = resnet

        if self.mode:
            self.conv = models.resnet50(pretrained=True)
            # self.avgpool = self.conv.avgpool

            more_conv = [nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding='same'),
                            nn.BatchNorm2d(1024),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding='same'),
                            nn.BatchNorm2d(512),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same'),
                            nn.BatchNorm2d(512),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2, stride=2)]

            # linear = [nn.Linear(1,32),
            #             nn.ReLU(inplace=True),
            #             nn.Linear(32,64),
            #             nn.ReLU(inplace=True),
            #             nn.Linear(64,128)]

            linear = [nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128)]

        else:
            # take layers up to block4_conv1, instead
            vgg = models.vgg19(pretrained=True)
            self.conv = vgg.features[:21]
            # self.avgpool = vgg.avgpool

            more_conv = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same'),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same'),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same'),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2, stride=2)]

            # linear = [nn.Linear(49,64),
            #            nn.ReLU(inplace=True),
            #            nn.Linear(64,64),
            #            nn.ReLU(inplace=True),
            #            nn.Linear(64,128)]

            linear = [nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128)]

        for params in self.conv.parameters():
            params.requires_grad = all_trainable

        self.refine_convs = nn.Sequential(*more_conv)
        self.sketch_mapper = nn.Sequential(*linear)

    def forward(self,inputs):

        batch_size = inputs.shape[0]

        if self.mode:
            x = self.conv.conv1(inputs)
            x = self.conv.bn1(x)
            x = self.conv.relu(x)
            x = self.conv.maxpool(x)
            x = self.conv.layer1(x)
            x = self.conv.layer2(x)
            x = self.conv.layer3(x)
            conv_feats = self.refine_convs(x)
        else:
            x = self.conv(inputs)
            conv_feats = self.refine_convs(x)

        #avg_pool_feats = self.avgpool(conv_feats).view(batch_size, 512, -1)
        avg_pool_feats = conv_feats.view(batch_size, 512, -1)
        std, mean = th.std_mean(avg_pool_feats, dim=(1,2), unbiased=False, keepdim=True)
        avg_pool_feats = (avg_pool_feats - mean) / std
        outputs = self.sketch_mapper(avg_pool_feats)

        return outputs

class VQSketchEncoder(nn.Module):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()

        self.encoder = Encoder(**ddconfig)
        self.avg_pool = nn.AdaptiveAvgPool2d((16, 16))
        self.sketch_mapper = nn.Sequential(
                                            nn.Linear(256, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128)
                                            )
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = th.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = th.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.up_post_conv = th.nn.Conv2d(ddconfig["z_channels"], 512, 1)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path):
        sd = th.load(path)
        self.load_state_dict(sd, strict=False)

    def forward(self, x):
        batch_size = x.shape[0]
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)

        quant = self.post_quant_conv(quant)
        quant = self.up_post_conv(quant)
        quant = self.avg_pool(quant).view(batch_size, 512, -1)
        dec = self.sketch_mapper(quant)
        return dec, emb_loss, info

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
