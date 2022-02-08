import torch as th
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

vgg16 = models.vgg16_bn(pretrained=True)
print(vgg16)