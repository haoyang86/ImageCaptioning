#GT ID: ywei364
#this is the encoder file for M2 model
#encoder design pipelines:
#1. for the encoder backbone, the ResNet - 50 was choosen
#2. which layer output to use? the last convolutional feature map before global average pooling
#3. whats the recommanded input image size? 224 * 224 will prefer
#4. whether the pretrained CNN weights should update during training? start with backbone freeze
#5. do we keep raw 2048 channels or project them to a smaller encoder_dim? to project to 512 dim to make decoder / attension work easier
#6. what exactly encoder should return? spatial features + global summary
#7. if add extra processing after projection? yes, add dropout

# input: image shape (B, 3, 224, 224) 
# output 1: encoder_out (B, 49, 512)
# output 2: global_feat (B, 512)

import torch
import torch.nn as nn
from torchvision import models


class EncoderSpatial(nn.Module):
    """
    the encoder moudle for the final project M2 model
    """
    def __init__(self, encoder_dim = 512, freeze_backbone = True, pretrained = True, dropout = 0.2):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.freeze_backbone = freeze_backbone
        self.pretrained = pretrained

        #load full ResNet-50
        if pretrained:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            resnet = models.resnet50(weights=None)

        #only keep layers to Layer4 in resnet 50
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        #freeze backbone
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        #project channel dimension: 2048 - > encoder_dim
        self.projection = nn.Linear(2048, encoder_dim)
        # dropout after projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, images):
          feature = self.backbone(images)
          flatten_feature = feature.flatten(2)
          feature = flatten_feature.permute(0, 2, 1)
          projection_feature = self.projection(feature)

          encoder_output = self.dropout(projection_feature)
          global_feat = encoder_output.mean(dim = 1)

          return encoder_output, global_feat



        

