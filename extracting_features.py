from torchvision import models
import torch
import torch.nn as nn
import pickle

class Feature_extractor(nn.Module):
    def __init__(self):
        super(Feature_extractor,self).__init__()
        weight = models.EfficientNet_B0_Weights.DEFAULT
        efficientnet = models.efficientnet_b0(weights = weight)
        self.features = nn.Sequential(*list(efficientnet.children())[:-1],
                                      nn.Flatten(),
                                        )
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.features(x) 


# fet_ext = Feature_extractor()
