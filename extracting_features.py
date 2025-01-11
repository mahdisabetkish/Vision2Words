import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models



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

        self.projection = nn.Linear(1280, 128)
        self.batchnorm = nn.BatchNorm1d(128)  # Add BatchNorm for normalization

    
    def forward(self, x):
        efficient = self.features(x)
        project = self.projection(efficient)
        normalized = self.batchnorm(project)
        return normalized
