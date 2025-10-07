import torch
import torchvision.models as models
import torch.nn as nn

class SpectralFeatureExtractor(nn.Module):
    def __init__(self):
        super(SpectralFeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        resnet.fc = nn.Linear(resnet.fc.in_features, 512)
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)  # (B, 512)