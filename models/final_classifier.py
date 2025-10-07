import torch
import torch.nn as nn

class FinalClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=5):
        super(FinalClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)