import torch
import torch.nn as nn
import torchvision.models as models

class ImageForgeryDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ImageForgeryDetector, self).__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
