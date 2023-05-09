import torch.nn as nn
import torchvision.models as models


class customModel(nn.Module):
    def __init__(self, num_classes):
        super(customModel, self).__init__()
        self.backbone = models.efficientnet_b4(pretrained=True)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x