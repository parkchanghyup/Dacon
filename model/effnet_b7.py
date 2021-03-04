
from torch import nn

from efficientnet_pytorch import EfficientNet

class MultiLabeleffnet(nn.Module):
    def __init__(self):
        super(MultiLabeleffnet, self).__init__()
        self.conv2d = nn.Conv2d(1, 3, 3, stride=1)
        self.effnet = EfficientNet.from_pretrained('efficientnet-b7')
        self.FC = nn.Linear(1000, 26)

    def forward(self, x):

        x = F.silu(self.conv2d(x))
        x = F.silu(self.effnet(x))
        x = torch.sigmoid(self.FC(x))
        
        return x
