import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: define conv1, conv2 (nn.Conv2d), fc1, fc2, fc3 (nn.Linear)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # TODO: conv1 → relu → maxpool → conv2 → relu → maxpool
        #       → flatten → fc1 → relu → fc2 → relu → fc3
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x