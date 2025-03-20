import torch
from torch import nn

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Adaptive Pooling to ensure fixed-size input to FC layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))  

        # Fully Connected Layers
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(128, 256),  # 128 comes from the last conv layer output channels
            nn.ReLU(),
            nn.Linear(256, 200),  # 200 is the number of classes in TinyImageNet
        )

    def forward(self, x):
        # Forward pass through Conv layers
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()

        # Adaptive Pooling to ensure fixed feature size
        x = self.global_avg_pool(x)  

        # Flatten before feeding into FC layers
        x = torch.flatten(x, 1)
        x = self.linear_relu_stack(x)

        return x
