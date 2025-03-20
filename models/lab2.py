from torch import nn

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6422528,256),
            nn.ReLU(),
            nn.Linear(256,200),
            nn.ReLU(),# 200 is the number of classes in TinyImageNet
        )

    def forward(self, x):
        # Define forward pass

        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = torch.flatten(x, 1)
        x = self.linear_relu_stack(x)

        return x
model = CustomNet().to(device)
