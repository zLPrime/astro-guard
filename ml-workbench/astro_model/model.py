import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the simplest fully connected neural network
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # First conv layer: 1 input channel -> 16 output channels.
        # Using kernel_size=3 with padding=1 to keep spatial dimensions.
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # Second conv layer: 16 input channels -> 32 output channels.
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Use MaxPool2d to reduce spatial dimensions by a factor of 2.
        self.pool = nn.MaxPool2d(2, 2)
        
        # After two conv+pool layers:
        # - Height: 480 -> 240 (after first pool) -> 120 (after second pool)
        # - Width: 640 -> 320 -> 160
        # So the feature map size is [batch, 32, 120, 160].
        # Compute flattened size:
        self.flattened_size = 32 * 120 * 160
        
        # Fully connected layers:
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # x: [batch_size, 1, 480, 640]
        x = self.pool(F.relu(self.conv1(x)))  # -> [batch_size, 16, 240, 320]
        x = self.pool(F.relu(self.conv2(x)))  # -> [batch_size, 32, 120, 160]
        x = x.view(x.size(0), -1)             # Flatten: [batch_size, 32*120*160]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x