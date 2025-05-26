import torch
import torch.nn as nn
import torch.nn.functional as F

class JDLightCurve1DCNN(nn.Module):
    def __init__(self, num_classes: int = 10, input_length: int = 400, in_channels=1):
        super(JDLightCurve1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Compute the flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, input_length)
            out = self.pool1(F.relu(self.conv1(dummy_input)))
            out = self.pool2(F.relu(self.conv2(out)))
            out = self.pool3(F.relu(self.conv3(out)))
            self.flattened_size = out.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class JDLightCurve1DCNN6L(nn.Module):
    def __init__(self, num_classes: int = 10, input_length: int = 400, in_channels=1):
        super(JDLightCurve1DCNN6L, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool1d(kernel_size=2)

        self.conv6 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.pool6 = nn.MaxPool1d(kernel_size=2)

        # Dynamically calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, input_length)
            x = self.pool1(F.relu(self.conv1(dummy_input)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.pool3(F.relu(self.conv3(x)))
            x = self.pool4(F.relu(self.conv4(x)))
            x = self.pool5(F.relu(self.conv5(x)))
            x = self.pool6(F.relu(self.conv6(x)))
            self.flattened_size = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.pool6(F.relu(self.conv6(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class JDLightCurve1DCNN6LBatchNormDropout(nn.Module):
    def __init__(self, num_classes: int = 10, input_length: int = 400, in_channels=1):
        super(JDLightCurve1DCNN6LBatchNormDropout, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(2)

        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        self.pool5 = nn.MaxPool1d(2)

        self.conv6 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(512)
        self.pool6 = nn.MaxPool1d(2)

        self.dropout_conv = nn.Dropout(0.3)
        self.dropout_fc = nn.Dropout(0.5)

        # Dynamically calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, input_length)
            x = self.pool1(self.dropout_conv(F.relu(self.bn1(self.conv1(dummy_input)))))
            x = self.pool2(self.dropout_conv(F.relu(self.bn2(self.conv2(x)))))
            x = self.pool3(self.dropout_conv(F.relu(self.bn3(self.conv3(x)))))
            x = self.pool4(self.dropout_conv(F.relu(self.bn4(self.conv4(x)))))
            x = self.pool5(self.dropout_conv(F.relu(self.bn5(self.conv5(x)))))
            x = self.pool6(self.dropout_conv(F.relu(self.bn6(self.conv6(x)))))
            self.flattened_size = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.dropout_conv(F.relu(self.bn1(self.conv1(x)))))
        x = self.pool2(self.dropout_conv(F.relu(self.bn2(self.conv2(x)))))
        x = self.pool3(self.dropout_conv(F.relu(self.bn3(self.conv3(x)))))
        x = self.pool4(self.dropout_conv(F.relu(self.bn4(self.conv4(x)))))
        x = self.pool5(self.dropout_conv(F.relu(self.bn5(self.conv5(x)))))
        x = self.pool6(self.dropout_conv(F.relu(self.bn6(self.conv6(x)))))

        x = x.view(x.size(0), -1)
        x = self.dropout_fc(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
class JDLightCurve1DCNN9LBatchNormDropout(nn.Module):
    def __init__(self, num_classes: int = 10, input_length: int = 400):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)  # pool after conv3

        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        self.conv6 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(512)
        self.pool2 = nn.MaxPool1d(2)  # pool after conv6

        self.conv7 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm1d(512)
        self.conv8 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm1d(512)
        self.conv9 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm1d(512)
        self.pool3 = nn.MaxPool1d(2)  # pool after conv9

        self.dropout_conv = nn.Dropout(0.3)
        self.dropout_fc = nn.Dropout(0.5)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Conv block 1-3 + pool1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout_conv(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout_conv(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout_conv(x)
        x = self.pool1(x)

        # Conv block 4-6 + pool2
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout_conv(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout_conv(x)
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.dropout_conv(x)
        x = self.pool2(x)

        # Conv block 7-9 + pool3
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.dropout_conv(x)
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.dropout_conv(x)
        x = F.relu(self.bn9(self.conv9(x)))
        x = self.dropout_conv(x)
        x = self.pool3(x)

        x = self.global_pool(x)  # (B, 512, 1)
        x = x.squeeze(-1)        # (B, 512)

        x = self.dropout_fc(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x