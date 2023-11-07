import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out += residual
        out = self.relu(out)
        return out


class Net(nn.Module):
    def __init__(self, label_num, window_size):
        super().__init__()

        self.conv1 = nn.Conv1d(window_size, 64, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.residual_block1 = ResidualBlock(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.residual_block2 = ResidualBlock(128)
        self.fc1 = nn.Linear(128 * 64, 1024)
        self.batchnorm3 = nn.BatchNorm1d(1024)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, label_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # x = self.residual_block1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # x = self.residual_block2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, hidden_size, window_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.bn = nn.BatchNorm1d(window_size)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        u = self.tanh(self.linear(x))
        a = self.softmax(u)
        s = (a * x).sum(axis=1)
        # s = a * x
        return s


class ABLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, window_size, num_classes, num_layers=1):
        super(ABLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size * 2, window_size)  # 2 for bidirection
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.cnn = Net(num_classes, window_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.attention(out)
        out = self.fc1(out)
        out = self.bn(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.fc2(out)

        # out = self.cnn(out)
        return out


# CSI net
class CSINet(nn.Module):
    def __init__(self, label_num, window_size, amp=False):
        super(CSINet, self).__init__()
        final_length = ((window_size - 4) // 2 - 3) // 2 + 1
        if amp:
            self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3))
        else:
            self.conv1 = nn.Conv2d(2, 16, kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(32)

        # The size of the flattened feature maps can be computed based on input size and conv layers.
        # Here it's given as an example, but it's essential to adjust based on actual size after convolutions.
        self.fc1 = nn.Linear(32 * final_length * 62, 32)

        self.fc2 = nn.Linear(32, label_num)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, (2, 2))
        # x = F.dropout(x, p=0.25, training=self.training)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, (2, 2))
        # x = F.dropout(x, p=0.25, training=self.training)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.25, training=self.training)

        x = self.fc2(x)
        return x
