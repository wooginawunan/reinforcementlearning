import torch.nn as nn
import torch.nn.functional as F
import torch
from resnet import resnet18

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 8)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 4)
        self.conv3 = nn.Conv2d(64, 32, 3)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        n, c, _, _ = x.size()
        x = x.view(n, c, -1).mean(-1)

        return x

class Model(nn.Module):
    def __init__(self, history_length, classification=False, resnet=False, moddrop=False, shared=False):
        super(Model, self).__init__()
        
        if shared:
            if resnet:
                if moddrop:
                    self.network = resnet18(num_classes=32)
                else:
                    self.network = resnet18(num_classes=3 if not classification else 5)
            else:
                self.network = Network()

        else:
            if resnet:
                if moddrop:
                    self.networks = nn.ModuleList([resnet18(num_classes=32) for i in range(history_length)])
                else:
                    self.networks = nn.ModuleList([resnet18(num_classes=3 if not classification else 5) \
                        for i in range(history_length)])
            else:
                self.networks = nn.ModuleList([Network() for i in range(history_length)])

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(32, 32)
        self.fc1 = nn.Linear(32, 3 if not classification else 5)
        self.classification = classification
        self.resnet = resnet
        self.moddrop = moddrop
        self.shared = shared
        self.history_length = history_length

    def forward(self, x):

        if self.shared:
            x = torch.stack([self.network(x[:, i, :, :].unsqueeze_(1)) for i in range(self.history_length)], -1)
        else:
            x = torch.stack([net(x[:, i, :, :].unsqueeze_(1)) for i, net in enumerate(self.networks)], -1)
        if self.moddrop:
            x = self.dropout(x)
            x = x.mean(-1)
            x = self.fc1(x)
        else:
            x = x.mean(-1)
        if self.resnet:
            if self.classification:
                return x
            else:
                return F.softmax(x)

        x = self.dropout(F.relu(self.fc(x)))

        if self.classification:
            return self.fc1(x)

        x = F.softmax(self.fc1(x))

        return x
