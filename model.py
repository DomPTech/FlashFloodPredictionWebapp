import torch
import torch.nn as nn

class FlashFloodClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FlashFloodClassifier, self).__init__()
        self.layer_1 = nn.Linear(input_dim, 64)
        self.layer_2 = nn.Linear(64, 32)
        self.layer_3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        return self.sigmoid(self.layer_3(x))