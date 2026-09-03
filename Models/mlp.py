import torch.nn as nn
import torch.nn.functional as F

class SmallNN(nn.Module):

    def __init__(self, total_clients, num_classes, d=4):
        super().__init__()
        self.fc1 = nn.Linear(total_clients*num_classes, total_clients*d)
        self.fc2 = nn.Linear(total_clients*d, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
