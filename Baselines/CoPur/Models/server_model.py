import torch.nn as nn
import torch.nn.functional as F

def my_leaky_relu(x):
    return F.leaky_relu(x, negative_slope=0.9)

class CoPurAggregator(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256):

        super(CoPurAggregator, self).__init__()

        self.d1 = nn.Linear(in_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.activation1 = my_leaky_relu

    def forward(self, x):
        x = self.activation1(self.d1(x))
        x = self.activation1(self.d2(x))
        return self.out(x)
