import torch.nn as nn
import torch.nn.functional as F

def my_leaky_relu(x):
    return F.leaky_relu(x, negative_slope=0.8)

class CoPurAE(nn.Module):
    def __init__(self, in_dim=170,out_dim=10, hidden_dim=200, encode_dim=120):
        
        super(CoPurAE, self).__init__()
        
        self.d1 = nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.d2 = nn.Linear(hidden_dim, encode_dim)
        self.d3 = nn.Linear(encode_dim, hidden_dim)
        self.d4 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        x = my_leaky_relu(self.d1(x))
        x = self.d2(x)  # No activation
        x2 = x.clone()
        x = my_leaky_relu(self.d3(x2))
        x = self.d4(x)  # No activation
        return x