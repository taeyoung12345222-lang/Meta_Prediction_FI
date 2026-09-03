import torch
import torch.nn as nn
import torch.nn.init as init

class WeightedAggregatorNN(nn.Module):
    def __init__(self, total_clients, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # Create a learnable weight matrix for per-class weights for each client
        self.weights = nn.Parameter(torch.randn(total_clients, num_classes, dtype=torch.float32))
        init.xavier_uniform_(self.weights)

    def forward(self, x):
        # Assume x is a batch of logit vectors of shape (batch_size, total_clients*num_classes)
        x = x.view(x.size(0), -1, self.num_classes)

        # Apply weights to each client's logits
        weighted_logits = x * self.weights
        
        # Sum up the weighted logits across clients to get aggregated logits for each class
        out = torch.sum(weighted_logits, dim=1)

        return out