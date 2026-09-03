from torch import nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

class LinearAggregator(nn.Module):
    def __init__(self, total_clients, num_classes):
        self.num_classes = num_classes
        self.num_clients = total_clients
        super(LinearAggregator, self).__init__()
        # One weight per client (not per class)
        self.weights = nn.Parameter(torch.randn(1, self.num_clients, 1))
        # Initialize weights
        init.xavier_uniform_(self.weights)
        # Bias that is added after summing the weighted logits
        self.bias = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # x is expected to be of shape [batch_size, num_clients * num_classes]
        batch_size = x.shape[0]

        # Reshape x to [batch_size, num_clients, num_classes]
        x = x.view(batch_size, -1, self.num_classes)

        # Multiply each client's logits by the corresponding weights
        # weights is [1, num_clients, 1] for broadcasting
        weighted_logits = x * self.weights

        # Sum along the clients dimension to combine the weighted logits
        aggregated_logits = weighted_logits.sum(dim=1)

        # Add the bias (broadcasting the bias across the batch)
        final_logits = aggregated_logits + self.bias

        return final_logits