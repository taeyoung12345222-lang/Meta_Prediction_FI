import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet8 import ResNet8

# for CIFAR10, SVHN
class GatingFunction(nn.Module):
    def __init__(self, num_experts):
        super(GatingFunction, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_experts)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# for CIFAR10, SVHN
class FederatedMoE(nn.Module):
    def __init__(self, num_experts, use_best_expert=False):
        super(FederatedMoE, self).__init__()
        self.num_experts = num_experts
        self.gating_function = GatingFunction(self.num_experts)
        self.use_best_expert = use_best_expert

    def forward(self, x):
        # Calculate the gating coefficients
        gating_coefficients_unnormalised = self.gating_function(x[0])

        # Apply softmax to gating coefficients to ensure they sum to 1
        gating_coefficients_normalised = F.softmax(gating_coefficients_unnormalised, dim=1)

        # Calculate the weighted sum of expert predictions
        if self.use_best_expert:
            best_expert_index = torch.argmax(gating_coefficients_normalised, dim=1)
            expert_predictions = torch.split(x[1], x[1].shape[1] // self.num_experts, dim=1)
            weighted_sum = torch.stack([expert_predictions[j][i] for i, j in enumerate(best_expert_index)])
        else:
            # x[1] has shape (batch_size, num_experts * num_classes)
            expert_predictions = torch.split(x[1], x[1].shape[1] // self.num_experts, dim=1)
            weighted_sum = sum(coeff.unsqueeze(1) * pred for coeff, pred in zip(gating_coefficients_normalised.unbind(1), expert_predictions))

        return weighted_sum

# for CIFAR100
class GatingFunction2(nn.Module):
    def __init__(self, n): # here n is num_experts * num_classes
        super(GatingFunction2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# for CIFAR100    
class FederatedMoE2(nn.Module):
    def __init__(self, num_experts, num_classes):
        super(FederatedMoE2, self).__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.gating_function = GatingFunction2(self.num_experts * self.num_classes)

    def forward(self, x):
        # Calculate the gating coefficients
        gating_coefficients = self.gating_function(x[0])

        # x[1] has shape (batch_size, num_experts * num_classes)
        # gating_coefficients has shape (batch_size, num_experts * num_classes)
        weighted_sum = gating_coefficients * x[1]

        # Reshape the weighted sum to have shape (batch_size, num_experts, num_classes)
        weighted_sum = weighted_sum.view(-1, self.num_experts, self.num_classes)

        # sum over the experts dimension to get the final prediction
        weighted_sum = torch.sum(weighted_sum, dim=1)

        return weighted_sum
