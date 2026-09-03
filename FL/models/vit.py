from torch import nn
import torch
from torchvision.models import vit_b_32

class ViT_B32(nn.Module):
    """
    Class for a Vision Transformer (ViT-B/32) Model for CIFAR-100
    """

    def __init__(self, num_classes=100):
        super().__init__()
        # Load the ViT-B/32 backbone from torchvision
        self.model = vit_b_32(weights="IMAGENET1K_V1")

        # Replace the classifier head for CIFAR-100
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.model(x)
