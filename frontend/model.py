import torch
import torch.nn as nn
from torchvision import models

class PneumoniaModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = models.resnet18(
            weights="DEFAULT"
        )

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            2
        )

    def forward(self, x):

        return self.model(x)