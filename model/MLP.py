import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_features=8, num_hidden_layers=2, output_features=4):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()  # Use ModuleList to ensure all layers are registered

        # First layer
        self.layers.append(nn.Linear(input_features, 32))
        self.layers.append(nn.LeakyReLU(0.01))
        
        # Second layer
        self.layers.append(nn.Linear(32, 64))
        self.layers.append(nn.LeakyReLU(0.01))

        # Third layer
        self.layers.append(nn.Linear(64, 128))
        self.layers.append(nn.LeakyReLU(0.01))

        # Additional hidden layers as specified by num_hidden_layers
        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(128, 128))
            self.layers.append(nn.LeakyReLU(0.01))

        # Output layer
        self.layers.append(nn.Linear(128, output_features))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
