import torch
from torch import nn


class Autoencoder(nn.Module):
    """
    Supported:
      one_hidden: 784 -> bottleneck -> 784
      two_hidden: 784 -> 400 -> bottleneck -> 784
      three_hidden: 784 -> 400 -> bottleneck -> 400 -> 784

    The bottleneck is linear. Other hidden layers use sigmoid.
    """

    def __init__(self, bottleneck_dim, architecture="one_hidden"):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.architecture = architecture

        if architecture == "one_hidden":
            self.encoder = nn.Sequential(
                nn.Linear(784, bottleneck_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck_dim, 784),
                nn.Sigmoid(),
            )

        elif architecture == "two_hidden":
            self.encoder = nn.Sequential(
                nn.Linear(784, 400),
                nn.Sigmoid(),
                nn.Linear(400, bottleneck_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck_dim, 784),
                nn.Sigmoid(),
            )

        elif architecture == "three_hidden":
            self.encoder = nn.Sequential(
                nn.Linear(784, 400),
                nn.Sigmoid(),
                nn.Linear(400, bottleneck_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck_dim, 400),
                nn.Sigmoid(),
                nn.Linear(400, 784),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_layers):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
            ])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
