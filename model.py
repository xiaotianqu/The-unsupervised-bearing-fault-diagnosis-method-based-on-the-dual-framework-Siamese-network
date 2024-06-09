import torch.nn as nn
class DeepSiameseNetwork(nn.Module):
    def __init__(self, in_channels,):
        super(DeepSiameseNetwork, self).__init__()


        self.shared_layer = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(96, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.flattened_size = 6272

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 1),
            nn.ReLU()
        )

    def forward(self, x1, x2):
        out1 = self.shared_layer(x1)
        out1 = out1.view(out1.size(0), -1)
        out2 = self.shared_layer(x2)
        out2 = out2.view(out2.size(0), -1)
        out1 = self.fc_layers(out1)
        out2 = self.fc_layers(out2)
        return out1, out2
