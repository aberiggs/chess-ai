import torch.nn as nn 

"""
8x8x15 convolutional neural network
- 8x8 board
- 6 different types of pieces with 2 colors (12 channels)
- en passant squares (1 channel)
- castling rights with 2 colors (2 channels)

Will output a 8x8x8x8 tensor representing the probability of each piece being moved from a position to another
- From position within the first 8x8 board
- To position within the second 8x8 board
"""

class FastChessModel(nn.Module):
    def __init__(self):
        super(FastChessModel, self).__init__()
        self.conv_nn_stack = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(15, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # Flatten the tensor for the fully connected layers
            nn.Flatten(),
            # Fully connected layers
            nn.Linear(64*8*8, 64*8*8),
            nn.ReLU(),
            # Output layer
            nn.Linear(64*8*8, 8*8*8*8)
        )

    def forward(self, x):
        logits = self.conv_nn_stack(x)
        return logits

class FastChessModel_V1(nn.Module):
    def __init__(self):
        super(FastChessModel_V1, self).__init__()
        self.conv_nn_stack = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(15, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # Flatten the tensor for the fully connected layers
            nn.Flatten(),
            # Fully connected layers
            nn.Linear(128*8*8, 64*8*8),
            nn.ReLU(),
            nn.Linear(64*8*8, 64*8*8),
            nn.ReLU(),
            # Output layer
            nn.Linear(64*8*8, 8*8*8*8)
        )

    def forward(self, x):
        logits = self.conv_nn_stack(x)
        return logits

class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()
        self.conv_nn_stack = nn.Sequential(
            # This architecture is inspired by the VGG-Net architecture
            # Convolutional layers
            nn.Conv2d(15, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            # Flatten the tensor
            nn.Flatten(),
            
            # Fully connected layers
            nn.Linear(256*8*8, 4*8*8*8*8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4*8*8*8*8, 4*8*8*8*8),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Output layer
            nn.Linear(4*8*8*8*8, 8*8*8*8)
        )
            
    def forward(self, x):
        logits = self.conv_nn_stack(x)
        return logits
    

class ChessModel_V3(nn.Module):
    def __init__(self):
        super(ChessModel_V3, self).__init__()
        self.conv_nn_stack = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(15, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # Flatten the tensor for the fully connected layers
            nn.Flatten(),
            # Fully connected layers
            nn.Linear(256*8*8, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Output layer
            nn.Linear(512, 8*8*8*8)
        )

    def forward(self, x):
        logits = self.conv_nn_stack(x)
        # return logits

class ChessModel_V2(nn.Module):
    def __init__(self):
        super(ChessModel_V2, self).__init__()
        self.conv_nn_stack = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(15, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            # Flatten the tensor for the fully connected layers
            nn.Flatten(),
            # Fully connected layers
            nn.Linear(256*8*8, 128*8*8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128*8*8, 32*8*8),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Output layer
            nn.Linear(32*8*8, 8*8*8*8)
        )

    def forward(self, x):
        logits = self.conv_nn_stack(x)
        return logits


class ChessModel_V1(nn.Module):
    def __init__(self):
        super(ChessModel_V1, self).__init__()
        self.conv_nn_stack = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(15, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Flatten the tensor for the fully connected layers
            nn.Flatten(),
            # Fully connected layers
            nn.Linear(64*8*8, 64*8*8),
            nn.ReLU(),
            nn.Linear(64*8*8, 64*8*8),
            nn.ReLU(),
            # Output layer
            nn.Linear(64*8*8, 8*8*8*8)
        )

    def forward(self, x):
        logits = self.conv_nn_stack(x)
        return logits
