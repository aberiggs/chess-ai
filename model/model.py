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
class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()
        self.conv_nn_stack = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(15, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Fully connected layers
            nn.Flatten(),
            nn.Linear(64*8*8, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            # Output layer
            nn.Linear(4096, 8*8*8*8)
        )

    def forward(self, x):
        logits = self.conv_nn_stack(x)
        return logits
