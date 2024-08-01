import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import Dataset

# import the board_to_tensor and move_to_output_tensor functions from model/conversions.py
from conversions import board_to_tensor, move_to_output_tensor

class ChessDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        move_color = self.data[idx]['board'].turn 
        board = board_to_tensor(self.data[idx]['board'], move_color)
        next_move = move_to_output_tensor(self.data[idx]['next_move'])

        sample = {'board': board, 'next_move': next_move}
        return sample
