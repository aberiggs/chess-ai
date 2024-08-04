import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import chess
from torch.utils.data import Dataset

# import the board_to_tensor and move_to_output_tensor functions from model/conversions.py
from conversions import board_to_tensor, move_to_output_tensor

class ChessDataset(Dataset):
    def __init__(self, data):
        self.fens = []
        self.next_moves = []
        for sample in data:
            self.fens.append(sample['board'].fen())
            self.next_moves.append(sample['next_move'])

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        board = chess.Board(self.fens[idx])
        color = board.turn
        board_tensor = board_to_tensor(board, color)
        next_move = move_to_output_tensor(self.next_moves[idx])

        sample = {'board': board_tensor, 'next_move': next_move}
        return sample
