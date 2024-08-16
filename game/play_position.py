import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chess
import torch

from model.predict import predict
from model.model import ChessModel_V4 as ChessModel
from model.model import ChessModel_V1

model_path = "../model/v4_model.pth"
model_fast_path = "../model/8-4-2024.pth"

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Loading models...")
model = ChessModel().to(device)
model.load_state_dict(torch.load(model_path, weights_only=True))
model_fast = ChessModel_V1().to(device)
model_fast.load_state_dict(torch.load(model_fast_path, weights_only=True))

fen = input("FEN: ")
board = chess.Board(fen)
should_continue = True
print(board)
while (should_continue):
    print()
    print("Inferencing...")

    predicted_move = predict(board, model, model_fast, device)

    print(f"Predicted move: {predicted_move}")
    board.push(predicted_move)
    print(board)
    print()
    
    print("Continue? (y/n)")
    should_continue = input() == "y"
    

print("Done.")