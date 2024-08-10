import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch import nn
from torch.utils.data import DataLoader

from model import ChessModel

def test_accuracy():
    test_size = 1024
    vds = torch.load("../data/lichess_2017-03_dataset.pth", weights_only=False)
    vdl = DataLoader(vds, batch_size=test_size, shuffle=True)
    
    # get the first batch
    batch = next(iter(vdl))
    model = ChessModel()
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    model.eval()
    model_input = batch["board"]
    best_moves = batch["next_move"]
    logits = model(model_input)
    
    
    correct = 0
    input_length = len(batch["board"])
    print(f"Testing on {input_length} samples")
    for i in range(input_length):
        sorted_probs = torch.argsort(logits[i])
        top_preds = sorted_probs[-4:]
        actual = best_moves[i].item()
        print(f"Actual: {actual}, Top predictions: {top_preds}")
        if actual in top_preds:
            correct += 1
    
    print(f"Accuracy: {correct/input_length*100:.2f}%")
    
    
if __name__ == "__main__":
    test_accuracy()