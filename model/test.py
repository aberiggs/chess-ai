import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch import nn
from torch.utils.data import DataLoader

from model import ChessModel

def test():

    ds = torch.load("../data/lichess_2017_02_dataset.pth", weights_only=False)
    print("Preparing to train on", len(ds), "samples")

    batch_size = 64
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    print("DataLoader ready with", len(dl), "batches")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Performing testing using {device}")

    model = ChessModel().to(device)
    model.load_state_dict(torch.load("model.pth", weights_only=True))

    model.eval()

    batch_count = 0
    total_count = 0
    correct_count = 0
    for batch in dl:
        model_input = batch["board"].to(device)
        best_moves = batch["next_move"].to(device)
        logits = model(model_input)
        
        total_count += batch_size
        for i in range(batch_size):
            if torch.argmax(logits[i]) == torch.argmax(best_moves[i]):
                correct_count += 1

        if (batch_count+1) % 250 == 0:
            print(f"[Batch #{batch_count + 1}] Accuracy: {(float(correct_count)/total_count)*100}%")
            total_count = 0
            correct_count = 0
        batch_count += 1
        
        
            
            

if __name__ == "__main__":
    test()