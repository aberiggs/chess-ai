import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch import nn
from torch.utils.data import DataLoader

from model import ChessModel

def train():
    ds = torch.load("../data/lichess_2017_dataset.pth", weights_only=True)
    print("Preparing to train on", len(ds), "samples")
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Performing training using {device}")

    model = ChessModel().to(device)

    learning_rate = 0.001
    epochs = 1;
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print("----- STARTING EPOCH", epoch + 1, "-----")
        model.train()
        batch_count = 0
        for batch in dl:
            running_loss = 0

            model_input = batch["board"].to(device)
            best_moves = batch["next_move"].to(device)

            # print("Model input shape:", model_input.shape)
            # print("Best moves shape:", best_moves.shape)
            optimizer.zero_grad() 
            logits = model(model_input)
            loss = loss_fn(logits, best_moves)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            if batch_count % 100 == 0:
                print(f"[Batch #{batch_count + 1}] Loss: {running_loss}")
            batch_count += 1

    torch.save(model.state_dict(), "model.pth")
        
            
            

if __name__ == "__main__":
    train()