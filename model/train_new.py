import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch import nn
from torch.utils.data import DataLoader

from model import ChessModel

def train():

    ds = torch.load("../data/lichess_2017_02_dataset.pth", weights_only=False)
    print("Preparing to train on", len(ds), "samples")

    batch_size = 128
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    print("DataLoader ready with", len(dl), "batches")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Performing training using {device}")

    model = ChessModel().to(device)
    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth", weights_only=True))

    learning_rate = 0.0003
    epochs = 5;
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print("----- STARTING EPOCH", epoch + 1, "-----")
        model.train()
        batch_count = 0
        running_loss = 0
        for batch in dl:

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

            if (batch_count+1) % 500 == 0:
                print(f"[Batch #{batch_count + 1}] Loss: {running_loss/500}")
                running_loss = 0
            batch_count += 1
        
        torch.save(model.state_dict(), "model.pth")
        
            
            

if __name__ == "__main__":
    train()