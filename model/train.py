import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch import nn
from torch.utils.data import DataLoader

from model import ChessModel

def train():
    ds = torch.load("../data/lichess_2018-03_dataset.pth", weights_only=False)
    print("Preparing to train on", len(ds), "samples")

    batch_size = 256
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    print("DataLoader ready with", len(dl), "batches")
    
    vds = torch.load("../data/lichess_2017-03_dataset.pth", weights_only=False)
    vdl = DataLoader(vds, batch_size=batch_size, shuffle=True)

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

    learning_rate = 0.001
    epochs = 100;
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    best_loss = float("inf")
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        print("----- STARTING EPOCH", epoch + 1, "-----")
        
        print("Training model...")
        model.train()
        batch_count = 0
        training_loss = 0
        for batch in dl:
            model_input = batch["board"].to(device)
            best_moves = batch["next_move"].to(device)

            optimizer.zero_grad() 
            logits = model(model_input)
            loss = loss_fn(logits, best_moves)
            training_loss += loss.item()
            loss.backward()
            optimizer.step()

            if (batch_count+1) % 500 == 0:
                print(f"[Batch #{batch_count + 1}] Running loss: {training_loss}")
            batch_count += 1
        
        training_loss /= len(dl)
        
        print("\nValidating model...")
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for batch in vdl:
                model_input = batch["board"].to(device)
                best_moves = batch["next_move"].to(device)
                logits = model(model_input)
                loss = loss_fn(logits, best_moves)
                validation_loss += loss.item()
                
        validation_loss /= len(vdl)
        print(f"Epoch {epoch + 1} complete. Training loss: {training_loss} - Validation loss: {validation_loss}")
        print("----- END OF EPOCH", epoch + 1, "-----\n")

        if validation_loss < best_loss:
            print("New best model found. Saving...")
            best_loss = validation_loss
            torch.save(model.state_dict(), "model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping. Patience limit reached.")
                break

if __name__ == "__main__":
    train()