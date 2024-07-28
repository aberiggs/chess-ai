import chess
import torch

from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader

from conversions import board_to_tensor, move_to_output_tensor
from model import ChessModel

def process_text(text):
    split = text.split("~")
    fen = split[0]
    next_move = split[1].split("_")[0].split(">")[0]
    return fen, next_move

def train():
    ds = load_dataset("TannerGladson/chess-roberta-evaluation", split="filler")
    ds.set_format("torch", columns=["text"])
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    ds_test = load_dataset("TannerGladson/chess-roberta-evaluation", split="test")
    ds_test.set_format("torch", columns=["text"])
    dl_test = DataLoader(ds_test, batch_size=64, shuffle=True)

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
    epochs = 5;
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print("----- STARTING EPOCH", epoch + 1, "-----")
        model.train()
        batch_count = 0
        for batch in dl:
            running_loss = 0
            # create a tensor containing all fens and next moves
            boards_arr = []
            best_moves_arr = []
            
            for text in batch["text"]:
                fen, next_move = process_text(text)
                board = chess.Board(fen)
                boards_arr.append(board_to_tensor(board, board.turn))
                best_moves_arr.append(move_to_output_tensor(board.parse_san(next_move)))

            # create a tensor of the best moves
            model_input = torch.stack(boards_arr).to(device)
            best_moves = torch.stack(best_moves_arr).to(device)

            #print("Model input shape:", model_input.shape)

            optimizer.zero_grad() 
            logits = model(model_input)
            # print("Logits shape:", logits.shape)
            # print("Best moves shape:", best_moves.shape)
            loss = loss_fn(logits, best_moves)
            running_loss += loss.item()
            # print(f"Got: {torch.argmax(logits, dim=1)} Expected: {torch.argmax(best_moves, dim=1)}")   
            loss.backward()
            optimizer.step()

            if batch_count % 100 == 0:
                print(f"[Batch #{batch_count + 1}] Loss: {running_loss}")
            batch_count += 1

            
        """
        print(f"Epoch {epoch + 1} finished - Loss={loss.item()}")
        model.eval()
        batch_count = 0
        for batch in dl_test:
            count = 0
            correct = 0
            for text in batch["text"]:
                fen, next_move = process_text(text)
                board = chess.Board(fen)
                model_input = board_to_tensor(board, board.turn).to(device)
                best_move = move_to_output_index(board.parse_san(next_move))

                model_input = model_input.unsqueeze(0)
                logits = model(model_input)
                prediction = torch.argmax(logits).item()
                if prediction == best_move:
                    correct += 1
                count += 1
            print(f"[Batch #{batch_count + 1}] Accuracy: {correct / count}")
            batch_count += 1
        """
        
        torch.save(model.state_dict(), "model.pth")
   
        
            
            

if __name__ == "__main__":
    train()