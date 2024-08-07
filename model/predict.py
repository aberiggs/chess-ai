import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chess
from conversions import board_to_tensor, move_to_index

def logits_to_move(logits, board) -> chess.Move:
    legal_moves = list(board.legal_moves)

    index_to_move = {move_to_index(move): move for move in legal_moves}

    # get the se
    output_layer = logits[0].tolist()

    best_move = legal_moves[0]
    for index in index_to_move:
        if output_layer[index] > output_layer[move_to_index(best_move)]:
            best_move = index_to_move[index]
    return best_move

def predict(board, model, device):
    model_input = board_to_tensor(board, board.turn).to(device)
    model_input = model_input.unsqueeze(0)
    logits = model(model_input)
    return logits_to_move(logits, board)


def main():
    # Create chess game to test
    import chess
    import chess.pgn
    import torch
    from model import ChessModel #, ChessModel_V1

    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    modela = ChessModel().to(device)
    modela.load_state_dict(torch.load("model.pth", weights_only=True))
    modela.eval()
    modelb = ChessModel().to(device)
    modelb.load_state_dict(torch.load("model.pth", weights_only=True))
    modelb.eval()
    board = chess.Board();

    pgn = chess.pgn.Game()
    node = pgn
    model = None
    while not board.is_game_over():
        print(board)
        if board.turn == chess.WHITE:
            model = modelb
        else:
            model = modela
        move = predict(board, model, device)
        board.push(move)
        # add move to pgn
        node = node.add_main_variation(move)
        print("\n")

    print(board.fen())
    print()

    pgn.headers["Result"] = board.result()
    print(pgn)
    print()

main()