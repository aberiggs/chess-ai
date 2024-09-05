import torch
import chess
import chess.pgn

from mcts_predict import predict_move as mcts_move
from mcts_predict_zero import predict_move as mcts_zero_move

def play_game():
    board = chess.Board()
    
    pgn = chess.pgn.Game()
    node = pgn
    
    pgn.headers["White"] = "Sprock AI (MCTS)"
    pgn.headers["Black"] =  "Sprock AI (MCTS Zero)"
    
    while not board.is_game_over():
        print(board)
        move = None
        if board.turn == chess.WHITE:
            move = mcts_move(board)
        else:
            move = mcts_zero_move(board)
        board.push(move)
        node = node.add_main_variation(move)
        print("\n")
    
    print(board)
    print(board.fen())

    print()

    pgn.headers["Result"] = board.result()
    print(pgn)
    print()
    
    

if __name__ == "__main__":
    
    play_game();
    