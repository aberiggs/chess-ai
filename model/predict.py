import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time

import chess
import torch

from data.conversions import board_to_tensor, index_to_move

# store the model output in a dictionary with the move as the key
saved_outputs = {}

def move_gen(board, model, device, max_moves=3, min_prob=0.0):
    board_fen = board.fen()
    if board_fen in saved_outputs:
        return saved_outputs[board_fen]
    
    gen_time = time.time()
    model_input = board_to_tensor(board, board.turn).to(device)
    model_input = model_input.unsqueeze(0)
    # print(f"Generated model input in {time.time() - gen_time} seconds")
    logits = model(model_input)
    # print(f"Generated logits in {time.time() - gen_time} seconds")
    legal_moves = list(board.legal_moves)
    
    softmaxed_output = torch.softmax(logits[0], dim=0)
    sorted_probs = torch.argsort(softmaxed_output, descending=True)
    # print(f"Generated sorted probs in {time.time() - gen_time} seconds")
  
    potential_moves = []
    i = 0
    while len(potential_moves) < min(max_moves, len(legal_moves)):
        if i == len(sorted_probs): #or softmaxed_output[sorted_probs[i].item()].item() < min_prob:
            break
        move = index_to_move(sorted_probs[i].item())
        
        # Check if move is a promotion
        if move in legal_moves:
            potential_moves.append(move)
        else:
            move.promotion = chess.QUEEN
            if move in legal_moves:
                potential_moves.append(move)
        i += 1

    if len(potential_moves) == 0:
        # If we have no moves, just return the 1st legal move
        potential_moves = [legal_moves[0]]
        
    saved_outputs[board_fen] = potential_moves
        
    # print(f"Took {time.time() - gen_time} seconds to generate {len(potential_moves)} move(s)\n")
    return potential_moves

def simulate(board, model, device, depth = 0):
    if board.is_game_over():
        result = board.result()
        if result == "1-0":
            return 1
        elif result == "0-1":
            return -1
        else:
            return 0 
    
    if depth > 80:
        return -2
    
    potential_moves = list(board.legal_moves)

    if depth < 1:
        potential_moves = move_gen(board, model, device, 6)
  
    move = potential_moves[torch.randint(len(potential_moves), (1,)).item()]

    board.push(move)
    val = simulate(board, model, device, depth + 1)
    board.pop()
    return val
        
def predict(board, model, model_fast, device):
    potential_moves = move_gen(board, model, device, 3)
    value_map = {}
    start_time = time.time()
    for move in potential_moves:
        board.push(move)
        terminal_nodes = 0
        value = 0
        sim_size = 2000
        sim_time = time.time()
        print("\nSimulating", sim_size, "games for move", move)
        for _ in range(sim_size):
            this_val = simulate(board, model_fast, device)
            if this_val >= -1 and this_val <= 1:
                terminal_nodes += 1
                value += this_val
        print(f"Sim time for {sim_size} took {time.time() - sim_time} seconds. Average sim time was {(time.time() - sim_time) / sim_size} seconds.")
        board.pop()
        
        value = value / terminal_nodes
        if board.turn == chess.BLACK:
            value = -value

        print("Encountered", terminal_nodes, "terminal nodes with an average value of", value)
        value_map[move] = value
        
    
    print(f"\nTotal time taken: {time.time() - start_time}")
    print("--------------------")
    return max(value_map, key=value_map.get)

def main():
    # Create chess game to test
    import chess
    import chess.pgn
    import torch
    from model import ChessModel, ChessModel_V1, ChessModel_V4

    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    modela = ChessModel_V1().to(device)
    modela.load_state_dict(torch.load("8-4-2024.pth", weights_only=True))
    modela.eval()
    modelb = ChessModel_V4().to(device)
    modelb.load_state_dict(torch.load("v4_model.pth", weights_only=True))
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

if __name__ == "__main__":
    main()