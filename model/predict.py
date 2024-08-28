import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time

from torch.multiprocessing import Pool, Queue
import torch.multiprocessing as multiprocessing 

import chess
import torch

from model.model import ChessModel, FastChessModel

from data.conversions import board_to_tensor, index_to_move


model = None
proc_model = None

is_child = multiprocessing.parent_process() is not None
if is_child:
    model_fast_path = "../model/chess_model_fast.pth"
    proc_model = FastChessModel().to("cuda")
    proc_model.load_state_dict(torch.load(model_fast_path, weights_only=True))
    proc_model.eval()
else:
    multiprocessing.set_start_method('spawn', force=True)
    model_path = "../model/chess_model.pth"
    model = ChessModel().to("cuda")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print("Model loaded")
    
def move_gen(board, model, device, max_moves=3, min_prob=0.0):
    board_fen = board.fen()
    # if board_fen in saved_outputs:
        # return saved_outputs[board_fen]
        
    
    #gen_time = time.time()
    model_input = board_to_tensor(board, board.turn).to(device)
    model_input = model_input.unsqueeze(0)
    #print(f"Generated model input in {time.time() - gen_time} seconds")
    logits = model(model_input)
    #print(f"Generated logits in {time.time() - gen_time} seconds")
    legal_moves = list(board.legal_moves)
    
    softmaxed_output = torch.softmax(logits[0], dim=0)
    sorted_probs = torch.argsort(softmaxed_output, descending=True)
    # print(f"Generated sorted probs in {time.time() - gen_time} seconds")
    
    # print("Top 10 moves:")
    # for i in range(10):
        # print(index_to_move(sorted_probs[i].item()), softmaxed_output[sorted_probs[i].item()].item())
    # print()
  
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

    if depth < 2:
        potential_moves = move_gen(board, model, device, 6)
  
    move = potential_moves[torch.randint(len(potential_moves), (1,)).item()]

    board.push(move)
    val = simulate(board, model, device, depth + 1)
    board.pop()
    return val

def record_simulation(board, device):
    if proc_model is None:
        print("Model not set")
        return
    
    try:
        result = simulate(board, proc_model, device)
        return result
    except:
        return -2

def predict(board, device):
    start_time = time.time()
    if model is None:
        print("Model not set")
    potential_moves = move_gen(board, model, device, 3)
    move_results = {}
    value_map = {}
    
    futures = {}
    pool = Pool(6)
    for move in potential_moves:
        board.push(move)
        
        move_results[move] = []
        futures[move] = []
        
        sim_size = 3000
        print("\nSimulating", sim_size, "games for move", move)
        
        items = [(board.copy(), device) for _ in range(sim_size)]
        
        futures[move].extend(pool.starmap(record_simulation, items, chunksize=50))
        board.pop()
    
    pool.close()
    pool.join()

    print()
    for move in futures:
        value = 0
        terminal_nodes = 0
        for future in futures[move]:
            result = future
            if result == -2:
                continue
            terminal_nodes += 1
            value += result
        
        if board.turn == chess.BLACK:
            value = -value
        
        if terminal_nodes == 0:
            value_map[move] = 0
        else:
            value_map[move] = value/terminal_nodes

        print(f"Move: {move} Value: {value_map[move]} Terminal Nodes: {terminal_nodes}")
        
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
    print("Starting...")
    #main()
