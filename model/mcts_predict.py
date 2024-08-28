import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time

import torch.multiprocessing as multiprocessing 

import chess
import torch
import numpy as np

from model.model import ChessModel, FastChessModel

from data.conversions import board_to_tensor, index_to_move

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

model = None
fast_model = None

is_child = multiprocessing.parent_process() is not None

model_fast_path = "../model/chess_model_fast.pth"
fast_model = FastChessModel().to(device)
fast_model.load_state_dict(torch.load(model_fast_path, weights_only=True))
fast_model.eval()
print("Fast policy model loaded")

if not is_child:
    multiprocessing.set_start_method('spawn', force=True)
    model_path = "../model/chess_model.pth"
    model = ChessModel().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print("Main policy model loaded")
    
# Define a monte carlo tree search node
class Node:
    def __init__(self, board, parent=None, move=None, untried_moves=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0
        self.turn = board.turn
        if untried_moves is None:
            self.untried_moves = list(board.legal_moves)
            #gen_moves_fast(board, 5) # Theoretically should be more accurate, but it's not
        else:
            self.untried_moves = untried_moves
        
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
    
    def best_child(self, c_param=0.8):
        choices_weights = [(((float(c.value) / c.visits)+1)/2) + c_param * np.sqrt((2 * np.log(self.visits) / c.visits)) for c in self.children]
        return self.children[np.argmax(choices_weights)]
    
    def highest_winrate_child(self):
        choices_weights = [c.value / c.visits for c in self.children]
        return self.children[np.argmax(choices_weights)]
    
    def most_visited_child(self):
        choices_weights = [c.visits for c in self.children]
        return self.children[np.argmax(choices_weights)]
    
    def select_child(self):
        if not self.is_fully_expanded():
            return self.add_child(self.untried_moves.pop())
        else:
            return self.best_child()
        
    def add_child(self, move):
        new_board = self.board.copy()
        new_board.push(move)
        child = Node(new_board, self, move)
        self.children.append(child)
        return child
    
    def update(self, result):
        self.visits += 1
        self.value += result
        
        parent = self.parent
        while parent is not None:
            result = -result
            parent.visits += 1
            parent.value += result
            parent = parent.parent
        
    

def perform_iteration(node):
    # selection
    current_node = node
    while current_node.is_fully_expanded() and len(current_node.children) > 0:
        current_node = current_node.select_child()
    
    # expansion
    if not current_node.is_fully_expanded():
        current_node = current_node.add_child(current_node.untried_moves.pop())
    
    # simulation (rollout)
    temp_board = current_node.board.copy()
    depth = 0
    while not temp_board.is_game_over() and depth < 80:
        move = np.random.choice(list(temp_board.legal_moves))
        temp_board.push(move)
        
    # backpropagation
    result = 0
    if temp_board.is_checkmate():
        result = 1 if temp_board.turn == current_node.turn else -1
    current_node.update(result)


def gen_moves_fast(board, max_moves = 3):
    return gen_moves_base(board, fast_model, max_moves)
    
def gen_moves(board, max_moves = 3):
    return gen_moves_base(board, model, max_moves)
    
def gen_moves_base(board, target_model, max_moves = 3):
    assert(target_model is not None)
    
    model_input = board_to_tensor(board, board.turn).to(device)
    model_input = model_input.unsqueeze(0)
    logits = target_model(model_input)
    legal_moves = list(board.legal_moves)
    
    softmaxed_output = torch.softmax(logits[0], dim=0)
    sorted_probs = torch.argsort(softmaxed_output, descending=True)
  
    potential_moves = []
    i = 0
    while len(potential_moves) < min(max_moves, len(legal_moves)):
        if i == len(sorted_probs):
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

    assert(len(potential_moves) > 0)
    return potential_moves

def predict_move(board):
    start_time = time.time()
    potential_moves = gen_moves(board, 3)
    
    mcts_root = Node(board, untried_moves=potential_moves)
    
    print("Performing MCTS...\n")
    for _ in range(1000):
        perform_iteration(mcts_root)
        
    
    best_move = mcts_root.most_visited_child().move
    
    for c in mcts_root.children:
        print(f"Move: {c.move} Visits: {c.visits} Value: {c.value}" + (" (*)" if c.move == best_move else ""))

    print(f"\nTotal time taken: {time.time() - start_time}")
    print("--------------------")
    
    return best_move