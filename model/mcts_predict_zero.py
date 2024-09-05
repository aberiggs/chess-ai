import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time


import chess
import torch
import numpy as np

from model.model import ChessModel, ChessValueModel 

from data.conversions import board_to_tensor, index_to_move, move_to_index

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

model = None
value_model = None

value_model_path = "../model/chess_value_model.pth"
value_model = ChessValueModel().to(device)
value_model.load_state_dict(torch.load(value_model_path, weights_only=True))
value_model.eval()
print("Value model loaded")

# model_fast_path = "../model/chess_model_fast.pth"
# fast_model = FastChessModel().to(device)
# fast_model.load_state_dict(torch.load(model_fast_path, weights_only=True))
# fast_model.eval()
# print("Fast policy model loaded")

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
        self.move_probabilities = gen_legal_move_predictions(board)
        
        if untried_moves is None:
            self.untried_moves = list(board.legal_moves)
        else:
            self.untried_moves = untried_moves
        
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
    
    def best_child(self, c_param=0.6):
        max_prob = max(self.move_probabilities.values())
        #values = {c: value_model(board_to_tensor(c.board, c.board.turn).to(device).unsqueeze(0)).item() for c in self.children}
        lamb = 0.4

        choices_weights = [(((float(c.value) / c.visits)+1)/2) + (c_param * (self.move_probabilities[c.move] / max_prob) / (c.visits + 1)) for c in self.children]
        return self.children[np.argmax(choices_weights)]
    
    def highest_winrate_child(self):
        choices_weights = [c.value / c.visits for c in self.children]
        return self.children[np.argmax(choices_weights)]
    
    def most_visited_child(self):
        if self.move == None:
            for c in self.children:
                value_estimate = value_model(board_to_tensor(c.board, c.board.turn).to(device).unsqueeze(0)).item()
                print(f"Value estimate of {c.move} is {value_estimate}")
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
        if (self.turn == chess.WHITE):
            result = -result
            
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
    #temp_board = current_node.board.copy()
    #depth = 0
    #while not temp_board.is_game_over() and depth < 80:
        #move = np.random.choice(list(temp_board.legal_moves))
        #temp_board.push(move)
    
        
    # backpropagation
    #result = 0
    #if temp_board.is_checkmate():
        #result = 1 if temp_board.turn == current_node.turn else -1
        
    current_node.update(value_model(board_to_tensor(current_node.board, current_node.board.turn).to(device).unsqueeze(0)).item())

def gen_legal_move_predictions(board):
    model_input = board_to_tensor(board, board.turn).to(device)
    model_input = model_input.unsqueeze(0)
    logits = model(model_input)
    softmaxed_output = torch.softmax(logits[0], dim=0)
    
    legal_move_predictions = {}
    
    for move in board.legal_moves:        
        index = move_to_index(move)
        legal_move_predictions[move] = softmaxed_output[index].item()
        
    return legal_move_predictions
        

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
    for move in potential_moves:
        print(f"Move: {move} Probability: {softmaxed_output[move_to_index(move)].item()}")
    print()
    return potential_moves

def predict_move(board):
    start_time = time.time()
    potential_moves = gen_moves(board, 3)
    
    mcts_root = Node(board, untried_moves=potential_moves)
    
    print("Performing MCTS...\n")
    for _ in range(500):
        perform_iteration(mcts_root)
        
    
    best_move = mcts_root.most_visited_child().move
    
    for c in mcts_root.children:
        print(f"Move: {c.move} Visits: {c.visits} Value: {c.value}" + (" (*)" if c.move == best_move else ""))

    print(f"\nTotal time taken: {time.time() - start_time}")
    print("--------------------")
    
    return best_move