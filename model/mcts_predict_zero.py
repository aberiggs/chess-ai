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
    def __init__(self, board, parent=None, move=None, max_moves = 10):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0
        self.move_turn = board.turn
        self.move_probabilities = gen_legal_move_predictions(board, max_moves)
        
        self.untried_moves = []
        for move in self.move_probabilities.keys():
                self.untried_moves.append(move)
                
        if parent is None:
            print("Initial move probabilities:")
            for move in self.move_probabilities.keys():
                print(f"Move: {move} P: {self.move_probabilities[move]}")
        
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
    
    def best_child(self, c_param=50):
        flip_val = 1
        if self.move_turn == chess.BLACK:
            flip_val = -1
        
        choices_weights = [((((flip_val * float(c.value)) / c.visits)+1)/2) + c_param * self.move_probabilities[c.move] * ( np.sqrt((2 * np.log(self.visits) / c.visits)) / (c.visits + 1)) for c in self.children]
        
        #if (self.parent is None):
            #for c in self.children:
                #print(f"Q: {(((flip_val * float(c.value)) / c.visits)+1)/2}, P: {self.move_probabilities[c.move]}, U: {c_param * self.move_probabilities[c.move] * ( np.sqrt((2 * np.log(self.visits) / c.visits)) / (c.visits + 1))}")
        
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
            parent.visits += 1
            parent.value += result
            parent = parent.parent
        
def perform_iteration(node):
    # selection
    current_node = node
    while current_node.is_fully_expanded() and len(current_node.children) > 0:
        current_node = current_node.select_child()
    
    # check if node is terminal
    if current_node.board.is_game_over():
        result = 0
        game_result = current_node.board.result()
        if game_result == "1-0":
            result = 1
        elif game_result == "0-1":
            result = -1
        else:
            result = 0
        current_node.update(result)
        return
    
    # expansion
    if not current_node.is_fully_expanded():
        current_node = current_node.add_child(current_node.untried_moves.pop())
    
    # simulation (rollout)
    temp_board = current_node.board.copy()
    depth = 0
    while not temp_board.is_game_over() and depth < 20:
        move = np.random.choice(list(temp_board.legal_moves))
        temp_board.push(move)
    
    sim_val = 0
    if temp_board.is_checkmate():
        game_result = temp_board.result()
        if game_result == "1-0":
            sim_val = 1
        elif game_result == "0-1":
            sim_val = -1
        
    # backpropagation
    inferenced_val = value_model(board_to_tensor(current_node.board, chess.WHITE).to(device).unsqueeze(0)).item()
    #print(inferenced_val)
    #print(current_node.board.fen())
    #print()
    
    mix_coeff = .6 # How much the sim value should be weighted (vs the inferenced value)
    combined_val = mix_coeff * sim_val + (1 - mix_coeff) * inferenced_val
    # if (current_node.parent is not None and current_node.parent.parent is None):
        # print (f"Move: {current_node.move} Simulated value: {sim_val} Inferenced value: {inferenced_val} Combined value: {combined_val}")
    
    current_node.update(combined_val)

def gen_legal_move_predictions(board, max_moves = 10):
    model_input = board_to_tensor(board, board.turn).to(device)
    model_input = model_input.unsqueeze(0)
    logits = model(model_input)
    softmaxed_output = torch.softmax(logits[0], dim=0)
    
    # Sort the probabilities from highest to lowest
    sorted_output = torch.argsort(softmaxed_output, descending=True)
    
    legal_moves = []
    # Get rid of all non-queen promotions
    for move in board.legal_moves:
        if move.promotion is None or move.promotion == chess.QUEEN:
            legal_moves.append(move)
    
    legal_move_predictions = {}
    
    
    # Get the top max_moves legal moves
    i = 0
    while len(legal_move_predictions) < min(max_moves, len(legal_moves)):
        if (i >= len(sorted_output)):
            print("Legal moves:")
            for move in legal_moves:
                print(f"Move: {move}")
            print("legal_move_predictions:")
            for move in legal_move_predictions.keys():
                print(f"Move: {move} P: {legal_move_predictions[move]}")
        move = index_to_move(sorted_output[i].item())
        if move in legal_moves:
            legal_move_predictions[move] = softmaxed_output[move_to_index(move)].item()
        else:
            # Check if move should be a promotion
            move.promotion = chess.QUEEN
            if move in legal_moves:
                legal_move_predictions[move] = softmaxed_output[move_to_index(move)].item() 
        i += 1
    
    return legal_move_predictions
        
def predict_move(board):
    start_time = time.time()
    
    mcts_root = Node(board, max_moves = 20)
    
    print("Performing MCTS...\n")
    
    # create a timer to limit the amount of time spent on MCTS
    sec_allowed = 15
    iterations = 0
    
    time.time()
    while (time.time() - start_time) < (sec_allowed):
        perform_iteration(mcts_root)
        iterations += 1
        
    search_time = time.time() - start_time
    
    best_move = mcts_root.most_visited_child().move
    
    for c in mcts_root.children:
        print(f"Move: {c.move} Visits: {c.visits} Value: {c.value}" + (" (*)" if c.move == best_move else ""))
    
    # print(f"\nBest move's children:")       
    # child_best_move = mcts_root.most_visited_child().most_visited_child().move
    # print(f"Is white: {mcts_root.most_visited_child().move_turn == chess.WHITE}")
    # for c in mcts_root.most_visited_child().children:
        # print(f"Move: {c.move} Visits: {c.visits} Value: {c.value}" + (" (*)" if c.move == child_best_move else ""))

    print(f"\nTotal time taken: {search_time} for {iterations} iterations ({iterations / search_time} iterations/sec)")
    print("--------------------")
    
    return best_move