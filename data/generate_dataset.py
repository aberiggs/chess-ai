import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import zstandard as zstd
import chess
import chess.pgn

from data.chess_dataset import ChessDataset

def decompress_zstd(in_file, out_file):
    with open(in_file, 'rb') as compressed_file:
        decomp = zstd.ZstdDecompressor()
        with open(out_file, 'wb') as decompressed_file:
            decomp.copy_stream(compressed_file, decompressed_file)

def extract_data(game) -> list:
    data = []
    board = game.board()
    game_result = game.headers['Result'][0]
    if game_result == '1':
        winning_color = chess.WHITE
    else:
        winning_color = chess.BLACK
    
    for move in game.mainline_moves():
        if board.turn == winning_color:
            data.append({'fen': board.fen(), 'next_move': move})
        board.push(move)
    return data

def parse_pgn_file(file, max_games = 0) -> list:
    data = []
    tenths_done = -1
    game_count = 0
    extracted_game_count = 0
    with open(file) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None: # No more games in file
                break 
            game_count += 1
            
            white_elo = int(game.headers['WhiteElo'])
            black_elo = int(game.headers['BlackElo'])
            if white_elo > 2000 or black_elo > 2000:
                data += extract_data(game)
                extracted_game_count += 1
            
            if max_games > 0 and tenths_done < (game_count / max_games) * 10:
                tenths_done += 1
                print("[", end="")
                print("=" * tenths_done, end="")
                print("." * (10 - tenths_done), end="")
                print("]")
                
            if game_count == max_games:
                break

    print(f"Extracted {len(data)} moves from {extracted_game_count} games")
    return data

def generate_dataset(pgn_file: str, dst_file: str):
    print("Parsing pgn: ", os.path.abspath(pgn_file))
    data = parse_pgn_file(pgn_file, 1000000)

    dataset = ChessDataset(data)
    print("Generated a dataset with", len(dataset), "samples")
    torch.save(dataset, dst_file)
    print("Saved dataset to", os.path.abspath(dst_file))

if __name__ == "__main__":
    compressed_file = "lichess_2018-03.pgn.zst" 
    pgn_file = "data.pgn"
    decompress_zstd(compressed_file, "data.pgn")
    dst_file = compressed_file.split(".")[0] + "_dataset.pth"
    generate_dataset(pgn_file, dst_file)
    # remove the pgn file
    os.remove(pgn_file)