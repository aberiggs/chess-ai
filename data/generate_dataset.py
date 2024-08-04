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

def parse_pgn_file(file, max_games = 0) -> list:
    games = []
    with open(file) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None: # No more games in file
                break 
            games.append(game)
            if len(games) == max_games:
                break

    print("Parsed", len(games), "games")
    return games

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
            data.append({'board': board.copy(), 'next_move': move})
        board.push(move)
    return data

def generate_dataset(pgn_file: str, dst_file: str):
    print("Generating dataset from", os.path.abspath(pgn_file))
    games = parse_pgn_file(pgn_file, 800000)
    data = []
    games_in_set = 0
    for game in games:
        white_elo = int(game.headers['WhiteElo'])
        black_elo = int(game.headers['BlackElo'])
        if white_elo < 2000 and black_elo < 2000:
            continue
        data += extract_data(game)
        games_in_set += 1

    print(f"Finished extracting moves from {games_in_set} games")

    dataset = ChessDataset(data)
    print("Generated a dataset with", len(dataset), "samples")
    torch.save(dataset, dst_file)
    print("Saved dataset to", os.path.abspath(dst_file))

if __name__ == "__main__":
    pgn_file = "data.pgn"
    dst_file = "lichess_2017_02_dataset.pth"
    generate_dataset(pgn_file, dst_file)