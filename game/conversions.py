import chess
import torch

def board_to_tensor(board, color):
    """
    Convert a chess board to a tensor
    """
    # Initialize tensor
    tensor = torch.zeros(15, 8, 8)
    
    # Fill tensor with pieces
    for i in range(8):
        for j in range(8):
            piece = board.piece_at(chess.square(i, j))
            if piece is not None:
                piece_type = piece.piece_type
                piece_color = piece.color
                tensor[piece_type + 6 * (piece_color == color), i, j] = 1
    
    # Fill tensor with en passant squares
    if board.ep_square is not None:
        tensor[12, board.ep_square // 8, board.ep_square % 8] = 1
    
    # Fill tensor with castling rights
    white_castle_channel = 13 if color is chess.WHITE else 14
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[white_castle_channel, 6, 0] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[white_castle_channel, 2, 0] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[27 - white_castle_channel, 6, 7] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[27 - white_castle_channel, 2, 7] = 1
    
    return tensor

def move_to_output_index(move):
    """
    Convert a move to the coordinate in the output tensor (8x8x8x8)
    """
    # Get the coordinates of the move
    from_square = move.from_square
    to_square = move.to_square
    
    # Convert the coordinates to the output tensor
    from_x, from_y = from_square // 8, from_square % 8
    to_x, to_y = to_square // 8, to_square % 8

    # 0bjjjkkklllmmm : j=from_x, k=from_y, l=to_x, m=to_y
    return from_x << 9 + from_y << 6 + to_x << 3 + to_y

    