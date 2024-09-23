# Function to draw the board
def draw_board():
    screen.blit(board_image, (0, 0))

# Create dictionary for piece to image mapping

# Function to draw pieces on the board
def draw_pieces(board):
    PIECE_TO_IMAGE = {
        chess.Piece.from_symbol('P'): pygame.image.load('assets/white_pawn.png'),
        chess.Piece.from_symbol('N'): pygame.image.load('assets/white_knight.png'),
        chess.Piece.from_symbol('B'): pygame.image.load('assets/white_bishop.png'),
        chess.Piece.from_symbol('R'): pygame.image.load('assets/white_rook.png'),
        chess.Piece.from_symbol('Q'): pygame.image.load('assets/white_queen.png'),
        chess.Piece.from_symbol('K'): pygame.image.load('assets/white_king.png'),
        chess.Piece.from_symbol('p'): pygame.image.load('assets/black_pawn.png'),
        chess.Piece.from_symbol('n'): pygame.image.load('assets/black_knight.png'),
        chess.Piece.from_symbol('b'): pygame.image.load('assets/black_bishop.png'),
        chess.Piece.from_symbol('r'): pygame.image.load('assets/black_rook.png'),
        chess.Piece.from_symbol('q'): pygame.image.load('assets/black_queen.png'),
        chess.Piece.from_symbol('k'): pygame.image.load('assets/black_king.png'),
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_image = PIECE_TO_IMAGE[piece]
            piece_image = pygame.transform.scale(piece_image, (width // 8, height // 8))
            row, col = divmod(square, 8)
            screen.blit(piece_image, (col * (width // 8), (7 - row) * (height // 8)))

def update_screen(board):
    draw_board()
    draw_pieces(board)
    pygame.display.flip()

# Main game loop
def main():
    board = chess.Board()
    running = True
    selected_square = None

    update_screen(board)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                col, row = x // (width // 8), 7 - y // (height // 8)
                square = chess.square(col, row)

                if selected_square is None:
                    selected_square = square
                else:
                    move = chess.Move(selected_square, square)
                    if move in board.legal_moves:
                        board.push(move)
                    else:
                        move.promotion = chess.QUEEN
                        if move.promotion in board.legal_moves:
                            board.push(move)
                    
                    selected_square = None
                update_screen(board)

    pygame.quit()


def play_ai(ai_color):
    """
    Test positions
    1. 6k1/p5p1/p1p2n1p/3R3P/PB3K2/8/1P1N4/8 b - - 0 34
    2. r1bq1rk1/pp4pp/4p3/2npPp2/5P1b/2NBPKPP/PBP5/R2Q3R b - - 0 16
    3. r1bq1rk1/pp4pp/4p3/3pPp2/5P1b/2NPPKPP/PB6/R2Q3R b - - 0 17
    4. rnbqkb1r/pppppppp/5n2/8/3P4/4P3/PPP2PPP/RNBQKBNR b KQkq - 0 2
    5. 8/5pk1/4p3/6bK/8/4q3/8/8 b - - 6 70
    6. r1bq1rk1/ppp1ppbp/5np1/3PP3/2P5/2N5/PP2B1PP/R1BQK2R b KQ - 0 11
    7. rnbqk2r/ppp2ppp/4p3/2bnP1B1/8/6PN/PP3P1P/RN1QKB1R b KQkq - 1 9
    """
    board = chess.Board("8/8/8/8/8/2k5/5ppp/1K6 b - - 11 85")
    running = True
    selected_square = None
    
    update_screen(board)

    pgn = chess.pgn.Game()
    pgn.headers["White"] = ai_color == chess.WHITE and "AI" or "Human"
    pgn.headers["Black"] = ai_color == chess.BLACK and "AI" or "Human"
    
    node = pgn
    while running:
        if board.is_game_over():
            break

        if board.turn == ai_color:
            move = predict_move(board)
            board.push(move)
            node = node.add_main_variation(move)
            update_screen(board)
            continue
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    col, row = x // (width // 8), 7 - y // (height // 8)
                    square = chess.square(col, row)

                    if selected_square is None:
                        selected_square = square
                    else:
                        move = chess.Move(selected_square, square)
                        if move in board.legal_moves:
                            board.push(move)
                            node = node.add_main_variation(move)
                        # Check for promotion
                        else:
                            move.promotion = chess.QUEEN
                            if move in board.legal_moves:
                                board.push(move)
                                node = node.add_main_variation(move)
                
                        selected_square = None

                    update_screen(board)

    print("Game over. Result:", board.result())
    print()
    pgn.headers["Result"] = board.result()
    print(pgn)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
    pygame.quit()

if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    import chess
    import chess.pgn

    import chess
    import random as rand

    from model.mcts_predict_zero import predict_move
    
    import pygame
    print("Starting game...")
    # Initialize Pygame
    pygame.init()

    # Set up the display
    width, height = 600, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Chess')

    # Load chess board image
    board_image = pygame.image.load('assets/board.png')
    board_image = pygame.transform.scale(board_image, (width, height))

    rand.seed(0)
    random_color = rand.choice([chess.WHITE, chess.BLACK])
    play_ai(random_color)
    # main()
