import random

class ChessBoard:
    def __init__(self):
        self.board = [[""] * 8 for _ in range(8)]
        self.alph = "abcdefgh"
        self.w_king_moved = False
        self.w_rook_moved = [False, False]
        self.b_king_moved = False
        self.b_rook_moved = [False, False]
        self.en_passant_target = None

    def setup_board(self):
        pieces = "RNBQKBNR"
        for i in range(8):
            self.board[0][i] = "W" + pieces[i]
            self.board[1][i] = "WP"
            self.board[6][i] = "BP"
            self.board[7][i] = "B" + pieces[i]

    def parse_move(self, mv):
        if mv == "O-O":
            return "O-O"
        elif mv == "O-O-O":
            return "O-O-O"
        else:
            x1, y1, x2, y2 = ord(mv[0]) - ord('a'), int(mv[1]) - 1, ord(mv[2]) - ord('a'), int(mv[3]) - 1
            return x1, y1, x2, y2

    def move(self, mv):
        x1, y1, x2, y2 = self.parse_move(mv)
        piece = self.board[y1][x1]
        self.board[y2][x2] = piece
        self.board[y1][x1] = ""

            # Handle en passant
        if piece[1] == 'P':
            if abs(y2 - y1) == 2:
                self.en_passant_target = (y2, x2)
            elif (y2, x2) == self.en_passant_target:
                self.board[y1][x2] = ""

            # Promote pawn to queen
        if piece[1] == 'P' and y2 in [0, 7]:
            self.board[y2][x2] = piece[0] + 'Q'

            # Handle castling flags
        if piece == "WK":
            self.w_king_moved = True
            if(abs(x2-x1) > 1):
                rx = 0 if x2 < x1 else 7
                d = (x1 - x2) // 2
                self.board[y2][x2 + d] = "WR"
                self.board[y2][rx] = ""
        elif piece == "BK":
            self.b_king_moved = True
            if(abs(x2-x1) > 1):
                rx = 0 if x2 < x1 else 7
                d = (x1 - x2) // 2
                self.board[y2][x2 + d] = "BR"
                self.board[y2][rx] = ""

        if piece[1] == 'R':
            if y1 == 0:
                self.w_rook_moved[0 if x1 == 0 else 1] = True
            elif y1 == 7:
                self.b_rook_moved[0 if x1 == 0 else 1] = True

    def checking_moves(self, turn):
        moves = []
        for y in range(8):
            for x in range(8):
                if self.board[y][x] and self.board[y][x][0] == turn:
                    piece = self.board[y][x][1]
                    moves += getattr(self, f'{piece.lower()}_moves')(x, y, turn)
        return moves

    def possible_moves(self, turn):
        moves = []
        for y in range(8):
            for x in range(8):
                if self.board[y][x] and self.board[y][x][0] == turn:
                    piece = self.board[y][x][1]
                    moves += getattr(self, f'{piece.lower()}_moves')(x, y, turn)
        # Add castling moves
        king = self.find_king(turn)
        if king == None:
            return moves
        if turn == 'W' and not self.w_king_moved:
            if not self.w_rook_moved[0] and all(self.is_empty(c, 0) for c in range(1, 4)):
                if not self.is_check(turn) and not self.is_check_path(4, 0, 2, 0, turn):
                    moves.append(self.move_str(king[0], king[1], king[0] - 2, king[1]))
            if not self.w_rook_moved[1] and all(self.is_empty(c, 0) for c in range(5, 7)):
                if not self.is_check(turn) and not self.is_check_path(4, 0, 6, 0, turn):
                    moves.append(self.move_str(king[0], king[1], king[0] + 2, king[1]))
        if turn == 'B' and not self.b_king_moved:
            if not self.b_rook_moved[0] and all(self.is_empty(c, 7) for c in range(1, 4)):
                if not self.is_check(turn) and not self.is_check_path(4, 7, 2, 7, turn):
                    moves.append(self.move_str(king[0], king[1], king[0] - 2, king[1]))
            if not self.b_rook_moved[1] and all(self.is_empty(c, 7) for c in range(5, 7)):
                if not self.is_check(turn) and not self.is_check_path(4, 7, 6, 7, turn):
                    moves.append(self.move_str(king[0], king[1], king[0] + 2, king[1]))
        return moves

    def is_check_path(self, x1, y1, x2, y2, turn):
        step_x = 1 if x2 > x1 else -1
        for x in range(x1, x2 + step_x, step_x):
            if self.is_under_attack(x, y1, turn):
                return True
        return False

    def is_under_attack(self, x, y, turn):
        opponent_turn = 'B' if turn == 'W' else 'W'
        opponent_moves = self.checking_moves(opponent_turn)
        return any(mv[2:] == f'{self.alph[x]}{y + 1}' for mv in opponent_moves)

    def p_moves(self, x, y, turn):
        direction = 1 if turn == 'W' else -1
        moves = []
        if self.is_empty(x, y + direction):
            moves.append(self.move_str(x, y, x, y + direction))
            if (y == 1 and turn == 'W') or (y == 6 and turn == 'B'):
                if self.is_empty(x, y + 2 * direction):
                    moves.append(self.move_str(x, y, x, y + 2 * direction))
        for dx in [-1, 1]:
            if (y + direction, x + dx) == self.en_passant_target:
                moves.append(self.move_str(x, y, x + dx, y + direction))
            elif self.is_enemy(x + dx, y + direction, turn):
                moves.append(self.move_str(x, y, x + dx, y + direction))
        return moves

    def n_moves(self, x, y, turn):
        moves = []
        for dx, dy in [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]:
            if self.is_valid(x + dx, y + dy) and not self.is_friendly(x + dx, y + dy, turn):
                moves.append(self.move_str(x, y, x + dx, y + dy))
        return moves

    def b_moves(self, x, y, turn):
        return self.sliding_moves(x, y, turn, [(1, 1), (1, -1), (-1, 1), (-1, -1)])

    def r_moves(self, x, y, turn):
        return self.sliding_moves(x, y, turn, [(1, 0), (-1, 0), (0, 1), (0, -1)])

    def q_moves(self, x, y, turn):
        return self.b_moves(x, y, turn) + self.r_moves(x, y, turn)

    def k_moves(self, x, y, turn):
        moves = []
        for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0)]:
            if self.is_valid(x + dx, y + dy) and not self.is_friendly(x + dx, y + dy, turn):
                moves.append(self.move_str(x, y, x + dx, y + dy))
        return moves

    def sliding_moves(self, x, y, turn, directions):
        moves = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            while self.is_valid(nx, ny) and not self.is_friendly(nx, ny, turn):
                moves.append(self.move_str(x, y, nx, ny))
                if self.is_enemy(nx, ny, turn):
                    break
                nx, ny = nx + dx, ny + dy
        return moves

    def is_valid(self, x, y):
        return 0 <= x < 8 and 0 <= y < 8

    def is_empty(self, x, y):
        return self.is_valid(x, y) and self.board[y][x] == ""

    def is_friendly(self, x, y, turn):
        return self.is_valid(x, y) and self.board[y][x] != "" and self.board[y][x][0] == turn

    def is_enemy(self, x, y, turn):
        return self.is_valid(x, y) and self.board[y][x] != "" and self.board[y][x][0] != turn

    def move_str(self, x1, y1, x2, y2):
        return f'{self.alph[x1]}{y1 + 1}{self.alph[x2]}{y2 + 1}'

    def evaluate_board(self):
        value = 0
        #Material value
        piece_values = {'P': 1, 'N': 3, 'B': 3.25, 'R': 5, 'Q': 9, 'K': 100000}
        for y in range(8):
            for x in range(8):
                if self.board[y][x]:
                    dir = 1 if self.board[y][x][0] == 'W' else -1
                    value += piece_values[self.board[y][x][1]] * dir
        #Space value (number of moves you can make)
        space = len(self.possible_moves('W')) - len(self.possible_moves('B'))
        return value + 0.03 * space

    def minimax(self, depth, turn, alpha=-float('inf'), beta=float('inf')):
        if depth == 0 or self.is_checkmate(turn):
            return ('', self.evaluate_board())
        moves = self.possible_moves(turn)
        if not moves:
            return ('', self.evaluate_board())
        next_turn = 'B' if turn == 'W' else 'W'
        best_move = ''
        if turn == 'W':
            max_eval = -float('inf')
            for move in moves:
                copy_board = ChessBoard()
                copy_board.board = [row[:] for row in self.board]
                copy_board.w_king_moved = self.w_king_moved
                copy_board.w_rook_moved = self.w_rook_moved
                copy_board.b_king_moved = self.b_king_moved
                copy_board.b_rook_moved = self.b_rook_moved
                copy_board.en_passant_target = self.en_passant_target
                copy_board.move(move)
                current_eval = copy_board.minimax(depth - 1, next_turn, alpha, beta)[1] + random.randrange(-100, 100) / 100000
                if current_eval > max_eval:
                    max_eval = current_eval
                    best_move = move
                alpha = max(alpha, current_eval)
                if beta <= alpha:
                    break
            return (best_move, max_eval)
        else:
            min_eval = float('inf')
            for move in moves:
                copy_board = ChessBoard()
                copy_board.board = [row[:] for row in self.board]
                copy_board.move(move)
                current_eval = copy_board.minimax(depth - 1, next_turn, alpha, beta)[1]
                if current_eval < min_eval:
                    min_eval = current_eval
                    best_move = move
                beta = min(beta, current_eval)
                if beta <= alpha:
                    break
            return (best_move, min_eval)

    def is_legal_move(self, mv, turn):
        legal_moves = self.possible_moves(turn)
        return mv in legal_moves

    def is_check(self, turn):
        king_pos = self.find_king(turn)
        if not king_pos:
            return False
        opponent_turn = 'B' if turn == 'W' else 'W'
        opponent_moves = self.checking_moves(opponent_turn)
        return any(mv[2:] == f'{self.alph[king_pos[0]]}{king_pos[1] + 1}' for mv in opponent_moves)

    def is_checkmate(self, turn):
        return self.is_check(turn) and not any(self.is_legal_move(mv, turn) for mv in self.possible_moves(turn))

    def find_king(self, turn):
        for y in range(8):
            for x in range(8):
                if self.board[y][x] == turn + 'K':
                    return x, y
        return None

class ChessEngine:
    def __init__(self):
        self.board = ChessBoard()
        self.board.setup_board()

    def get_best_move(self, depth, turn):
        return self.board.minimax(depth, turn)[0]

    def move(self, mv):
        self.board.move(mv)

    def print_board(self):
        print("    a  b  c  d  e  f  g  h")
        print("  +-------------------------+")
        for i, row in enumerate(self.board.board):
            row_str = f"{1 + i} | " + ' '.join(row or '. ' for row in row) + f" | {1 + i}"
            print(row_str)
        print("  +-------------------------+")
        print("    a  b  c  d  e  f  g  h")
        print()

def main():
    engine = ChessEngine()
    player_color = input("Do you want to play as White or Black? ").strip().lower()
    if player_color not in ['white', 'black']:
        print("Invalid input. Please enter 'White' or 'Black'.")
        return
    player_turn = 'W' if player_color == 'white' else 'B'
    ai_turn = 'B' if player_turn == 'W' else 'W'
    turn = 'W'
    
    while True:
        engine.print_board()
        if turn == player_turn:
            player_move = input("Enter your move (e.g., e2e4, O-O, O-O-O): ").strip()
            if engine.board.is_legal_move(player_move, player_turn):
                engine.move(player_move)
                print(f"{turn}'s move: {player_move}")
            else:
                print("Illegal move. Try again.")
                continue
        else:
            best_move = engine.get_best_move(4, ai_turn)
            if best_move:
                engine.move(best_move)
                print(f"Engine move: {best_move}")
            else:
                print("No valid moves available. Stalemate!")
                break

        if engine.board.is_checkmate(turn):
            print(f"Checkmate! {'White' if turn == 'B' else 'Black'} wins.")
            break
        if engine.board.is_check(turn):
            print(f"Check! {'White' if turn == 'W' else 'Black'} is in check.")
        
        turn = ai_turn if turn == player_turn else player_turn

if __name__ == "__main__":
    main()
