import math

# Create empty board
board = [' ' for _ in range(9)]

def print_board():
    for row in [board[i*3:(i+1)*3] for i in range(3)]:
        print('| ' + ' | '.join(row) + ' |')

def is_winner(brd, player):
    win_combinations = [
        [0,1,2], [3,4,5], [6,7,8], # rows
        [0,3,6], [1,4,7], [2,5,8], # columns
        [0,4,8], [2,4,6]           # diagonals
    ]
    for combo in win_combinations:
        if all(brd[i] == player for i in combo):
            return True
    return False

def is_draw(brd):
    return ' ' not in brd

def get_available_moves(brd):
    return [i for i in range(9) if brd[i] == ' ']

# Minimax algorithm
def minimax(brd, is_maximizing):
    if is_winner(brd, 'O'):  # AI wins
        return 1
    elif is_winner(brd, 'X'):  # Human wins
        return -1
    elif is_draw(brd):
        return 0

    if is_maximizing:
        best_score = -math.inf
        for move in get_available_moves(brd):
            brd[move] = 'O'
            score = minimax(brd, False)
            brd[move] = ' '
            best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for move in get_available_moves(brd):
            brd[move] = 'X'
            score = minimax(brd, True)
            brd[move] = ' '
            best_score = min(score, best_score)
        return best_score

def best_move():
    best_score = -math.inf
    move = -1
    for i in get_available_moves(board):
        board[i] = 'O'
        score = minimax(board, False)
        board[i] = ' '
        if score > best_score:
            best_score = score
            move = i
    return move

# Game loop
def play_game():
    print("Welcome to Tic-Tac-Toe!")
    print_board()

    while True:
        # Human move
        try:
            move = int(input("Enter your move (0-8): "))
            if board[move] != ' ':
                print("Invalid move! Try again.")
                continue
        except (ValueError, IndexError):
            print("Enter a number between 0 and 8.")
            continue

        board[move] = 'X'

        if is_winner(board, 'X'):
            print_board()
            print("You win!")
            break
        elif is_draw(board):
            print_board()
            print("It's a draw!")
            break

        # AI move
        ai_move = best_move()
        board[ai_move] = 'O'
        print("\nAI plays:")
        print_board()

        if is_winner(board, 'O'):
            print("AI wins!")
            break
        elif is_draw(board):
            print("It's a draw!")
            break

# Start the game
if __name__ == "__main__":
    play_game()
