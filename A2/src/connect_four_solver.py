import math
import random
import numpy as np

# Board dimensions for Connect Four
ROW_COUNT = 6
COLUMN_COUNT = 7

def create_board():
    """
    Creates an empty Connect Four board (numpy 2D array).

    Returns:
    np.ndarray:
        A 2D numpy array of shape (ROW_COUNT, COLUMN_COUNT) filled with zeros (float).
    """
    return np.zeros((ROW_COUNT,COLUMN_COUNT), dtype=int)


def drop_piece(board, row, col, piece):
    """
    Places a piece (1 or 2) at the specified (row, col) position on the board.

    Parameters:
    board (np.ndarray): The current board, shape (ROW_COUNT, COLUMN_COUNT).
    row (int): The row index where the piece should be placed.
    col (int): The column index where the piece should be placed.
    piece (int): The piece identifier (1 or 2).
    
    Returns:
    None. The 'board' is modified in-place. Do NOT return a new board!
    """
    # TODO: implement
    board[row][col] = piece


def is_valid_location(board, col):
    """
    Checks if dropping a piece in 'col' is valid (column not full).

    Parameters:
    board (np.ndarray): The current board.
    col (int): The column index to check.

    Returns:
    bool: True if it's valid to drop a piece in this column, False otherwise.
    """
    # TODO: implement
    if board[0][col] == 0:
        return True
    else:
        return False
    


def get_next_open_row(board, col):
    """
    Gets the next open row in the given column.

    Parameters:
    board (np.ndarray): The current board.
    col (int): The column index to search.

    Returns:
    int: The row index of the lowest empty cell in this column.
    """
    # TODO: implement
    rows, cols = board.shape
    for row in reversed(range(rows)):
        if board[row][col] == 0:
            return row
    return None

def winning_move(board, piece):
    """
    Checks if the board state is a winning state for the given piece.

    Parameters:
    board (np.ndarray): The current board.
    piece (int): The piece identifier (1 or 2).

    Returns:
    bool: True if 'piece' has a winning 4 in a row, False otherwise.
    This requires checking horizontally, vertically, and diagonally.
    """
    # TODO: implement
    rows, cols = board.shape
    pieces = []
    for row in range(rows):
        for col in range(cols):
            if board[row][col] == piece:
                pieces.append((row,col))
    for tup in pieces:
        if (tup[0],tup[1]+1) in pieces and (tup[0],tup[1]+2) in pieces and (tup[0],tup[1]+3) in pieces:
            return True
        if (tup[0]+1,tup[1]) in pieces and (tup[0]+2,tup[1]) in pieces and (tup[0]+3,tup[1]) in pieces:
            return True
        if (tup[0]-1,tup[1]+1) in pieces and (tup[0]-2,tup[1]+2) in pieces and (tup[0]-3,tup[1]+3) in pieces:
            return True
        if (tup[0]-1,tup[1]-1) in pieces and (tup[0]-2,tup[1]-2) in pieces and (tup[0]-3,tup[1]-3) in pieces:
            return True
    return False


def get_valid_locations(board):
    """
    Returns a list of columns that are valid to drop a piece.

    Parameters:
    board (np.ndarray): The current board.

    Returns:
    list of int: The list of column indices that are not full.
    """
    # TODO: implement
    valid_loc = []
    rows, cols = board.shape
    for col in range(cols):
        if get_next_open_row(board, col) is not None:
            valid_loc.append(col)
    return col

def is_terminal_node(board):
    """
    Checks if the board is in a terminal state:
      - Either player has won
      - Or no valid moves remain

    Parameters:
    board (np.ndarray): The current board.

    Returns:
    bool: True if the game is over, False otherwise.
    """
    # TODO: implement
    if winning_move(board,1) or winning_move(board,2):
        return True
    if len(get_valid_locations(board)) == 0:
        return True
    return False

def score_position(board, piece):
    """
    Evaluates the board for the given piece.
    (Already implemented to highlight center-column preference.)

    Parameters:
    board (np.ndarray): The current board.
    piece (int): The piece identifier (1 or 2).

    Returns:
    int: Score favoring the center column. 
         (This can be extended with more advanced heuristics.)
    """
    # This is already done for you; no need to modify
    # The heuristic here scores the center column higher, which means
    # it prefers to play in the center column.
    score = 0
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    center_count = center_array.count(piece)
    score += center_count * 3
    return score


def minimax(board, depth, alpha, beta, maximizingPlayer):
    """
    Performs minimax with alpha-beta pruning to choose the best column.

    Parameters:
    board (np.ndarray): The current board.
    depth (int): Depth to which the minimax tree should be explored.
    alpha (float): Alpha for alpha-beta pruning.
    beta (float): Beta for alpha-beta pruning.
    maximizingPlayer (bool): Whether it's the maximizing player's turn.

    Returns:
    tuple:
        - (column (int or None), score (float)):
          column: The chosen column index (None if no moves).
          score: The heuristic score of the board state.
    """
    # TODO: implement
    
    def minimax_rec(board, depth, alpha, beta, maximizingPlayer, current_depth):
        

        ROWS, COLS = board.shape
        valid_columns = [c for c in range(COLS) if board[0, c] == 0]  # Available moves

        # Base case: Reached max depth or no valid moves
        if current_depth == depth or len(valid_columns) == 0:
            return None, score_position(board, 1 if maximizingPlayer else 2)

        if maximizingPlayer:
            max_score = -np.inf
            best_column = None

            for col in valid_columns:
                row = get_next_open_row(board, col)
                temp_board = board.copy()
                temp_board[row, col] = 1  # Assume maximizing player is Player 1

                _, score = minimax_rec(temp_board, depth, alpha, beta, False, current_depth + 1)

                if score > max_score:
                    max_score = score
                    best_column = col

                alpha = max(alpha, score)
                if beta <= alpha:  # Alpha-beta pruning
                    break

            return best_column, max_score

        else:  # Minimizing player
            min_score = np.inf
            best_column = None

            for col in valid_columns:
                row = get_next_open_row(board, col)
                temp_board = board.copy()
                temp_board[row, col] = 2  # Assume minimizing player is Player 2

                _, score = minimax_rec(temp_board, depth, alpha, beta, True, current_depth + 1)

                if score < min_score:
                    min_score = score
                    best_column = col

                beta = min(beta, score)
                if beta <= alpha:  # Alpha-beta pruning
                    break

            return best_column, min_score 
    return minimax_rec(board, depth, alpha, beta, maximizingPlayer, 0)


if __name__ == "__main__":
    # Simple debug scenario
    # Example usage: create a board, drop some pieces, then call minimax
    example_board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    print("Debug: Created an empty Connect Four board.\n")
    print(example_board)

    # TODO: Students can test their own logic here once implemented, e.g.:
    # drop_piece(example_board, some_row, some_col, 1)
    # col, score = minimax(example_board, depth=4, alpha=-math.inf, beta=math.inf, maximizingPlayer=True)
    # print("Chosen column:", col, "Score:", score)
