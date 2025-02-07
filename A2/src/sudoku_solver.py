def print_board(board):
    """
    Prints the Sudoku board in a grid format.
    0 indicates an empty cell.

    Parameters:
    board (list[list[int]]): A 9x9 Sudoku board where 0 represents an empty cell.

    Returns:
    None
    """
    for row_idx, row in enumerate(board):
        # Print a horizontal separator every 3 rows (for sub-grids)
        if row_idx % 3 == 0 and row_idx != 0:
            print("- - - - - - - - - - -")

        row_str = ""
        for col_idx, value in enumerate(row):
            # Print a vertical separator every 3 columns (for sub-grids)
            if col_idx % 3 == 0 and col_idx != 0:
                row_str += "| "

            if value == 0:
                row_str += ". "
            else:
                row_str += str(value) + " "
        print(row_str.strip())


def find_empty_cell(board):
    """
    Finds an empty cell (indicated by 0) in the Sudoku board.

    Parameters:
    board (list[list[int]]): A 9x9 Sudoku board where 0 represents an empty cell.

    Returns:
    tuple or None:
        - If there is an empty cell, returns (row_index, col_index).
        - If there are no empty cells, returns None.
    """
    # TODO: implement
    rows, cols =  board.shape
    for row in rows:
        for col in cols:
            if board[row][col] == 0:
                return (row,col)
    return None       


def is_valid(board, row, col, num):
    """
    Checks if placing 'num' at board[row][col] is valid under Sudoku rules:
      1) 'num' is not already in the same row
      2) 'num' is not already in the same column
      3) 'num' is not already in the 3x3 sub-box containing that cell

    Parameters:
    board (list[list[int]]): A 9x9 Sudoku board.
    row (int): Row index of the cell.
    col (int): Column index of the cell.
    num (int): The candidate number to place.

    Returns:
    bool: True if valid, False otherwise.
    """
    # TODO: implement
    if num in board[row]:
        return False

    for r in range(9):
        if board[r][col] == num:
            return False

    box_row_start = (row // 3) * 3  # Get the starting row index of the 3x3 grid
    box_col_start = (col // 3) * 3  # Get the starting column index of the 3x3 grid

    for r in range(box_row_start, box_row_start + 3):
        for c in range(box_col_start, box_col_start + 3):
            if board[r][c] == num:
                return False

    return True  
def solve_sudoku(board):
    """
    Solves the Sudoku puzzle in 'board' using backtracking.

    Parameters:
    board (list[list[int]]): A 9x9 Sudoku board where 0 indicates an empty cell.

    Returns:
    bool:
        - True if the puzzle is solved successfully.
        - False if the puzzle is unsolvable.
    """
    # TODO: implement
    empty_cell = find_empty_cell(board)
    
    if not empty_cell:  
        return True

    row, col = empty_cell

    for num in range(1, 10):  # Try numbers 1-9
        if is_valid(board, row, col, num):
            board[row][col] = num  # Place the number

            if solve_sudoku(board):  
                return True

            board[row][col] = 0  # Backtrack if the solution doesn't work

    return False  


def is_solved_correctly(board):
    """
    Checks that the board is fully and correctly solved:
    - Each row contains digits 1-9 exactly once
    - Each column contains digits 1-9 exactly once
    - Each 3x3 sub-box contains digits 1-9 exactly once

    Parameters:
    board (list[list[int]]): A 9x9 Sudoku board.

    Returns:
    bool: True if the board is correctly solved, False otherwise.
    """
    # TODO: implement
    def has_all_numbers(nums):
        """Checks if a list contains exactly the numbers 1-9 once."""
        return sorted(nums) == list(range(1, 10))

    # Check rows
    for row in board:
        if not has_all_numbers(row):
            return False

    # Check columns
    for col in range(9):
        column_values = [board[row][col] for row in range(9)]
        if not has_all_numbers(column_values):
            return False

    # Check 3x3 sub-boxes
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            box_values = [board[r][c] for r in range(box_row, box_row + 3) for c in range(box_col, box_col + 3)]
            if not has_all_numbers(box_values):
                return False

    return True  


if __name__ == "__main__":
    # Example usage / debugging:
    example_board = [
        [7, 8, 0, 4, 0, 0, 1, 2, 0],
        [6, 0, 0, 0, 7, 5, 0, 0, 9],
        [0, 0, 0, 6, 0, 1, 0, 7, 8],
        [0, 0, 7, 0, 4, 0, 2, 6, 0],
        [0, 0, 1, 0, 5, 0, 9, 3, 0],
        [9, 0, 4, 0, 6, 0, 0, 0, 5],
        [0, 7, 0, 3, 0, 0, 0, 1, 2],
        [1, 2, 0, 0, 0, 7, 4, 0, 0],
        [0, 4, 9, 2, 0, 6, 0, 0, 7],
    ]

    print("Debug: Original board:\n")
    print_board(example_board)
    # TODO: Students can call their solve_sudoku here once implemented and check if they got a correct solution.
