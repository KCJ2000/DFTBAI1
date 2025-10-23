import numpy as np

def gaussian_elimination(matrix):
    """
    Perform Gaussian Elimination on a matrix to convert it to row-echelon form.
    
    Parameters:
    matrix (np.ndarray): A 2D numpy array representing the matrix to be transformed.
    
    Returns:
    np.ndarray: The matrix in row-echelon form.
    """
    # Get the number of rows and columns
    rows, cols = matrix.shape
    
    # Start with the first row
    current_row = 0
    
    for col in range(cols):
        # Find the row with the largest absolute value in the current column
        pivot_row = np.argmax(np.abs(matrix[current_row:, col])) + current_row
        
        # Swap the current row with the pivot row
        matrix[[current_row, pivot_row]] = matrix[[pivot_row, current_row]]
        
        # Make the pivot element 1
        pivot = matrix[current_row, col]
        if pivot != 0:
            matrix[current_row] = matrix[current_row] / pivot
        
        # Eliminate the current column below the pivot
        for row in range(current_row + 1, rows):
            factor = matrix[row, col]
            matrix[row] = matrix[row] - factor * matrix[current_row]
        
        # Move to the next row
        current_row += 1
        
        # If we have reached the last row, stop
        if current_row >= rows:
            break
    
    return matrix

# Example usage
if __name__ == "__main__":
    # Define a matrix
    matrix = np.array([
        [2, 1, -1],
        [-3, -1, 2],
        [-2, -1+1e-15, 2]
    ], dtype=float)
    
    print("Original matrix:")
    print(matrix)
    
    # Perform Gaussian Elimination
    row_echelon_matrix = gaussian_elimination(matrix)
    
    print("\nMatrix in row-echelon form:")
    print(row_echelon_matrix)