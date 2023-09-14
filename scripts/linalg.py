def forward_elimination(matrix, vector, n):
    """
    Performs forward elimination to transform matrix to upper triangular form.
    """
    for row in range(n):
        # print(f"Matrix before iteration {row}:")
        # for r in matrix:
        #     print(r)
        # print(f"Vector before iteration {row}:")
        # for s in vector:
        #     print(s)

        max_row = row
        for i in range(row + 1, n):
            if abs(matrix[i][row]) > abs(matrix[max_row][row]):
                max_row = i
        matrix[row], matrix[max_row] = matrix[max_row], matrix[row]
        vector[row], vector[max_row] = vector[max_row], vector[row]
        
        # print(f"Matrix mid iteration {row}:")
        # for r in matrix:
        #     print(r)
        # print(f"Vector mid iteration {row}:")
        # for s in vector:
        #     print(s)

        # Check for singularity
        if matrix[row][row] == 0:
            raise ValueError("Matrix is singular.")

        for i in range(row + 1, n):
            factor = matrix[i][row] / matrix[row][row]
            for j in range(row, n):
                matrix[i][j] -= factor * matrix[row][j]
            vector[i] -= factor * vector[row]
        
        # print(f"Matrix after iteration {row}:")
        # for r in matrix:
        #     print(r)
        # print(f"Vector mid iteration {row}:")
        # for s in vector:
        #     print(s)
        # print("\n")

def back_substitution(matrix, vector, n):
    """
    Performs back substitution to solve the system for the upper triangular matrix.
    """
    x = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        x[i] = vector[i]
        for j in range(i + 1, n):
            x[i] -= matrix[i][j] * x[j]
        x[i] /= matrix[i][i]
    return x

def gaussian_elimination(matrix, vector):
    """
    Solves the system of linear equations using Gaussian elimination.
    """
    n = len(vector)
    forward_elimination(matrix, vector, n)
    return back_substitution(matrix, vector, n)

def linalg_solve(A, b):
    """
    Solves the system of equations Ax = b.
    """
    # Convert A and b to lists for easier processing
    matrix = [list(row) for row in A]
    vector = list(b)
    return gaussian_elimination(matrix, vector)

# Test
A = [
    [2, 1, -1],
    [-3, -1, 2],
    [-2, 1, 2]
]
b = [8, -11, -3]

forward_elimination(A, b, len(b))
back_substitution(A, b, len(b))
