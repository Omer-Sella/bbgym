"""
This module contains binary Gaussian elimination, which is also the basis for matrix inverse.
The name, "Fun with matrices" is in honour of Dr. James Wotton's brilliant video lecture
"""
import numpy as np
import copy
BINARY_DATA_TYPE = np.int32


def binaryGaussianEliminationOnRows(matrix, returnDtype = BINARY_DATA_TYPE):
    """
    Arguments:
    Input:
        matrix (np.ndarray): A binary matrix to perform Gaussian elimination on - no copy is made.
    Output:
        reducedMatrix (np.ndarray): The input matrix in reduced row echelon form.
        matrixInverse (np.ndarray): The pseudo inverse of the input matrix, obtained by performing the same row operations on the identity matrix.
        rank (int): The rank of the input matrix, which is the number of non-zero rows in the reduced row echelon form.
    Description:
    Perform Gaussian elimination on the rows of a binary matrix, to get it in reduced row echelon form.
    We also keep track of the row operations we perform, so we can return the inverse of the matrix as well.
    We also return the rank of the matrix, which is the number of non-zero rows in the reduced row echelon form.
    """
    
    matrixInverse = np.eye(matrix.shape[0], dtype = returnDtype)
    # special handling for the case where the input matrix is all zeros:
    rank = 0
    if np.all(matrix == 0):
        return matrix.astype(returnDtype), copy.copy(matrix).astype(returnDtype), rank
    else:
        for k in range(0, matrix.shape[1], 1): #OMer: changed from matrix.shape[0] to matrix.shape[1] - this should be the number of columns, not rows. I need to add tests for this.
            # Find the first row index i >= k, such that row i has non zero element at column k
            for i in range(rank, matrix.shape[0], 1):
                # If such a row is found, swap it with row rank, and break out of the loop
                if matrix[i, k] != 0:
                    temp = copy.copy(matrix[rank, :])
                    matrix[rank, :] = matrix[i, :]
                    matrix[i, :] = copy.copy(temp) 
                    
                    # Do the same for the pseudo inverse matrix
                    temp = copy.copy(matrixInverse[rank,:])
                    matrixInverse[rank,:] = matrixInverse[i,:]
                    matrixInverse[i, :] = copy.copy(temp)
                    
                    # Now XOR row rank with any rows, j,  that have matrix[j,k] != 0 (same for the pseudo inverse)
                    for j in range(0, matrix.shape[0], 1):
                        if j != rank:
                            if matrix[j, k] != 0:
                                matrix[j, :] = (matrix[j, :] + matrix[rank, :]) %2
                                matrixInverse[j, :] = (matrixInverse[j, :] + matrixInverse[rank, :]) %2
                    rank += 1 
                    break
    return matrix.astype(returnDtype), matrixInverse.astype(returnDtype), rank                                                                                                 


def solveHomogenicBinaryLinearSystem(matrixA):
    """
    Docstring for solveBinaryLinearSystem
    
    :param matrixA: Binary matrix A
    :return: a basis for the space of solutions to Ax = 0 over F(2)
    
    Then find a basis for the space of solutions to Ax = 0 over F(2)
    1. Given a binary matrix, A, perform Gaussian elimination on the rows of A to get A in reduced echcelon form.
    2. Identify the free variables, which are variables that the reduced echelon form doesn't say anything about them, namely x_rank .. x_n-1. So the dimension of the solution space is n - rank, where n is the number of columns of A, and rank is the rank of A.
    3. For each free variable, set it to 1 and the others to 0, then start at the last row of the reduced echelon form and work upwards, substituting in values for the free variables to get values for the leading variables.
    4. Each such assignment gives a basis vector for the solution space.
    5. Return a matrix which colums span the solution space.
    """
    
    
    reducedAugmentedMatrix, augmentedMatrix, rank = binaryGaussianEliminationOnRows(copy.copy(matrixA))
    
    solutions = []
    if rank == matrixA.shape[1]:
        # Full column rank, only the trivial solution exists, so we return an empty list, not the all 0 solution
        #solutions.append(np.zeros( (matrixA.shape[1]), dtype = BINARY_DATA_TYPE))
        pass
    elif rank == 0:
        # Zero column rank, the solution space is the entire space
        for i in range(matrixA.shape[1]):
            solution = np.zeros( (matrixA.shape[1]), dtype = BINARY_DATA_TYPE)
            solution[i] = 1
            solutions.append(solution)
    else:
        # So the rank is at least 1,and less than matrixA.shape[1]
        for freeVarIndex in range(rank, matrixA.shape[1]):
            # allocate space for the solution set all variables to 0
            solution = np.zeros( (matrixA.shape[1]), dtype = BINARY_DATA_TYPE)
            # Set the free variable to 1
            solution[freeVarIndex] = 1
            # Now work upwards from the last row to determine the values of the other variables
            for rowIndex in range(rank-1, -1, -1):
                # print(rowIndex)
                # Find the leading variable in this row
                leadingVarIndex = None
                for colIndex in range(matrixA.shape[1]):
                    if reducedAugmentedMatrix[rowIndex, colIndex] == 1:
                        leadingVarIndex = colIndex
                        break
                if leadingVarIndex is not None:
                    # Calculate the value of the leading variable
                    sumValue = 0
                    for colIndex in range(leadingVarIndex + 1, matrixA.shape[1]):
                        sumValue = ((reducedAugmentedMatrix[rowIndex, colIndex] * solution[colIndex]) + sumValue) %2
                    solution[leadingVarIndex] = sumValue
            solutions.append(solution)
    return np.array(solutions)

def binaryDeterminant(matrix, rowNumber = 0):    
    if not (matrix.shape[0] == matrix.shape[1]):
        raise ValueError("Determinant of matrices is only supported for square matrices.")
    
    if matrix.shape[0] == 2:
        determinantResult =  (matrix[0, 0] * matrix[1, 1]) + (matrix[0, 1] * matrix[1, 0]) % 2
    else:
        # Determinant according to row number 0
        determinantResult = False
        
        for col in range(matrix.shape[1]):
            cofactorMatrix = np.hstack( (np.vstack((matrix[:rowNumber, :col], matrix[rowNumber+1: ,:col])) , np.vstack( (matrix[:rowNumber, col+1:], matrix[rowNumber+1:, col+1:]))))
            determinantResult ^= ( matrix[rowNumber, col] & binaryDeterminant(cofactorMatrix))  # &(-1) ** (i+j) which equates to 1 over F(2)
    return determinantResult
    

if __name__ == "__main__":
    pass