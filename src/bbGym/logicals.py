"""
Compute the logical operators of a CSS code given its stabilizer generators.
Thanks to Dr. Pavel Panteleev for clarifying comments on computing logicals and checking the residual error 

"""
import bbGym.funWithMatrices as funWithMatrices
import copy
import numpy as np


def computeLogicals(stabilizerGeneratorsX, stabilizerGeneratorsZ):
    """
    Given the stabilizer generators of a CSS code, compute its logical operators.

    Input:
        stabilizerGeneratorsX (np.ndarray): The X stabilizer generators - not assumed to be in reduced form.
        stabilizerGeneratorsZ (np.ndarray): The Z stabilizer generators - not assumed to be in reduced form.
    Output:
        logicalOperatorsX (np.ndarray): The X logical operators - in reduced form, but could contain a component from stabilizerGeneratorsX.
        logicalOperatorsZ (np.ndarray): The Z logical operators - in reduced form, but could contain a component from stabilizerGeneratorsZ.
    
    We first reduce all input matrices to their row echelon form.

    For a CSS code with parity-check matrices $H_X$ and $H_Z$ (over $F_2$), corresponding to stabilizerGeneratorsX and stabilizerGeneratorsZ we first find logical $Z$ operators. 

    The $Z$ logical operators are those $Z$ operators that commute with all $X$ stabilizers (i.e., they are in the normalizer of S), but are not in the span of the $Z$ stabilizers, i.e. in S.
    
    We first look for all $Z$ operators that commute with all $X$ stabilizers by solving the equation $H_X z = 0$ over $F_2$ and obtaining a set of basis vectors for this space.
    
    Some of them are $Z$-stabilizers, some are logical $Z$ operators. (X stabilizers, X operators)
    
    To check if a logical $Z$ operator, v, is in the span of the matrix $H_Z$ == stabilizerGeneratorsZ, 
    we test whether the rank of stabilizerGeneratorsZ is strictly smaller than that of np.vstack((stabilizerGeneratorsX, v)). 
    If yes, it means that v is in the row space of stabilizerGeneratorsX.
    Any v that survived this test is a logical $Z$ operator. 
    """
    
    # Step 1: Switch to reduced form: the stabilizer generators
    stabilizerGeneratorsXReduced, stabilizerGeneratorsXInverse, stabilizerGeneratorsXrank = funWithMatrices.binaryGaussianEliminationOnRows(copy.copy(stabilizerGeneratorsX))
    stabilizerGeneratorsZReduced, stabilizerGeneratorsZInverse, stabilizerGeneratorsZrank = funWithMatrices.binaryGaussianEliminationOnRows(copy.copy(stabilizerGeneratorsZ))

    # Step 2: Find a basis to the null space of each (reduced) stabiliser matrix. The null space of the Z stabilisers are X operators, the null space of the X stabilizers are Z operators.
    logicalOperatorsX = funWithMatrices.solveHomogenicBinaryLinearSystem(stabilizerGeneratorsZReduced)
    logicalOperatorsZ = funWithMatrices.solveHomogenicBinaryLinearSystem(stabilizerGeneratorsXReduced)

    # Step 3: Pick only the operators that are linearly independant of the stabilisers - this time we are checking the X operators that are linearly independent of the X stabilizers, and Z operators independent of the Z stabilizers. 
    newLogicalOperatorsX = []
    for i in range(logicalOperatorsX.shape[0]):
        testMatrix = np.vstack((stabilizerGeneratorsXReduced, logicalOperatorsX[i, :]))
        _, _, testRank = funWithMatrices.binaryGaussianEliminationOnRows(copy.copy(testMatrix))
        if testRank > stabilizerGeneratorsXrank:
            # This row is in the span of the X stabilizers, remove it
            newLogicalOperatorsX.append(logicalOperatorsX[i,:])
    
    newLogicalOperatorsZ = []
    for i in range(logicalOperatorsZ.shape[0]):
        testMatrix = np.vstack((stabilizerGeneratorsZReduced, logicalOperatorsZ[i, :]))
        _, _, testRank = funWithMatrices.binaryGaussianEliminationOnRows(copy.copy(testMatrix))
        if testRank > stabilizerGeneratorsZrank:
            # This row is in the span of the stabilizers, remove it
            newLogicalOperatorsZ.append(logicalOperatorsZ[i, :])

    return np.array(newLogicalOperatorsX), np.array(newLogicalOperatorsZ)


def calculateCodeDimension(Hx, Hz):
    if Hx.shape[1] != Hz.shape[1]:
        print(f"Column dimensions of Hx and Hz should be the same, instead they are {Hx.shape[1]} and {Hz.shape[1]}.")
        raise ValueError
    else:
        _, _, rankHx = funWithMatrices.binaryGaussianEliminationOnRows(copy.copy(Hx))
        _, _, rankHz = funWithMatrices.binaryGaussianEliminationOnRows(copy.copy(Hz))
        return Hx.shape[1] - rankHx - rankHz