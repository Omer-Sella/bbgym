import numpy as np
# from numba.experimental import jitclass
# from numba import uint8 int8, int16, int32, int64, uint16, uint32, uint64, jit

"""
A class for arithmetic of GF(4) elements.
I --> 0 --> 0
X --> 1 --> 1
Z --> 2 --> omega
Y --> 3 --> (omega +1) = bar{omega} == the conjugate of omega in GF(4)

"""

# specForGf4 = [
#     ('value', uint8)
# ]

#@jitclass(specForGf4)
class gf4():
    def __init__(self, value):        
        if value < 0 or value > 3:
            raise ValueError("Value must be between 0 and 3 inclusive.")
        self.value = value
    
    def lsb(self):
        return self.value % 2
    
    def msb(self):
        return (self.value // 2)

    def __add__(self, other):
        if not isinstance(other, gf4):
            raise TypeError("Can only add two gf4 objects.")
        msb = (self.msb() + other.msb()) % 2
        lsb = (self.lsb() + other.lsb()) % 2
        return gf4(msb * 2 + lsb)
    
    def __mul__(self, other):
        if not isinstance(other, gf4):
            raise TypeError("Can only multiply two gf4 objects.")
        # \omega^2 = \omega + 1
        
        [msb, lsb] = [1,1] if self.msb() * other.msb() == 1 else [0,0]
        msb = (msb + self.lsb() * other.msb() + self.msb() * other.lsb() ) % 2
        lsb = (lsb + self.lsb() * other.lsb() )% 2
        return gf4(msb * 2 + lsb)
    
    def bar(self):
        return gf4(self.value) if self.msb() == 0 else gf4((self.msb() * 2 + (self.lsb() + 1) % 2))
         
    def tr(self, other = None):
        if other is not None:
            if not isinstance(other, gf4):
                raise TypeError("Trace can only be computed with another gf4 object.")
            return (self * other.bar()).tr()
        return (self + (self * self))

    def printValue(self):
        if self.value == 0:
            return "0"
        elif self.value == 1:
            return "1"
        elif self.value == 2:
            return "ω"
        elif self.value == 3:
            return "(ω + 1)"
        else:
            return f"Invalid gf4 element with value {self.value}"
    # def __repr__(self):
    #     if self.value == 0:
    #         return "0"
    #     elif self.value == 1:
    #         return "1"
    #     elif self.value == 2:
    #         return "ω"
    #     elif self.value == 3:
    #         return "(ω + 1)"
    #     else:
    #         return f"Invalid gf4 element with value {self.value}"
        
    def __eq__(self, other):
        if not isinstance(other, gf4):
            return False
        return self.value == other.value
    
    
# @jit(nopython=True)
def mul(a, b):  
    if not all(isinstance(a[i], gf4) for i in range(len(a))) or not all(isinstance(b[i], gf4) for i in range(len(b))):
        raise TypeError("Can only multiply two arrays of gf4 objects.")
    if len(a) != len(b):
        raise ValueError("Arrays must be of the same length.")
    result = gf4(0)
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


# @jit(nopython=True)
def add(a, b):
    if not all(isinstance(a[i], gf4) for i in range(len(a))) or not all(isinstance(b[i], gf4) for i in range(len(b))):
        raise TypeError("Can only add two arrays of gf4 objects.")
    if len(a) != len(b):
        raise ValueError("Arrays must be of the same length.")
    result = []
    for i in range(len(a)):
        result.append(a[i] + b[i])
    return result


# @jit(nopython=True)
def traceInner(a, b):
    # if not all(isinstance(a[i], gf4) for i in range(len(a))) or not all(isinstance(b[i], gf4) for i in range(len(b))):
    #    raise TypeError("Can only compute inner product of two arrays of gf4 objects.")
    if len(a) != len(b):
        raise ValueError("Arrays must be of the same length.")
    result = gf4(0)
    for i in range(len(a)):
        result += a[i] * (b[i].bar())
    return result.tr()


# @jit(nopython=True)
def toarray(a):
    # if not all(isinstance(a[i], gf4) for i in range(len(a))):
    #    raise TypeError("Can only convert an array of gf4 objects to a numpy array.")
    result = np.zeros(len(a))#, dtype=object)
    for i in range(len(a)):
        result[i] = a[i].value
    return result

    
# @jit(nopython=True)
def toTuple(a):
    if not all(isinstance(a[i], gf4) for i in range(len(a))):
        raise TypeError("Can only convert an array of gf4 objects to a tuple.")
    
    arr = toarray(a)
    return tuple(map(tuple, arr))


# @jit(nopython=True)
def gf4CartesianProduct(n):
    print(f"Generating Cartesian product of {n} copies of GF(4)")
    if n == 0:
        raise ValueError("n must be a positive integer.")
    
    result = [[gf4(i)] for i in range(4)]
    
    for _ in range(n - 1):
        newResult = []
        for element in result:
            for i in range(4):
                newResult.append(element + [gf4(i)])
        result = newResult
    
    return result


def binaryDualToGf4Matrix(Hx, Hz):
    """
    Create an equivalent GF(4) matrix from two binary matrices corresponding to the X component and Z component.
    
    Arguments:
    Hx - a binary matrix or list of lists (no safety)
    Hz - a binary matrix or list of lists (no safety)

    Returns:
    An array of same dimension as [[Hx, 0], [0, Hz]], with elements from gf4 made by the homomorphism: 1*b_x + omega*bZ == gf4(1*b_x + 2*b_z)
    """
    integerMatrix = np.vstack((Hx, 2 * Hz))
    gf4Matrix = np.zeros(integerMatrix.shape, dtype = object)
    for i in range(integerMatrix.shape[0]):
        for j in range(integerMatrix.shape[1]):
            gf4Matrix[i,j] = gf4(integerMatrix[i,j])
    
    return gf4Matrix, integerMatrix


def integerToDualBinary(arr):
    """
    Convert an array of integers in {0,1,2,3} to two binary arrays corresponding to the X (lsb) component and Z component (msb).
    """
    return arr % 2, (arr // 2) % 2


def binaryDualToInteger(arrX, arrZ):
    """
    Convert two binary arrays corresponding to the X (lsb) component and Z component (msb) to an array of integers in {0,1,2,3} using the homomorphism: 1*b_x + omega*bZ == gf4(1*b_x + 2*b_z)
    """
    return arrX + 2*arrZ


integerTraceProduct = [[0, 0, 0, 0], # I commutes with everything or tr(0,0) = 0
                       [0, 0, 1, 1,], # X commutes with itself and I and anticommutes with Z and Y
                       [0, 1, 0, 1], # Z commutes with itself and I
                       [0, 1, 1, 0]] # Y commutes with itself and I

integerSum = [[0, 1, 2, 3],
              [1, 0, 3, 2],
              [2, 3, 0, 1],
              [3, 2, 1, 0]]

if __name__ == "__main__":
    # for i in gf4CartesianProduct(4):
    #    print(i)
    # testArray = np.array([0,1,1,1,2,2,2,3,3,3,0,0,0,0,0,0])
    # print(integerToDualBinary(testArray))
    pass
    
