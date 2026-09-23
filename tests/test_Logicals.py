from bbGym import logicals
import numpy as np
from bbGym.polynomialCodes import codes, A1_HX, A1_HZ, A2_HX, A2_HZ, A3_HX, A3_HZ, A4_HX, A4_HZ, A5_HX, A5_HZ,  A6_HX, A6_HZ

def test_computeLogicals_A1():
    H_X = A1_HX.astype(np.int32)
    H_Z = A1_HZ.astype(np.int32)
    L_X, L_Z = logicals.computeLogicals(H_X, H_Z)


if __name__ == "__main__":
    test_computeLogicals_A1()