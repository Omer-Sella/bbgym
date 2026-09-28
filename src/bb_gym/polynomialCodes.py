"""
This file hardcodes some quantum codes from the paper "Degenerate Quantum LDPC Codes With Good Finite Length
Performance" by Pavel Panteleev and Gleb Kalachev
As well as "Bivariate Bicycle codes from High-threshold and low-overhead fault-tolerant quantum memory"By Bravi et al

Other than providing the reference (published) BB codes, it also provides three functions to generate the parity matrices from the polynomials that define them:
A,B matrices are generated from polynomials (and two constant matrices)
H_X, H_Z are generated from A,B


For polynomial arithmetic, I used the Reed Solomon code to verify the polynomials given in the paper, but I left it commented 
https://github.com/Omer-Sella/reedSolomon
There is no real need for it if you're just using the matrices.
Then set an environment variable REEDSOLOMON to the root directory of the project.
"""
from scipy.linalg import circulant
import numpy as np

codes = {}
bbCodes = {}
"""
Generalized bicycle (GB) codes. The matrices
A and B have form A = (a(x)), B = (b(x)), so
here we specify the polynomials a(x), b(x), and the
circulant size l.
"""
# A1 [[254, 28, d]] code (l = 127), 14 <=d<= 20.
# a(x) = 1 + x15 + x20 + x28 + x66,
# b(x) = 1 + x58 + x59 + x100 + x121.
a_254_28 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
b_254_28 = [1] + [0] * 57 + [1, 1] + [0] * 40 + [1] + [0] * 20 + [1]
a_254_28_padded = a_254_28 + [0] * (127 - len(a_254_28))
A1_A = circulant(a_254_28_padded)
b_254_28_padded = b_254_28 + [0] * (127 - len(b_254_28))
A1_B = circulant(b_254_28_padded)
A1_HX = np.hstack((A1_A, A1_B))
A1_HZ = np.hstack((A1_B.transpose(), A1_A.transpose()))
codes["A1_HX"] = A1_HX
codes["A1_HZ"] = A1_HZ
a_254_28.reverse()
b_254_28.reverse() # BUG: Why is there another reverse here ?


a_126_28_8 = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1]
a_126_28_8_padded = a_126_28_8 + [0] * (63 - len(a_126_28_8))
A2_A = circulant(a_126_28_8_padded).transpose()
#b(x) = 1 + x3 + x13 + x20 + x42.
b_126_28_8 = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
b_126_28_8_padded = b_126_28_8 + [0] * (63 - len(b_126_28_8))
A2_B = circulant(b_126_28_8_padded).transpose()
A2_HX = np.hstack((A2_A, A2_B))
A2_HZ = np.hstack((A2_B.transpose(), A2_A.transpose()))
codes["A2_HX"] = A2_HX
codes["A2_HZ"] = A2_HZ
# A3) [[48, 6, 8]] code (l = 24).
# a(x) = 1 + x2 + x8 + x15,
a_48_6_8 = [1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1]
a_48_6_8_padded = a_48_6_8 + [0] * (24 - len(a_48_6_8))
A3_A = circulant(a_48_6_8_padded).transpose()
# b(x) = 1 + x2 + x12 + x17.
b_48_6_8 = [1, 0, 1] + [0] * 9 + [1] + [0]*4 + [1]
b_48_6_8_padded = b_48_6_8 + [0] * (24 - len(b_48_6_8))
A3_B = circulant(b_48_6_8_padded).transpose()

A3_HX = np.hstack((A3_A, A3_B))
A3_HZ = np.hstack((A3_B.transpose(), A3_A.transpose()))
codes["A3_HX"] = A3_HX
codes["A4_HZ"] = A3_HZ
# A4) [[46, 2, 9]] code (l = 23).
# a(x) = 1 + x5 + x8 + x12,
a_46_2_9 = [1,0,0,0,0,1,0,0,1,0,0,0,1]
a_46_2_9_padded = a_46_2_9 + [0]*(23 - len(a_46_2_9))
A4_A = circulant(a_46_2_9_padded).transpose()
# b(x) = 1 + x + x5 + x7
b_46_2_9 = [1,1,0,0,0,1,0,1]
b_46_2_9_padded = b_46_2_9 + [0]*(23 - len(b_46_2_9))
A4_B = circulant(b_46_2_9_padded).transpose()

A4_HX = np.hstack((A4_A, A4_B))
A4_HZ = np.hstack((A4_B.transpose(), A4_A.transpose()))
codes["A4_HX"] = A4_HX
codes["A4_HZ"] = A4_HZ
b_46_2_9.reverse()
a_46_2_9.reverse()

# A5) [[180, 10, d]] code (l = 90), 15 <= d <= 18.
# a(x) = 1 + x28 + x80 + x89,
a_180_10 = [1] + [0]*27 + [1] + [0]*51 + [1] + [0]*8 + [1]
a_180_10_padded = a_180_10 + [0]*(90 - len(a_180_10))
A5_A = circulant(a_180_10_padded).transpose()
# b(x) = 1 + x2 + x21 + x25.
b_180_10 = [1,0,1] + [0]*18 + [1] + [0]*3 + [1]
b_180_10_padded = b_180_10 + [0]*(90 - len(b_180_10))
A5_B = circulant(b_180_10_padded).transpose()

A5_HX = np.hstack((A5_A, A5_B))
A5_HZ = np.hstack((A5_B.transpose(), A5_A.transpose()))
codes["A5_HX"] = A5_HX
codes["A5_HZ"] =  A5_HZ

#A6 [[900, 50, 15]] code (l = 450).
# a(x) = 1 + x97 + x372 + x425,
a_900_50_15 = [1] + [0]*96 + [1] + [0]*274 + [1] + [0]*52 + [1]
a_900_50_15_padded = a_900_50_15 + [0]*(450 - len(a_900_50_15))
A6_A = circulant(a_900_50_15_padded).transpose()
# b(x) = 1 + x50 + x265 + x390.
b_900_50_15 = [1] + [0]*49 + [1] + [0]*214 + [1] + [0]*124 + [1]
b_900_50_15_padded = b_900_50_15 + [0]*(450 - len(b_900_50_15))  
A6_B = circulant(b_900_50_15_padded).transpose()

A6_HX = np.hstack((A6_A, A6_B))
A6_HZ = np.hstack((A6_B.transpose(), A6_A.transpose()))
codes["A6_HX"] = A6_HX
codes["A6_HZ"] = A6_HZ

"""
Bivariate Bicycle codes from High-threshold and low-overhead fault-tolerant quantum memory

4 BivariateBicyclequantumLDPCcodes
LetIℓ andSℓbethe identitymatrixandthecyclicshiftmatrixof sizeℓ×ℓrespectively. 
Thei-throwofSℓhasa
singlenonzeroentryequal tooneatthecolumni+1 (modℓ).Forexample,
S2= 0 1
1 0 and S3=


0 1 0
0 0 1
1 0 0

.
Considermatrices
x=Sℓ⊗Im and y=Iℓ⊗Sm.
Notethatxy=yxandxℓ=ym=Iℓm.ABBcodeisdefinedbyapairofmatrices
A=A1+A2+A3 and B=B1+B2+B3 (1)
whereeachmatrixAiandBj isapowerofxory.Hereandbelowtheadditionandmultiplicationofbinarymatrices
isperformedmodulotwo,unlessstatedotherwise. Thus,wealsoassumetheAi aredistinctandtheBj aredistinct
toavoidcancellationof terms. Forexample,onecouldchooseA=x3+y+y2andB=y3+x+x2. NotethatA
andBhaveexactlythreenon-zeroentries ineachrowandeachcolumn. Furthermore,AB=BAsincexy=yx.The
abovedatadefinesaBBLDPCcodedenotedQC(A,B)withlengthn=2ℓmandcheckmatrices
HX=[A|B] and HZ=BT|AT 

[[n,k,d]] NetEncoding
Rater ℓ,m A B
[[72,12,6]] 1/12 6,6 x3+y+y2 y3+x+x2
[[90,8,10]] 1/23 15,3 x9+y+y2 1+x2+x7
[[108,8,10]] 1/27 9,6 x3+y+y2 y3+x+x2
[[144,12,12]] 1/24 12,6 x3+y+y2 y3+x+x2
[[288,12,18]] 1/48 12,12 x3+y2+y7 y3+x+x2
[[360,12,≤24]] 1/60 30,6 x9+y+y2 y3+x25+x26
[[756,16,≤34]] 1/95 21,18 x3+y10+y17 y5+x3+x19
"""


def generateABmatrices(l, m, aX, aY, bX, bY):
     s_l = np.roll(np.eye(l), 1, axis = 1)
     s_m = np.roll(np.eye(m), 1, axis = 1)
     I_l = np.eye(l)
     I_m = np.eye(m)
     x = np.kron(s_l, I_m)
     y = np.kron(I_l, s_m)
     A = np.zeros(x.shape)
     B = np.zeros(y.shape)
     for p in aX:
          A = (A + (np.linalg.matrix_power(x, p) % 2)) % 2
     for p in aY:
          A = (A + (np.linalg.matrix_power(y, p) % 2)) % 2
     for p in bX:
          B = (B + (np.linalg.matrix_power(x, p) % 2)) % 2
     for p in bY:
          B = (B + (np.linalg.matrix_power(y, p) % 2)) % 2
     return A, B


def bicycleCodeFromAB(A, B):
     H_X = np.hstack((A, B))
     H_Z = np.hstack((B.transpose(), A.transpose()))
     return H_X.astype(int), H_Z.astype(int)

def generateBicycleCode(l, m, aX, aY, bX, bY):
     A, B = generateABmatrices(l, m, aX, aY, bX, bY)
     Hx, Hz = bicycleCodeFromAB(A, B)
     return Hx, Hz
     
     
aX_72_12_6 = [3]
aY_72_12_6 = [1, 2]
bX_72_12_6 = [1, 2]
bY_72_12_6 = [3]
H_x_72_12_6, H_z_72_12_6 = generateBicycleCode(6, 6, aX_72_12_6, aY_72_12_6, bX_72_12_6, bY_72_12_6)
codes["Hx_72_12_6"] = H_x_72_12_6
codes["Hz_72_12_6"] = H_z_72_12_6
bbCodes["72_12_6"] = [H_x_72_12_6, H_z_72_12_6]

aX_90_8_10 = [9]
aY_90_8_10 = [1, 2]
bX_90_8_10 = [0, 2, 7]
bY_90_8_10 = []
Hx_90_8_10, Hz_90_8_10 = generateBicycleCode(15, 3, aX_90_8_10, aY_90_8_10, bX_90_8_10, bY_90_8_10)
codes["Hx_90_8_10"] = Hx_90_8_10
codes["Hz_90_8_10"] = Hz_90_8_10
bbCodes["90_8_10"] = [Hx_90_8_10, Hz_90_8_10]

aX_108_8_10 = [3]
aY_108_8_10 = [1, 2]
bX_108_8_10 = [1, 2]
bY_108_8_10 = [3]
HX_108_8_10, HZ_108_8_10 = generateBicycleCode(9, 6, aX_108_8_10, aY_108_8_10, bX_108_8_10, bY_108_8_10)
codes["Hx_108_8_10"] = HX_108_8_10
codes["Hz_108_8_10"] = HZ_108_8_10
bbCodes["108_8_10"] = [HX_108_8_10, HZ_108_8_10]

aX_144_12_12 = [3]
aY_144_12_12 = [1, 2]
bX_144_12_12 = [1, 2]
bY_144_12_12 = [3]
Hx_144_12_12, Hz_144_12_12 = generateBicycleCode(12, 6, aX_144_12_12, aY_144_12_12, bX_144_12_12, bY_144_12_12)
codes["Hx_144_12_12"] = Hx_144_12_12
codes["Hz_144_12_12"] = Hz_144_12_12
bbCodes["144_12_12"] = [Hx_144_12_12, Hz_144_12_12]

aX_288_12_18 = [3]
aY_288_12_18 = [2, 7]
bX_288_12_18 = [1, 2]
bY_288_12_18 = [3]
Hx_288_12_18, Hz_288_12_18 = generateBicycleCode(12, 12, aX_288_12_18, aY_288_12_18, bX_288_12_18, bY_288_12_18)
codes["Hx_288_12_18"] = Hx_288_12_18
codes["Hz_288_12_18"] = Hz_288_12_18
bbCodes["288_12_18"] = [Hx_288_12_18, Hz_288_12_18]

aX_360_12_24 = [9]
aY_360_12_24 = [1, 2]
bX_360_12_24 = [25, 26]
bY_360_12_24 = [3]
Hx_360_12_24, Hz_360_12_24 = generateBicycleCode(30, 6, aX_360_12_24, aY_360_12_24, bX_360_12_24, bY_360_12_24)
codes["Hx_360_12_24"] = Hx_360_12_24
codes["Hz_360_12_24"] = Hz_360_12_24
bbCodes["360_12_24"] = [Hx_360_12_24, Hz_360_12_24]


aX_756_16_34 = [3]
aY_756_16_34 = [10, 17] 
bX_756_16_34 = [3, 19]
bY_756_16_34 = [5]
Hx_756_16_34, Hz_756_16_34 = generateBicycleCode(21, 18, aX_756_16_34, aY_756_16_34, bX_756_16_34, bY_756_16_34)
codes["Hx_756_16_34"] = Hx_756_16_34
codes["Hz_756_16_34"] = Hz_756_16_34
bbCodes["756_16_34"] = [Hx_756_16_34, Hz_756_16_34]