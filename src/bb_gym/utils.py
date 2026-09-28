import numpy as np
from bb_gym.gf4 import integerToDualBinary #noqa
from bb_gym.logicals import computeLogicals #noqa
#from bbGym.minSum import ldpcDecoder
#from bbGym.memBP import decode
from scipy.integrate import trapezoid
import time

from bb_gym.polynomialCodes import A1_HX, A1_HZ
LDPC_INT_DATA_TYPE = np.int32
FLOAT_DATA_TYPE_UTILS = np.float32
GEOMETRIC5_ERROR_RANGE = np.geomspace(0.001, 0.1, 5)
CANONICAL_ERROR_RANGE = np.linspace(0.0001, 0.1, 5)
FIVE_POINT_GRID = np.linspace(1e-4, 1e-1, 5)
TEN_POINT_GRID = np.linspace(1e-4, 1e-1, 10)

LINEAR5 = np.linspace(0.0001, 0.1, 5)
GEOMETRIC5 = np.geomspace(0.001, 0.1, 5)
GEOMETRIC7 = np.geomspace(0.001, 0.1, 7)
GEOMETRIC10 = np.geomspace(0.0001, 0.1, 10)
UNION_ERROR_RANGE = np.unique(np.round(np.concatenate([
            np.linspace(0.0001, 0.1, 5),
            np.geomspace(0.001, 0.1, 5),
            np.geomspace(0.001, 0.1, 7),
            np.geomspace(0.0001, 0.1, 10)]), 10))

NAMED_ERROR_RANGES = {"linear5": CANONICAL_ERROR_RANGE, 
                      "geometric5": GEOMETRIC5_ERROR_RANGE, 
                      "dataCollection": UNION_ERROR_RANGE, 
                      "geometric7": GEOMETRIC7, 
                      "geometric10": GEOMETRIC10, 
                      "union" : UNION_ERROR_RANGE}


# baseline codes from the paperreward, not normalized to error range width, for errorRange = np.linspace(10**-4, 10**-1, 5) #Omer: I fixed this from 10 points to 5 points on 24/07/2026
baselines = {(6, 6): {"name": "[[72,12,6]]", "reward" : 0.035964, "number of logical qubits": 12, "reward_geometric5": 0.3460, "reward_linear5" : 0.3600}, # 0.033189,
             (9, 6): {"name": "[[108,8,10]]", "reward": 0.0374625, "number of logical qubits": 8, "reward_geometric5": 0.5455, "reward_linear5" : 0.375}, # 0.040959,
             (15, 3): {"name": "[[90,8,10]]", "reward" : 0.035964, "number of logical qubits": 8, "reward_geometric5": 0.5063, "reward_linear5" : 0.3600}, # 0.04218,
             (12, 6): {"name": "[[144,12,12]]", "reward" : 0.037462499999999996, "number of logical qubits": 12, "reward_geometric5": 0.554, "reward_linear5" : 0.3749}, # 0.038739,
#             (12,12): {"reward":0.041958,"number of logical qubits": 12, "reward_geometric5": 0.60, "reward_linear5":0.385}, #TODO: add code name and double check reward value0.0414,
             (21, 18): {"name": "[[756,16,≤34]]", "reward" : 0.0384615, "number of logical qubits": 16, "reward_geometric5" : 0.60, "reward_linear5" : 0.385}, # TODO: double check reward value
             (3, 27): {"number of logical qubits": None, }, # TODO: add name and double check reward value
             (5, 15): {"number of logical qubits": None, }, # TODO: add name and double check reward value
             (30, 6): {"name": "[[360,12,≤24]]", "number of logical qubits": 12, },
             (12, 12): {"name": "[[288,12,18]]", "number of logical qubits": 12, }
             }
"""
Just a reminder:
[[n,k,d]] NetEncoding   ℓ,m A B
            Rate r 
[[72,12,6]] 1/12        6,6 x3+y+y2 y3+x+x2
[[90,8,10]] 1/23        15,3 x9+y+y2 1+x2+x7
[[108,8,10]] 1/27       9,6 x3+y+y2 y3+x+x2
[[144,12,12]] 1/24      12,6 x3+y+y2 y3+x+x2
[[288,12,18]] 1/48      12,12 x3+y2+y7 y3+x+x2
[[360,12,≤24]] 1/60     30,6 x9+y+y2 y3+x25+x26
[[756,16,≤34]] 1/95     21,18 x3+y10+y17 y5+x3+x19
"""


def minSumEvaluateCode(numberOfTransmissions, seed, errorRange, numberOfIterations, H):
    """
    parameters
    ----------
    numberOfTransmissions : int
        Number of transmissions to simulate at each error probability.
    seed : int
        Seed for the local PRNG.
    errorRange : list of float
        List of error probabilities to simulate.
    numberOfIterations : int
        Number of iterations for the decoder.
    H : ndarray
        Parity check matrix.
    Returns
    ------- 
    logical error count : ndarray
        Array of counted errors for each error probability in errorRange. NOTE THIS IS ERROR COUNT, NOT ERROR RATE !!!

    Notes
    -----
    This is not an evalaution as a quantum code, meaning - we check that the result of the decoder is the same as the original error.
    We do not check whether the residual error is a logical error or not. 
    Concurrent futures require the seed to be between 0 and 2**32 -1
    """
    codewordSize = H.shape[1]
    localPrng = np.random.RandomState(seed)
    decoder = ldpcDecoder(H, syndromeDecoding = True)
    start = 0
    end = 0
    codeword = np.zeros(codewordSize, dtype = LDPC_INT_DATA_TYPE)
    decodedWord = np.zeros(codewordSize, dtype = LDPC_INT_DATA_TYPE)
    berArray = np.zeros(len(errorRange))
    for i, p in zip(range(len(errorRange)), errorRange):
        timeTotal = 0
        print(f"Error prob {p}, corresponding to i=={i}")
        for k in range(numberOfTransmissions):
            error = localPrng.choice([0, 1], size=codewordSize, replace=True, p=[1 - p, p])
            errorModulated = np.where(error == 0, -1.0, 1.0)
            berUncoded = 0
            berDecoded = 0
            start = time.time()
            status, decodedWord, softVector, iterationStoppedAt = decoder.decoderMainLoop(errorModulated, numberOfIterations)
            end = time.time()
            timeTotal += (end - start)
            berDecoded = np.count_nonzero(decodedWord != codeword)
            berArray[i] += berDecoded
        print("Time it took the decoder:")
        print(timeTotal)
        print("And the throughput is:")
        numberOfBits = numberOfTransmissions * codewordSize
        print(numberOfBits / timeTotal)
    return berArray / (numberOfTransmissions * codewordSize)

def memBPEvaluateCode(numberOfTransmissions, seed, errorRange, numberOfIterations, H):
    """
    parameters
    ----------
    numberOfTransmissions : int
        Number of transmissions to simulate at each error probability.
    seed : int
        Seed for the local PRNG.
    errorRange : list of float
        List of error probabilities to simulate.
    numberOfIterations : int
        Number of iterations for the decoder.
    H : ndarray
        Parity check matrix.
    Returns
    ------- 
    berArray : ndarray
        Array of bit error rates for each error probability in errorRange.

    Notes
    -----
    This is not an evalaution as a quantum code, meaning - we check that the result of the decoder is the same as the original error.
    We do not check whether the residual error is a logical error or not. 
    Concurrent futures require the seed to be between 0 and 2**32 -1
    """
    codewordSize = H.shape[1]
    localPrng = np.random.RandomState(seed)
    start = 0
    end = 0
    codeword = np.zeros(codewordSize, dtype = np.int32)
    decodedWord = np.zeros(codewordSize, dtype = np.int32)
    berArray = np.zeros(len(errorRange))
    for i,p in zip(range(len(errorRange)), errorRange):
        timeTotal = 0    
        print(f"Error prob {p}, corresponding to i=={i}")
        for k in range(numberOfTransmissions):
            error = localPrng.choice([0,1], size=codewordSize, replace=True, p=[1 - p, p])
            start = time.time()
            errorVector, marginals, converged, iteration = decode(H, initMarginals = None, errorProbabilities= [p]*codewordSize, sigma = (H @ error %2), Gammas = None, maxIterations=numberOfIterations, logProbabilities = False)
            end = time.time()
            timeTotal += (end - start)
            berDecoded = np.count_nonzero(errorVector != error)
            berArray[i] += berDecoded
        print("Time it took the decoder:")
        print(timeTotal)
        print("And the throughput is:")
        numberOfBits = numberOfTransmissions * codewordSize
        print(numberOfBits / timeTotal)
    return berArray / (numberOfTransmissions * codewordSize)

def wrapperForRoffesLdpc(H, syndrome, initialValues, decoderStoppingCriterion, ms_scaling_factor = 0.625, osd_method = "osd0", osd_order = 0):
    """
    Check description here: https://github.com/quantumgizmos/bp_osd/blob/main/README.ipynb
    Remember that Roffe's decoder is really a classical binary decoder !
    """
    from ldpc import BpOsdDecoder #noqa
    p_error = np.average(initialValues[:,1]) # initialValues[i,1] is the probability of error for the ith coordinate
    bpDecoder=BpOsdDecoder(H,#the parity check matrix
        error_rate=p_error,
        channel_probs= initialValues[:,1], #assign error_rate to each qubit. This will override "error_rate" input variable
        max_iter=decoderStoppingCriterion, #the maximum number of iterations for BP)
        bp_method="ms",
        ms_scaling_factor=ms_scaling_factor, #min sum scaling factor. If set to zero the variable scaling factor method is used
        osd_method=osd_method, #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
        osd_order=osd_order #the osd search depth
        )
    result = bpDecoder.decode(syndrome)
    return result, True # TODO: What does Roffe's decoder return in the case of decoder failure ? (assuming it counts it as a logical error)

def decoderEvaluator(decoderFunction, dualBinary, Hx, Hz, errorRange, decoderStoppingCriterion, numberOfSamples, seed = None):
    """
    A single function to evaluate quantum decoders over the depolarizing channel. 
    What this means is, that the decoder spits out a correction, this is then added to the actual error, and tested for logical errors.
    
    Arguments:
    Hx, Hz: A pair of binary matrices. Use the codes in polynomialCodes.py
    errorRange: A range of probabilities
    decoderStoppingCriterion - usually intended as maximum number of iterations.

    Returns:
    logicalErrorRate: array of floats. For each error probability p, the number of times a logical error was created divided by the number of samples attempted for the error probability.
    decoderFailureRate: array of floats. For each error probability p, the number of times the decoder failed divided by the number of samples attempted for the error probability.
    """
    import numpy as np
    from bb_gym.gf4 import integerTraceProduct as tp
    from bb_gym.gf4 import integerToDualBinary, binaryDualToInteger
    from bb_gym.logicals import computeLogicals
    if seed is None:
        seed = 7134066
    localRandom = np.random.RandomState(seed)
    logicalX, logicalZ = computeLogicals(Hx, Hz)

    decoderFailureRate = np.zeros(len(errorRange))
    logicalErrorRate = np.zeros(len(errorRange))
    for i in range(len(errorRange)):
        p = errorRange[i]
        decoderFailureRate[i] = 0
        logicalErrorRate[i] = 0
        for _ in range(numberOfSamples):
            #Sample (some number of times) an error, which is a vector over {0,1,2,3} representing I,X,Z,Y (to be consistent check with the documentation in gf4.py)
            error = localRandom.choice([0,1,2,3], size=Hx.shape[1], replace=True, p=[1 - 3*p, p, p, p])
            errorX, errorZ = integerToDualBinary(error)
            #Calculate the syndrome for this error
            syndromeX, syndromeZ = Hx.dot(errorZ)%2, Hz.dot(errorX)%2
            #Run whatever decoder you decided to evaluate with arguments Hx, Hz, syndrome. The decoder has to accept a binary pair and syndrome, and return errorEstimation (an estimated solution), which is an array over 0,1,2,3 and a success flag (boolean).
            
            if dualBinary:
                
                initialValuesX = np.tile(np.array([1 - 2 * p, 2 * p]), (Hx.shape[1], 1))
                initialValuesZ = np.tile(np.array([1 - 2 * p, 2 * p]), (Hz.shape[1], 1))
                estimatedErrorX, estimatedErrorZ, success = decoderFunction(Hx, Hz, syndromeX, syndromeZ, initialValuesX, initialValuesZ, decoderStoppingCriterion)
            else:
                initialValues = np.tile(np.array([1 - 3 * p, p, p, p]), (Hx.shape[1], 1))
                syndrome = np.hstack((syndromeX, syndromeZ))
                H = np.vstack((Hx, 2 * Hz))
                estimatedError, success = decoderFunction(H, syndrome, initialValues, decoderStoppingCriterion)
                estimatedErrorX, estimatedErrorZ = integerToDualBinary(estimatedError)
            # If the decoder thinks it failed, add 1 to the decoderFailure counter. 
            if not success:
                decoderFailureRate[i] = decoderFailureRate[i] + 1
            else:
                # If the decoder thinks it succeeded (success == True), then test the residual error, i.e.: the (gf(4)) sum of the estimated error and the original error, to see if either: 
                # It does not amount to a 0 syndrome (this means the decoder is wrong, meaning - it thinks that the estimatedError + error is a stabilizer, and that it countered the effect of the error up to a stabilizer, but it didn't, and the result is not in the normalizer).
                # It is a logical error.
                # In either case we increase the decoderErrorRate by 1.
                residualErrorX = (estimatedErrorX + errorX) % 2
                residualErrorZ = (estimatedErrorZ + errorZ) % 2
                # Check whether the residual error gives 0 syndrome:
                if not (np.all((np.dot(Hx, residualErrorZ)) % 2 == 0) and np.all((np.dot(Hz, residualErrorX) % 2) == 0)):
                    print("Decoder failure: the residual error does not give 0 syndrome, meaning the decoder is wrong")
                    logicalErrorRate[i] += 1
                else: # So we are in the case that the residual error commutes with all stabilizers, i.e., it is in the normalizer. So let's check if it is a stabilizer (commutes with all logicals), or a logical error (anticommutes with some logical operator)
                    if not (np.all((np.dot(logicalZ,residualErrorX) % 2) == 0) and np.all((np.dot(logicalX, residualErrorZ) % 2) == 0)):
                        # print(f"Logical error: the residual error commutes with all stabilizers but anticommutes with some logical operator")
                        logicalErrorRate[i] += 1
    return logicalErrorRate, decoderFailureRate


def binaryDecoderToDualBinaryDecoderWrapper(binaryDecoderFunction):
    # This wrapper was meant to take a binary decoder, and transform it into a dual-binary decoder. 
    # There is no logic or calculation aside from the decoder.
    def dualDecoder(Hx, Hz, syndromeX, syndromeZ, initialValuesX, initialValuesZ, decoderStoppingCriterion):
        estimatedErrorZ, successZ = binaryDecoderFunction(Hx, syndromeX, initialValuesX,decoderStoppingCriterion)
        estimatedErrorX, successX = binaryDecoderFunction(Hz, syndromeZ, initialValuesZ,decoderStoppingCriterion)
        return estimatedErrorX, estimatedErrorZ, successX and successZ
    return dualDecoder



def calculateRewardFromSamples(logicalErrorCount, numberOfSamples, errorRange, l, m, rewardEngineering):
    logicalErrorRate = logicalErrorCount / numberOfSamples
    reward = trapezoid(1 - logicalErrorRate, errorRange)
    if rewardEngineering:
        reward = reward / np.abs(np.min(errorRange) - np.max(errorRange))
    return reward


if __name__ == "__main__":
    import numpy as np
    import json
    import time
    SAVE_PATH = os.environ.get("QECC", ".") # TODO: I need to get the project top rather than hard code it here.
    # numberOfTransmissions = 20
    # seed = 123456
    # errorRange = np.linspace(0.001, 0.1, 10)
    # numberOfIterations = 50
    # H = A1_HX.astype(np.int32)
    # minSumEvaluateCode(numberOfTransmissions, seed, errorRange, numberOfIterations, H)
    # memBPEvaluateCode(numberOfTransmissions, seed, errorRange, numberOfIterations, H)
    # from bbGym.utils import decoderEvaluator
    from bb_gym.polynomialCodes import A1_HX, A1_HZ, bbCodes
    # from qecc.minSum import ldpcDecoderWrapper
    # errorRange = np.linspace(10**-4, 10**-1, 10)
    
    errorRange = NAMED_ERROR_RANGES["geometric5"]

    NUMBER_OF_SAMPLES = 50
    NUMBER_OF_DECODER_ITERATIONS = 50
    errorRate = {}
    timeMeasured = {}
    dualRoffeDecoder = binaryDecoderToDualBinaryDecoderWrapper(wrapperForRoffesLdpc)
    
    for key, value in bbCodes.items():
        errorRate = {}
        # errorRateDualBinaryMinSum = {}
        timeMeasured = {}
        # value = bbCodes["72_12_6"]
        # key = "72_12_6"
        print(f"Characterizing code {key}")
        start = time.time()
        logicalER, decoderFailureRate = decoderEvaluator(decoderFunction = dualRoffeDecoder, dualBinary = True, Hx = value[0], Hz = value[1], errorRange = errorRange, decoderStoppingCriterion = NUMBER_OF_DECODER_ITERATIONS, numberOfSamples = NUMBER_OF_SAMPLES)
        # logicalER, decoderFailureRate = decoderEvaluator(decoderFunction = refinedBPalgorithm3, dualBinary = False, Hx = value[0], Hz = value[1], errorRange = errorRange, decoderStoppingCriterion = NUMBER_OF_DECODER_ITERATIONS, numberOfSamples = NUMBER_OF_SAMPLES)
        end = time.time()
        # print(f"Logical error rate: {logicalER}")
        # print(f"Decoder failure rate: {decoderFailureRate}")
        # print(f"Time taken: {end - start}")

        # logicalERMS, decoderFailureRateMS = decoderEvaluator(decoderFunction = ldpcDecoderWrapper, dualBinary = True, Hx = value[0], Hz = value[1], errorRange = errorRange, decoderStoppingCriterion = 20, numberOfSamples = 30)
        errorRate = logicalER + decoderFailureRate
        
        timeMeasured = end - start
        data = {"Code name": key, 
                "errorRange": errorRange, 
                "logicalErrorRate": logicalER, 
                "decoderFailureRate": decoderFailureRate, 
                "errorRate": errorRate,
                "time": timeMeasured,
                "logicalER": logicalER,
                "decoderFER": decoderFailureRate,
                "Number of iterations": NUMBER_OF_DECODER_ITERATIONS,
                "Number of samples": NUMBER_OF_SAMPLES,
                "Decoder": "Dual binary BPOSD"}
        fileName = SAVE_PATH + "/code_evaluation_for_bb_gym_reward_calculation_50_samples_50_iterations_geometric5_" + key + ".npy"
        np.save(fileName, data, allow_pickle=True)
