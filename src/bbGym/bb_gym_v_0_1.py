# %%
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from polynomialCodes import generateABmatrices, bicycleCodeFromAB
from bbGym.logicals import calculateCodeDimension
from bbGym.gf4 import integerToDualBinary
from ldpc import BpOsdDecoder #noqa
from bbGym.logicals import computeLogicals
import json
import os
import platform
from datetime import datetime, timezone
from bbGym.utils import calculateRewardFromSamples

INT_DATA_TYPE = np.int16

# Following https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/


class bicycleBivariateCodeEnvironment(gym.Env):
    metadata = {"render_modes" : []}
    """
    A gymnasium environment to learn bicycle bivariate codes as described in Bivariate Bicycle codes from High-threshold and low-overhead fault-tolerant quantum memory
    Some parameters for BB codes from https://arxiv.org/pdf/2308.07915
    [[72, 12, 6]]
    [[90, 8, 10]]
    [[108, 8, 10]]
    [[144, 12,12]]
    [[288, 12,18]]
    """

    def __init__(self, l, m, 
                 errorRange = np.linspace(0.0001, 0.1, 10),
                 minimumNumberOfLogicalQubits = 6,
                 render_mode = None,
                 numberOfSamples = 50,
                 numberOfIterations = 50,
                 rewardEngineering = False,
                 seed = 0,
                 codeLogging = True,
                 bitFlipping = False,
                 useDictObservation = True,
                 resetType = "zero",
                 stepsLimit = 1000000, # Some large value to maintain legacy, but should be set wisely !
                 ):
        
        self.resetType = resetType
        self.episodeStepsLimit = stepsLimit
        self.numberOfSteps = 0
        self.useDictObservation = useDictObservation
        self.codeLogging = codeLogging
        self.render_mode = render_mode # There is no rendering, but we have to accept and store it to comply with gymnasium spec.
        self.minimumNumberOfLogicalQubits = minimumNumberOfLogicalQubits
        self.numberOfIterations = numberOfIterations
        self.numberOfSamples = numberOfSamples
        self.ms_scaling_factor = 0.625
        self._l = l
        self._m = m
        self.seed = seed # Omer: WARNING ! The assumption is that either on init, or later in reset, or parallelEnv or collector will set a seed.
        self.errorRange = errorRange
        self.bitFlipping = bitFlipping
        if any(a >= b for a, b in zip(errorRange, errorRange[1:])):
            raise ValueError(
                f"errorRange must be strictly increasing (e.g. [0.001, 0.01, 0.1]); got {list(errorRange)}"
            )
        
        # Since x=Sℓ⊗Im and y=Iℓ⊗Sm we have x^l = y^m = I_{l*m}, so aX, bX terms over l wrap around using x^l = 1 and
        # aY,bY terms higher than m wrap around as y^m = 1
        # The action space is a flat array containing containing actions [aX + 1,bX + 1,aY +1 ,bY + 1] in that order. +1 because there is a no op bit.
        
        self.action_space = spaces.MultiBinary(2 * l + 2 * m + 4)
        
        self.aX = np.zeros(l, INT_DATA_TYPE)
        self.bX = np.zeros(l, INT_DATA_TYPE)
        self.aY = np.zeros(m, INT_DATA_TYPE)
        self.bY = np.zeros(m, INT_DATA_TYPE)
        
        self.A, self.B = generateABmatrices(self._l, self._m,
                                            np.where(self.aX != 0)[0],
                                            np.where(self.aY != 0)[0],
                                            np.where(self.bX != 0)[0],
                                            np.where(self.bY != 0)[0])
        
        self.Hx, self.Hz = bicycleCodeFromAB(self.A, self.B)
        
        if self.useDictObservation:
            self.observation_space = spaces.Dict({
                                    "aX"  : spaces.Box(low=0.0, high=1.0, shape=(l,), dtype=np.float32),
                                    "bX"  : spaces.Box(low=0.0, high=1.0, shape=(l,), dtype=np.float32),
                                    "aY"  : spaces.Box(low=0.0, high=1.0, shape=(m,), dtype=np.float32),
                                    "bY"  : spaces.Box(low=0.0, high=1.0, shape=(m,), dtype=np.float32),
                                    "code": spaces.Box(low=0.0, high=1.0, shape=(2 * (l * m) ** 2,), dtype=np.float32),
                                    "k"   : spaces.Box(low=0.0, high=2.0 * l * m, shape=(1,), dtype=np.float32), })
        else:
            self.observation_space = spaces.MultiBinary(len(self._getObservation()))

        self.rewardEngineering = rewardEngineering

    # Gymnasium spec requires a render function and a close function:
    def render(self):
        pass

    def close(self):
        pass

    def _getObservation(self):
        if self.useDictObservation:
            observation = {
                            "aX"    :       self.aX.flatten().astype(np.float32),
                            "bX"    :       self.bX.flatten().astype(np.float32),
                            "aY"    :       self.aY.flatten().astype(np.float32),
                            "bY"    :       self.bY.flatten().astype(np.float32),
                            "code"  :       np.concatenate([self.A, self.B]).flatten().astype(np.float32),
                            "k"     :       np.array([self.numberOfLogicalQubits], dtype=np.float32),
                        }
        else:
            observation = np.concatenate([self.A, self.B]).flatten().astype(np.int8)
        return observation
    
    def reset(self, seed=None, options = None):
        if seed is None:
            self.seed = self.seed + 1
        else:
            self.seed = seed
        super().reset(seed = self.seed)

        self.numberOfSteps = 0
        self.aX = self.aX * 0
        self.aY = self.aY * 0
        self.bX = self.bX * 0
        self.bY = self.bY * 0
        
        if self.resetType == "random3":
            # localRandom = np.random.RandomState(self.seed)
            for p in [self.aX, self.aY, self.bX, self.bY]:
                numberOfNonZero = self.np_random.integers(0, 4)
                p[self.np_random.choice(len(p), size = numberOfNonZero, replace = False)] = 1
        elif self.resetType == "zero":
            pass
        else:
            raise ValueError(f"Not implemented for reset type {self.resetType}")
        
        self.A, self.B = generateABmatrices(self._l, self._m, # 
                                            np.where(self.aX != 0)[0],
                                            np.where(self.aY != 0)[0],
                                            np.where(self.bX != 0)[0],
                                            np.where(self.bY != 0)[0])
        
        self.Hx, self.Hz = bicycleCodeFromAB(self.A, self.B)
        self.numberOfLogicalQubits = calculateCodeDimension(self.Hx, self.Hz)
        observation = self._getObservation()
        # Reset the number of steps that were made to 0
        info = {}
        return observation, info
    
    def step(self, action):
        # super().step(action)
        # Unpack action from flat action
        actionCopy = np.array(action, dtype = INT_DATA_TYPE, copy = True) # This takes care of two things: 
                                                                          # 1. We are about to XOR the action from the policy 
                                                                          # (which is float data type) with an int. 
                                                                          # 2. We are about to assign a slice of the action to an internal polynomial, 
                                                                          # and that must not be a view, rather a copy.
        
        # Note that the action slicing is not the same as the observation slicing: action =  (aX,bX,aY,bY) 
        # whereas (assuming that we are returning the polynomials) 
        # observation = [aX | aY | bX | bY | A|B-flattened]
        aXAction = actionCopy[0 : self._l + 1]
        bXaction = actionCopy[self._l + 1 : 2 * self._l + 2]
        aYAction = actionCopy[2 * self._l + 2 : 2 * self._l + self._m + 3]
        bYAction = actionCopy[2 * self._l + self._m + 3 : 2 * self._l + 2 * self._m + 4]
        if self.bitFlipping: # This mode means we are switching bits on and off
            self.aX ^= aXAction[0:-1]
            self.bX ^= bXaction[0:-1]
            self.aY ^= aYAction[0:-1]
            self.bY ^= bYAction[0:-1]
        else: # This mode means we are completely ignoring the present state and replacing it (stateless environment)
            self.aX = aXAction[0:-1]
            self.bX = bXaction[0:-1]
            self.aY = aYAction[0:-1]
            self.bY = bYAction[0:-1]
        self.numberOfSteps += 1
        self.A, self.B = generateABmatrices(self._l, self._m, 
                                            np.where(self.aX != 0)[0], 
                                            np.where(self.aY != 0)[0], 
                                            np.where(self.bX != 0)[0], 
                                            np.where(self.bY != 0)[0])

        self.Hx, self.Hz = bicycleCodeFromAB(self.A, self.B)

        # Omer: Now we check that the resulting code admits the necessary logical qubits
        self.numberOfLogicalQubits = calculateCodeDimension(self.Hx, self.Hz)
        if self.numberOfLogicalQubits >= self.minimumNumberOfLogicalQubits:
            self.seed = self.seed + 1 # Omer: There is a bug hiding here - the seed of this environment might overlap another environment if promoted using +1. Right now I'm avoiding this using bookkeeping of random seeds, but consider changing this to np.randint(self.seed).
            logicalErrorRate = self.decoderEvaluation(self.seed)
            reward = calculateRewardFromSamples(logicalErrorCount=logicalErrorRate, 
                                                numberOfSamples=self.numberOfSamples, 
                                                errorRange=self.errorRange, 
                                                l = self._l, m = self._m, 
                                                rewardEngineering=self.rewardEngineering)
        else:
            reward = (self.numberOfLogicalQubits - self.minimumNumberOfLogicalQubits) / self.minimumNumberOfLogicalQubits

        terminated = False
        observation = self._getObservation()
        info = {}
        # In case we reach the step limit, we ****truncate**** (not terminate !).  
        # In case this is repopened, be aware of the following regarding truncated vs. terminated, as this is not a small change, it has drastic consequences on the learning
        # https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
        # https://gymnasium.farama.org/introduction/migration_guide/
        # https://docs.pytorch.org/rl/main/tutorials/collector_trajectory_assembly.html
        truncated = self.numberOfSteps >= self.episodeStepsLimit 
        return observation, reward, terminated, truncated, info

    def decoderEvaluation(self, seed):  
        # Note that the decoder evaluation is hard-coded here, but there is an option to plug in a different decoder (that's the whole point of the paper - fix the decoder, learn the code).
        localRandom = np.random.RandomState(seed) 
        logicalX, logicalZ = computeLogicals(self.Hx, self.Hz)

        bpDecoderHx = BpOsdDecoder(self.Hx, # This is the X stabilisers parity check matrix
                                    error_rate = 1.0, # Not a bug ! We initialize the error rate at 1.0, but when we set it according to the simulated error rate later.
                                    # channel_probs= initialValues[:,1], # Omer: we're not using this  assign error_rate to each qubit. This will override "error_rate" input variable
                                    max_iter=self.numberOfIterations, # the maximum number of iterations for BP
                                    bp_method="ms", # Min-Sum decoding
                                    ms_scaling_factor=self.ms_scaling_factor, #min sum scaling factor. If set to zero the variable scaling factor method is used
                                    osd_method="osd0", # OSD method. I use osd0 in the paper.
                                    osd_order=0) # osd search depth, I use 0 in the paper.
                                    
        bpDecoderHz = BpOsdDecoder(self.Hz, # This is the Z stabilisers parity check matrix
                                    error_rate=1.0,
                                    # channel_probs= initialValues[:,1], #assign error_rate to each qubit. This will override "error_rate" input variable
                                    max_iter=self.numberOfIterations,
                                    bp_method="ms",
                                    ms_scaling_factor=self.ms_scaling_factor,
                                    osd_method="osd0",
                                    osd_order=0)
        
        logicalErrorRate = np.zeros(len(self.errorRange))
        decoderFailure = np.zeros(len(self.errorRange))
        for i in range(len(self.errorRange)):
            p = float(self.errorRange[i])
            bpDecoderHx.error_rate = p
            bpDecoderHz.error_rate = p
            logicalErrorRate[i] = 0
            
            for _ in range(self.numberOfSamples):
                # Sample (numberOfSamples times) an error, which is a vector over {0,1,2,3} representing I,X,Z,Y (to be consistent check with the documentation in gf4.py)    
                error = localRandom.choice([0, 1, 2, 3], size=self.Hx.shape[1], replace=True, p=[1 - 3 * p, p, p, p])
                errorX, errorZ = integerToDualBinary(error)
                # Calculate the syndrome for this error
                estimatedErrorX = bpDecoderHz.decode(self.Hz.dot(errorX) % 2) # Don't get confused ! the X component is detected, and corrected by the Z stabilizers.
                estimatedErrorZ = bpDecoderHx.decode(self.Hx.dot(errorZ) % 2)
                # So the decoder figured an estimated error vectors (one for X component, one for Z component), and this will be applied as the fix to the real error, the result will be the residual error.
                residualErrorX = (estimatedErrorX + errorX) % 2
                residualErrorZ = (estimatedErrorZ + errorZ) % 2
                # Check whether the residual error gives 0 syndrome: (don't mistake this for "matches the original error", it could be up to a stabilizer and still give 0 syndrome.)
                if not (np.all((np.dot(self.Hx, residualErrorZ)) % 2 == 0) and np.all((np.dot(self.Hz, residualErrorX) % 2 ) == 0)):
                    # Decoder failure: the residual error does not give 0 syndrome, meaning the decoder is wrong
                    decoderFailure[i] += 1
                else: # So we are in the case that the residual error commutes with all stabilizers, i.e., it is in the normalizer. So let's check if it is a stabilizer (commutes with all logicals), or a logical error (anticommutes with some logical operator)
                    if not (np.all((np.dot(logicalZ, residualErrorX) % 2) == 0) and np.all((np.dot(logicalX, residualErrorZ) % 2) == 0)):
                        # the residual error commutes with all stabilizers but anticommutes with some logical operator
                        logicalErrorRate[i] += 1

        if self.codeLogging:
            logCodeEvaluation(self._l, self._m, self.aX, self.aY, self.bX, self.bY, self.numberOfLogicalQubits,
                                self.errorRange, logicalErrorRate, decoderFailure,
                                self.numberOfSamples, self.seed, "dual binary bposd 0", "depolarizing",
                                runId=None, logDirectory=None)

        return logicalErrorRate + decoderFailure # I removed the division /self.numberOfSamples this is now done when calculating the reward.

    def getSeed(self):
        return self.seed


def logCodeEvaluation(l, m, aX, aY, bX, bY, numberOfLogicalQubits,
                      errorRange, logicalErrorCounts, decoderFailureCounts,
                      numberOfSamples, seed, decoderConfig, noiseModel,
                      runId=None, logDirectory=None):
    """Append one decoder-evaluation record to a per-process JSON-lines file.
    This is used in data discovery mode, or for debug when you want to check the distribution (diversity) of codes explored.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runId": runId,
        "l": int(l),
        "m": int(m),
        "aX": [int(i) for i in aX],
        "aY": [int(i) for i in aY],
        "bX": [int(i) for i in bX],
        "bY": [int(i) for i in bY],
        "numberOfLogicalQubits": int(numberOfLogicalQubits),
        "errorRange": [float(p) for p in errorRange],
        "logicalErrorCounts": [int(c) for c in logicalErrorCounts],
        "decoderFailureCounts": [int(c) for c in decoderFailureCounts],
        "numberOfSamples": int(numberOfSamples),
        "seed": int(seed),
        "decoder": decoderConfig,
        "noiseModel": noiseModel,
    }
    if logDirectory is None:
        logDirectory = os.environ.get("QECC_DATA", ".")
    # One file per process: safe under ParallelEnv multiprocessing
    path = os.path.join(logDirectory, "supervisedLearning", "codeEvaluationTrainingData", f"l_{l}_m_{m}", f"codeEvaluations_{platform.node().split('.')[0]}_{os.getpid()}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)   # Create the directory if it doesn't exist
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":

    # Some basic examples of how to use:
    l = 6
    m = 6
    device = 'cpu'

    # Check the environment works with GymEnv
    from torchrl.envs.libs.gym import GymEnv
    from torchrl.envs.utils import check_env_specs
    base_env = GymEnv("qecc/bbcode-ldpc-v0", l = 6, m = 6, errorRange = np.linspace(0.0001, 0.1, 10), minimumNumberOfLogicalQubits = 6)

    check_env_specs(base_env)
    env = gym.make('qecc/bbcode-ldpc-v0', l = 6, m = 6, errorRange = np.linspace(0.0001, 0.1, 10), minimumNumberOfLogicalQubits = 6, bitFlipping = False)    
    env.reset()
    print(env.action_space.shape)
    # print(env.unwrapped.flatObservationSize)
    aX = np.zeros(l + 1)
    aY = np.zeros(m + 1)
    bX = np.zeros(l + 1)
    bY = np.zeros(m + 1)
    print(env.action_space.shape)
    aX[3] = 1
    aY[1] = 1
    aY[2] = 1
    bX[1] = 1
    bX[2] = 1
    bY[3] = 1
    action = np.concatenate([aX, bX, aY, bY])
    print(action)
    observation, reward, _, _, _ = env.step(action = action)
    print(reward) # Should give  ~ 0.0339