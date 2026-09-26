# %%
"""
Tests for bb_gym

Testing plan:
Sanity:
1. Make sure the environment resgiters with gymnasium, and passes their checks
2. Check the action space has the correct size / shape
3. Observation now moved to be a tensordict, so check that the polynomials are returned in the right order there.

Correctness and tests that came up from bugs:
1. The observation space is supposed to be binary, but there is nothing actually enforcing binary observations from the gymnasium side.
2. Check that if I plug in some reference codes, I get the expected rewrad.


"""

import numpy as np
import pytest
import bb_gym #noqa OMER: This is needed to register bb_gym with gymnasium (through __init__.py)

TEST_ERROR_RANGE = np.linspace(10**-4, 10**-1, 10) # TODO: I should take this from utils, since all spaces moved there.

########## bb_gym_v0 tests

def _fake_decoder(Hx, Hz, errorRange, seed=None):
    return np.zeros(len(errorRange)), np.zeros(len(errorRange))


def _make_action_v0(l, m, aX_idx, aY_idx, bX_idx, bY_idx):
    """Build the flat MultiBinary action vector [aX, aY, bX, bY] from exponent lists."""
    
    aX = np.zeros(l * m, dtype=np.int8)
    aY = np.zeros(l * m, dtype=np.int8)
    bX = np.zeros(l * m, dtype=np.int8)
    bY = np.zeros(l * m, dtype=np.int8)
    for i in aX_idx:
        aX[i] = 1
    for i in aY_idx:
        aY[i] = 1
    for i in bX_idx:
        bX[i] = 1
    for i in bY_idx:
        bY[i] = 1
    return np.concatenate([aX, aY, bX, bY])


# ---------------------------------------------------------------------------
# Positive reward checks using ascending errorRange and a fast decoder.
# These use minimumNumberOfLogicalQubits=1 so the decoder is always called.
# ---------------------------------------------------------------------------

############## bb_gym_ldpc_v0 tests (new environment with the decoder baked in and bit flipping mode)
def _make_v2_env(l = 6, m = 6, minimumNumberOfLogicalQubits = 6, bitFlipping = False):
    import gymnasium as gym
    env = gym.make(
        'bb_gym/bbcode-ldpc-v0',
        l=l, m=m,
        errorRange=TEST_ERROR_RANGE,
        minimumNumberOfLogicalQubits=minimumNumberOfLogicalQubits,
        bitFlipping = bitFlipping
    )
    return env


def _makeAction_v2(l, m, aXIndex, bXIndex, aYIndex, bYIndex):
    aXFlip = np.zeros(l + 1)
    bXFlip = np.zeros(l + 1)
    aYFlip = np.zeros(m + 1)
    bYFlip = np.zeros(m + 1)
    aXFlip[aXIndex] = 1
    bXFlip[bXIndex] = 1
    aYFlip[aYIndex] = 1
    bYFlip[bYIndex] = 1
    return np.hstack((np.hstack((aXFlip, bXFlip)), np.hstack((aYFlip, bYFlip)))).astype(np.int8)




def test_bbcodeV2IsRegistered():
    import gymnasium as gym
    allEnvs = gym.envs.registry.keys()
    assert "bb_gym/bbcode-ldpc-v0" in allEnvs


def test_actionSpaceShape():
     env = _make_v2_env(l = 6, m = 10)
     print(env.action_space.shape)
     #assert env.action_space.shape == ?
    


def test_observationSpaceSize():
    env = _make_v2_env()
    expected = 2 * (6 * 6) ** 2
    assert env.observation_space['code'].shape == (expected,)


def test_IBM_72_12_6_v_LDPC():
    """[[72, 12, 6]]: l=6, m=6, A=x³+y+y², B=y³+x+x²"""
    import gymnasium as gym
    l, m = 6, 6
    env = gym.make(
        'bb_gym/bbcode-ldpc-v0',
        l=l, m=m,
        errorRange=TEST_ERROR_RANGE,
        minimumNumberOfLogicalQubits=6,
        bitFlipping = False
    )
    env.reset()
    action = _makeAction_v2(l, m, aXIndex=[3], aYIndex=[1, 2], bXIndex=[1, 2], bYIndex=[3])
    observation, reward, *_ = env.step(action) # Should come back close to 0.033189 if the error range is np.linspace(10**-4, 10**-1, 10) 
    assert float(reward) > 0.029 # TODO: I'm not sure why the reward comes back as SupportsFloat instead of float flag this for inspection.


def test_IBM_90_8_10_LDPC():
    """[[90, 8, 10]]: l=15, m=3, A=x⁹+y+y², B=1+x²+x⁷"""
    import gymnasium as gym
    l, m = 15, 3
    env = gym.make(
        'bb_gym/bbcode-ldpc-v0',
        l=l, m=m,
        errorRange=TEST_ERROR_RANGE,
        minimumNumberOfLogicalQubits=8,
        bitFlipping = False
    )
    env.reset()
    action = _makeAction_v2(l, m, aXIndex=[9], aYIndex=[1, 2], bXIndex=[0, 2, 7], bYIndex=[])
    _, reward, *_ = env.step(action) # should come back roughly 0.04218
    assert float(reward) > 0.035


def test_IBM_108_8_10_LDPC():
    """[[108, 8, 10]]: l=9, m=6, A=x³+y+y², B=y³+x+x²"""
    import gymnasium as gym
    l, m = 9, 6
    env = gym.make(
        'bb_gym/bbcode-ldpc-v0',
        l=l, m=m,
        errorRange=TEST_ERROR_RANGE,
        minimumNumberOfLogicalQubits=8,
        bitFlipping = False
    )
    env.reset()
    action = _makeAction_v2(l, m, aXIndex=[3], aYIndex=[1, 2], bXIndex=[1, 2], bYIndex=[3])
    _, reward, *_ = env.step(action) # Should come back as ~ 0.040959 
    assert float(reward) > 0.035



def test_bbCodesEnvIsRegistered():
    import gymnasium as gym
    # Once bb_gym is imported, it's init file should have registered the environment. We can check that by checking the registry of gymnasium.
    allEnvs = gym.envs.registry.keys()
    assert "bb_gym/bbcode-ldpc-v0" in allEnvs

# def test_resetZerosPolynomials():
#     env = _make_v2_env_with_dualBinaryBPOSDDecoder(l=6, m=6, max_ax=5, max_ay=5, max_bx=5, max_by=5)
#     env.reset()
#     assert np.all(env.aX == 0)
#     assert np.all(env.aY == 0)
#     assert np.all(env.bX == 0)
#     assert np.all(env.bY == 0)


# def test_bitFlipChangesPolynomial():
#     l = 6
#     m = 6
#     env = _make_v2_env(l=l, m=m)
#     env.reset()
#     # flip aX[2]; no-op on aY (5), bX (5), bY (5)
#     env.step(_makeAction(l, m, 2, l, m, m))
#     assert env.aX[2] == 1
#     assert np.sum(env.aX) == 1
#     assert np.all(env.aY == 0)
#     assert np.all(env.bX == 0)
#     assert np.all(env.bY == 0)


# def test_doubleFlipRestores():
#     l = 6
#     m = 6
#     env = _make_v2_env(l=l, m=m, bitFlipping=True)
#     env.reset()
#     old = env.aX[2]
#     env.step(_makeAction(l,m,2,5,5,5))
#     env.step(_makeAction(l,m,2,5,5,5))
#     assert old == env.aX[2]


# def test_noOpLeavesAllPolynomialsUnchanged():
#     env = _make_v2_env(l=8, m=8, bitFlipping= True)
#     env.reset()
#     env.aX[1] = 1
#     env.bX[4]   = 1
#     oldAx = copy.deepcopy(env.aX)
#     oldBx = copy.deepcopy(env.bX)
#     env.step(_makeAction(8,8,8,8,8,8,))
#     assert np.all(env.aX == oldAx)
#     assert np.all(env.aY == 0)
#     assert np.all(env.bX == oldBx)
#     assert np.all(env.bY == 0)


def test_stepReturnSignature():
    env = _make_v2_env(l=6, m=6)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(_makeAction_v2(6, 6, 6, 6, 6, 6))
    #assert obs.shape == ()
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert isinstance(info, dict)


def test_stepReturnsNegativeRewardWhenDimensionTooLow():
    # Use an impossibly high threshold so no code can satisfy it
    env = _make_v2_env(l = 6, m = 6, minimumNumberOfLogicalQubits = 10000)
    env.reset()
    _, reward, *_ = env.step(_makeAction_v2(6, 6, 1, 2, 3, 4))
    assert reward < 0


def test_v2EnvPassesGymSpecCheck():
    from torchrl.envs.libs.gym import GymEnv
    from torchrl.envs.utils import check_env_specs
    base_env = GymEnv(
        "bb_gym/bbcode-ldpc-v0",
        l=6, m=6,
        errorRange=TEST_ERROR_RANGE,
        minimumNumberOfLogicalQubits=6,
    )
    check_env_specs(base_env)


# ---------------------------------------------------------------------------
# IBM Table 1 codes — positive reward checks (ascending errorRange required)
# A(x,y) = sum_{i in aX} x^i  +  sum_{j in aY} y^j
# B(x,y) = sum_{i in bX} x^i  +  sum_{j in bY} y^j
# ---------------------------------------------------------------------------

def _build_IBM_code_v2(env, aX_idx, aY_idx, bX_idx, bY_idx, l,m,):
    """Build a code by sequential single-bit flips."""
    obs, reward = None, None
    for idx in aX_idx:
        obs, reward, *_ = env.step(_makeAction_v2(l, m, idx, l, m, m))
    for idx in aY_idx:
        obs, reward, *_ = env.step(_makeAction_v2(l, m, l, l, idx, m))
    for idx in bX_idx:
        obs, reward, *_ = env.step(_makeAction_v2(l, m, l, idx, m, m))
    for idx in bY_idx:
        obs, reward, *_ = env.step(_makeAction_v2(l, m, l, l, m, idx))
    return reward


def test_observationDictionaryReturnsCorrectOrder():
    from torchrl.envs.libs.gym import GymEnv
    env = GymEnv("bb_gym/bbcode-ldpc-v0",
                          l = 6,
                          m = 6, 
                          errorRange = np.linspace(0.0001, 0.1, 5),
                          minimumNumberOfLogicalQubits = 6,
                          rewardEngineering = True,
                          bitFlipping = True,
                          useDictObservation = True) 
    tensorDict = env.reset()
    assert "aX" in tensorDict.keys()
    assert "bX" in tensorDict.keys()
    assert "aY" in tensorDict.keys()
    assert "bY" in tensorDict.keys()
    assert "code" in tensorDict.keys()
    assert "k" in tensorDict.keys()


if __name__ == "__main__":
    test_bbcodeV2IsRegistered()
    # test_observationDictionaryReturnsCorrectOrder()
    # test_actionSpaceShape()
    # test_v2EnvPassesGymSpecCheck()
    # test_IBM_72_12_6_v_LDPC()
    # test_stepReturnsNegativeRewardWhenDimensionTooLow()
    # test_observationSpaceIsBinary()