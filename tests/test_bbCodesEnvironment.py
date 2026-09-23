# %%
"""
Table 1 from:
  "High-threshold and low-overhead fault-tolerant quantum memory"
  Bravyi, Cross, Gambetta, Nazarov, Rall, Yoder — Nature 2024 (arXiv:2308.07915)

Small examples of Bivariate Bicycle (BB) LDPC codes and their performance for the
circuit-based noise model.  All codes have weight-6 checks, thickness-2 Tanner graph,
and a depth-7 syndrome measurement circuit.  A code with parameters [[n, k, d]] requires
2n physical qubits in total and achieves the net encoding rate r = k/2n (we round r down
to the nearest inverse integer).  Circuit-level distance d_circ is the minimum number of
faulty operations in the syndrome measurement circuit required to generate an undetectable
logical error.  The pseudo-threshold p_0 is a solution of the break-even equation
p_L(p) = k·p, where p and p_L are the physical and logical error rates respectively.

  ┌────────────────┬───────────┬─────────┬────────┬───────────┬────────────┐
  │  [[n, k, d]]  │  Rate r   │ d_circ  │  p_0   │  p_L(p1)  │  p_L(p2)  │
  ├────────────────┼───────────┼─────────┼────────┼───────────┼────────────┤
  │ [[72,  12,  6]]│   1/12    │   ≤6    │ 0.0048 │  7×10⁻⁵  │  7×10⁻⁸  │
  │ [[90,   8, 10]]│   1/23    │   ≤8    │ 0.0053 │  5×10⁻⁶  │  4×10⁻¹⁰ │
  │ [[108,  8, 10]]│   1/27    │   ≤8    │ 0.0058 │  3×10⁻⁶  │  1×10⁻¹⁰ │
  │ [[144, 12, 12]]│   1/24    │   ≤10   │ 0.0065 │  2×10⁻⁷  │  8×10⁻¹³ │
  │ [[288, 12, 18]]│   1/48    │   ≤18   │ 0.0069 │  2×10⁻¹² │  1×10⁻²² │
  └────────────────┴───────────┴─────────┴────────┴───────────┴────────────┘
"""

import numpy as np
import pytest

TEST_ERROR_RANGE = np.linspace(10**-4, 10**-1, 10)

########## bbgym_v0 tests

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


def test_observationSpaceIsBinary():    
    import gymnasium as gym
    l, m = 12, 12
    env = gym.make('qecc/bbcode-v0', l=l, m=m,
                   evaluationDecoderFunction=_fake_decoder,
                   errorRange=TEST_ERROR_RANGE,
                   minimumNumberOfLogicalQubits=1)
    observation, info = env.reset()

    assert np.all(observation == (observation % 2))

    observation, reward, terminated, _, _ = env.step(_make_action_v0(l, m, [3], [2, 7], [1, 2], [3]))
    
    assert np.all(observation == (observation % 2))


def test_IBM_72_12_6():
    """[[72, 12, 6]]: l=6, m=6, A=x³+y+y², B=y³+x+x²"""
    import gymnasium as gym
    from qecc.bb_gym import exampleDecoderFunction2
    l, m = 6, 6
    env = gym.make(
        'qecc/bbcode-v0',
        l=l, m=m,
        evaluationDecoderFunction=exampleDecoderFunction2,
        errorRange=TEST_ERROR_RANGE,
        minimumNumberOfLogicalQubits=12,
    )
    env.reset()
    action = _make_action_v0(l, m, aX_idx=[3], aY_idx=[1, 2], bX_idx=[1, 2], bY_idx=[3])
    _, reward, *_ = env.step(action) # Should come back close to 0.033189 if the error range is np.linspace(10**-4, 10**-1, 10) 
    
    assert float(reward) > 0.029 # TODO: I'm not sure why the reward comes back as SupportsFloat instead of float flag this for inspection.


def test_IBM_90_8_10():
    """[[90, 8, 10]]: l=15, m=3, A=x⁹+y+y², B=1+x²+x⁷"""
    import gymnasium as gym
    from qecc.bb_gym import exampleDecoderFunction2
    l, m = 15, 3
    env = gym.make(
        'qecc/bbcode-v0',
        l=l, m=m,
        evaluationDecoderFunction=exampleDecoderFunction2,
        errorRange=TEST_ERROR_RANGE,
        minimumNumberOfLogicalQubits=8,
    )
    env.reset()
    action = _make_action_v0(l, m, aX_idx=[9], aY_idx=[1, 2], bX_idx=[0, 2, 7], bY_idx=[])
    _, reward, *_ = env.step(action) # should come back roughly 0.04218
    
    assert float(reward) > 0.035


def test_IBM_108_8_10():
    """[[108, 8, 10]]: l=9, m=6, A=x³+y+y², B=y³+x+x²"""
    import gymnasium as gym
    from qecc.bb_gym import exampleDecoderFunction2
    l, m = 9, 6
    env = gym.make(
        'qecc/bbcode-v0',
        l=l, m=m,
        evaluationDecoderFunction=exampleDecoderFunction2,
        errorRange=TEST_ERROR_RANGE,
        minimumNumberOfLogicalQubits=8,
    )
    env.reset()
    action = _make_action_v0(l, m, aX_idx=[3], aY_idx=[1, 2], bX_idx=[1, 2], bY_idx=[3])
    _, reward, *_ = env.step(action) # Should come back as ~ 0.040959 
    assert float(reward) > 0.035


def test_IBM_144_12_12():
    """[[144, 12, 12]]: l=12, m=6, A=x³+y+y², B=y³+x+x²"""
    import gymnasium as gym
    from bb_gym import exampleDecoderFunction2
    l, m = 12, 6
    env = gym.make(
        'qecc/bbcode-v0',
        l=l, m=m,
        evaluationDecoderFunction=exampleDecoderFunction2,
        errorRange=TEST_ERROR_RANGE,
        minimumNumberOfLogicalQubits=12,
    )
    env.reset()
    action = _make_action_v0(l, m, aX_idx=[3], aY_idx=[1, 2], bX_idx=[1, 2], bY_idx=[3])
    _, reward, *_ = env.step(action) #Should come back as ~ 0.038739
    assert float(reward) > 0.03


def test_IBM_288_12_18():
    """[[288, 12, 18]]: l=12, m=12, A=x³+y²+y⁷, B=y³+x+x²"""
    import gymnasium as gym
    from qecc.bb_gym import exampleDecoderFunction2

    l, m = 12, 12
    env = gym.make(
        'bb_gym/bbcode-v0',
        l=l, m=m,
        evaluationDecoderFunction=exampleDecoderFunction2,
        errorRange=TEST_ERROR_RANGE,
        minimumNumberOfLogicalQubits=12,
    )
    env.reset()
    action = _make_action_v0(l, m, aX_idx=[3], aY_idx=[2, 7], bX_idx=[1, 2], bY_idx=[3])
    _, reward, *_ = env.step(action) # Should come back as ~ 0.0414
    assert float(reward) > 0.035 


# def test_bbCodesEnvIsWorking():
#     from qecc.minSum import ldpcDecoderWrapper
#     from qecc.utils import decoderEvaluator
#     def decoderFunction(Hx,Hz,errorRange):
#         numberOfSamples = 30
#         logicalErrors, decoderFailures =  decoderEvaluator(decoderFunction = ldpcDecoderWrapper, dualBinary = True, Hx = Hx, Hz = Hz, errorRange = errorRange, decoderStoppingCriterion = 50, numberOfSamples = numberOfSamples)
#         #return {key: value/numberOfSamples for key,value in logicalErrors.items()} , {key: value/numberOfSamples for key,value in decoderFailures.items()}
#         return logicalErrors/numberOfSamples, decoderFailures/numberOfSamples



# ---------------------------------------------------------------------------
# Positive reward checks using ascending errorRange and a fast decoder.
# These use minimumNumberOfLogicalQubits=1 so the decoder is always called.
# ---------------------------------------------------------------------------

def _fake_decoder(Hx, Hz, errorRange, seed=None):
    return np.zeros(len(errorRange)), np.zeros(len(errorRange))


def test_IBM_72_12_6_positiveReward():
    """[[72, 12, 6]]: l=6, m=6, A=x³+y+y², B=y³+x+x²"""
    import gymnasium as gym
    l, m = 6, 6
    env = gym.make('qecc/bbcode-v0', l=l, m=m,
                   evaluationDecoderFunction=_fake_decoder,
                   errorRange=TEST_ERROR_RANGE,
                   minimumNumberOfLogicalQubits=1)
    env.reset()
    _, reward, *_ = env.step(_make_action_v0(l, m, [3], [1, 2], [1, 2], [3]))
    assert reward > 0


def test_IBM_90_8_10_positiveReward():
    """[[90, 8, 10]]: l=15, m=3, A=x⁹+y+y², B=1+x²+x⁷"""
    import gymnasium as gym
    l, m = 15, 3
    env = gym.make('qecc/bbcode-v0', l=l, m=m,
                   evaluationDecoderFunction=_fake_decoder,
                   errorRange=TEST_ERROR_RANGE,
                   minimumNumberOfLogicalQubits=1)
    env.reset()
    _, reward, *_ = env.step(_make_action_v0(l, m, [9], [1, 2], [0, 2, 7], []))
    assert reward > 0


def test_IBM_108_8_10_positiveReward():
    """[[108, 8, 10]]: l=9, m=6, A=x³+y+y², B=y³+x+x²"""
    import gymnasium as gym
    l, m = 9, 6
    env = gym.make('qecc/bbcode-v0', l=l, m=m,
                   evaluationDecoderFunction=_fake_decoder,
                   errorRange=TEST_ERROR_RANGE,
                   minimumNumberOfLogicalQubits=1)
    env.reset()
    _, reward, *_ = env.step(_make_action_v0(l, m, [3], [1, 2], [1, 2], [3]))
    assert reward > 0


def test_IBM_144_12_12_positiveReward():
    """[[144, 12, 12]]: l=12, m=6, A=x³+y+y², B=y³+x+x²"""
    import gymnasium as gym
    l, m = 12, 6
    env = gym.make('qecc/bbcode-v0', l=l, m=m,
                   evaluationDecoderFunction=_fake_decoder,
                   errorRange=TEST_ERROR_RANGE,
                   minimumNumberOfLogicalQubits=1)
    env.reset()
    _, reward, *_ = env.step(_make_action_v0(l, m, [3], [1, 2], [1, 2], [3]))
    assert reward > 0


def test_IBM_288_12_18_positiveReward():
    """[[288, 12, 18]]: l=12, m=12, A=x³+y²+y⁷, B=y³+x+x²"""
    import gymnasium as gym
    l, m = 12, 12
    env = gym.make('qecc/bbcode-v0', l=l, m=m,
                   evaluationDecoderFunction=_fake_decoder,
                   errorRange=TEST_ERROR_RANGE,
                   minimumNumberOfLogicalQubits=1)
    env.reset()
    _, reward, *_ = env.step(_make_action_v0(l, m, [3], [2, 7], [1, 2], [3]))
    assert reward > 0


############## bbgym_ldpc_v0 tests (new environment with the decoder baked in and bit flipping mode)
def _make_v2_env(l = 6, m = 6, minimumNumberOfLogicalQubits = 6, bitFlipping = False):
    import gymnasium as gym
    env = gym.make(
        'qecc/bbcode-ldpc-v0',
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
    assert "qecc/bbcode-ldpc-v0" in allEnvs


def test_actionSpaceShape():
     env = _make_v2_env(l = 6, m = 10)
     print(env.action_space.shape)
     #assert env.action_space.shape == ?
    


def test_observationSpaceSize():
    env = _make_v2_env()
    expected = 2 * (6 * 6) ** 2
    assert env.observation_space.shape == (expected,)


def test_IBM_72_12_6_v_LDPC():
    """[[72, 12, 6]]: l=6, m=6, A=x³+y+y², B=y³+x+x²"""
    import gymnasium as gym
    l, m = 6, 6
    env = gym.make(
        'qecc/bbcode-ldpc-v0',
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
        'qecc/bbcode-ldpc-v0',
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
        'qecc/bbcode-ldpc-v0',
        l=l, m=m,
        errorRange=TEST_ERROR_RANGE,
        minimumNumberOfLogicalQubits=8,
        bitFlipping = False
    )
    env.reset()
    action = _makeAction_v2(l, m, aXIndex=[3], aYIndex=[1, 2], bXIndex=[1, 2], bYIndex=[3])
    _, reward, *_ = env.step(action) # Should come back as ~ 0.040959 
    assert float(reward) > 0.035


### Shared tests for all environment versions
@pytest.mark.parametrize(
        "envName",
    ["bbGym/bbcode-v0", "bbGym/bbcode-ldpc-v0"])
def test_bbCodesEnvIsRegistered(envName):
    import gymnasium as gym
    # Once qecc is imported, it's init file should have registered the environment. We can check that by checking the registry of gymnasium.
    allEnvs = gym.envs.registry.keys()
    assert envName in allEnvs

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
        "qecc/bbcode-ldpc-v0",
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
    test_observationDictionaryReturnsCorrectOrder()
    test_actionSpaceShape()
    test_v2EnvPassesGymSpecCheck()
    test_IBM_72_12_6_v_LDPC()
    test_stepReturnsNegativeRewardWhenDimensionTooLow()
    test_observationSpaceIsBinary()
    test_IBM_72_12_6()
    test_IBM_72_12_6_positiveReward()
    test_IBM_90_8_10()
    test_IBM_108_8_10()
    test_IBM_144_12_12()
    test_IBM_288_12_18()
