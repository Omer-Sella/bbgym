# __init__.py
__version__ = '0.1'
__author__ = 'Omer Sella'
__all__ = ["polynomialCodes", "funWithMatrices", "logicals", "gf4", "utils"]

from gymnasium.envs.registration import register

PACKAGE_NAME = "bb_gym"


register(
    id="bb_gym/bbcode-ldpc-v0",
    entry_point="bb_gym.bb_gym_v_0_1:bicycleBivariateCodeEnvironment",
)
