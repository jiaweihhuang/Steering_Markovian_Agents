import numpy as np
import random
import torch

def set_random_seed(seed: int) -> None:
    """
    Seed the different random generators.

    :param seed:
    :param using_cuda:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed + 10)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed + 20)

    if torch.cuda.is_available():
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_uniform_initial_policies(grid_number):
    interval = 1.0 / grid_number
    start = interval / 2
    pi_list = [
        start + i * interval for i in range(grid_number)
    ]
    dual_values = []
    for x in pi_list:
        obs = np.array([[np.log(x / (1.0 - x)), 0]])
        obs = obs - np.mean(obs)
        dual_values.append(obs)
    
    init_points = []

    for x in dual_values:
        for y in dual_values:
            init_points.append(
                {
                    'player_1': x.copy(),
                    'player_2': y.copy(),
                }
            )
    return init_points


def compute_uniform_initial_policies_one_player(grid_number):
    interval = 1.0 / grid_number
    start = interval / 2
    pi_list = [
        start + i * interval for i in range(grid_number)
    ]
    dual_values = []
    for x in pi_list:
        obs = np.array([[np.log(x / (1.0 - x)), 0]])
        obs = obs - np.mean(obs)
        dual_values.append(obs)
    
    return dual_values


'''
only works for two actions case
'''
def policy_to_dual_variables(x):
    return np.array([[np.log(x / (1.0 - x)), 0]])