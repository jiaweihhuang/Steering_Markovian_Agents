import numpy as np

GameInformation = {
    'PD': {
        'reward_kernel': {
            'player1_reward': np.array(
                    [
                        [-1., -3.],
                        [0., -2.],
                    ]
                ),
            'player2_reward': np.array(
                    [
                        [-1., 0.],
                        [-3., -2.],
                    ]
                ),
        },
        'utility_reward_kernel': {
            'player1_reward': np.array(
                    [
                        [-1., -3.],
                        [0., -2.],
                    ]
                ),
            'player2_reward': np.array(
                    [
                        [-1., 0.],
                        [-3., -2.],
                    ]
                ),
        },
        'target_state': {
            'player_1': np.array([10., -10.]).reshape([1, -1]),
            'player_2': np.array([10., -10.]).reshape([1, -1]),
        },
        'target_policy': {
            'player_1': np.array([1.0, 0.0]).reshape([1, -1]),
            'player_2': np.array([1.0, 0.0]).reshape([1, -1]),
        },
    },
    'Cooperative': {
        'reward_kernel': {
            'player1_reward': np.array(
                    [
                        [2., 0.],
                        [0., 1.],
                    ]
                ),
            'player2_reward': np.array(
                    [
                        [2., 0.],
                        [0., 1.],
                    ]
                ),
        },
        'utility_reward_kernel': {
            'player1_reward': np.array(
                    [
                        [2., 0.],
                        [0., 1.],
                    ]
                ),
            'player2_reward': np.array(
                    [
                        [2., 0.],
                        [0., 1.],
                    ]
                ),
        },
        'target_state': {
            'player_1': np.array([10., -10.]).reshape([1, -1]),
            'player_2': np.array([10., -10.]).reshape([1, -1]),
        },
        'target_policy': {
            'player_1': np.array([1.0, 0.0]).reshape([1, -1]),
            'player_2': np.array([1.0, 0.0]).reshape([1, -1]),
        },
    },
    'ZeroSum': {
        'reward_kernel': {
            'player1_reward': np.array(
                    [
                        [1., -1.],
                        [-1., 1.],
                    ]
                ),
            'player2_reward': np.array(
                    [
                        [-1., 1.],
                        [1., -1.],
                    ]
                ),
        },
        'utility_reward_kernel': {
            'player1_reward': np.array(
                    [
                        [1., -1.],
                        [-1., 1.],
                    ]
                ),
            'player2_reward': np.array(
                    [
                        [-1., 1.],
                        [1., -1.],
                    ]
                ),
        },
        'target_state': {
            'player_1': np.array([0., 0.]).reshape([1, -1]),
            'player_2': np.array([0., 0.]).reshape([1, -1]),
        },
        'target_dual_diff': {
            'player_1': 0.0,
            'player_2': 0.0,
        },
        'target_policy': {
            'player_1': np.array([0.5, 0.5]).reshape([1, -1]),
            'player_2': np.array([0.5, 0.5]).reshape([1, -1]),
        },
    },
    'StagHunt': {
        'reward_kernel': {
            'player1_reward': np.array(
                    [
                        [5., 0.],
                        [4., 2.],
                    ]
                ),
            'player2_reward': np.array(
                    [
                        [5., 4.],
                        [0., 2.],
                    ]
                ),
        },
        'utility_reward_kernel': {
            'player1_reward': np.array(
                    [
                        [5., 0.],
                        [4., 2.],
                    ]
                ),
            'player2_reward': np.array(
                    [
                        [5., 4.],
                        [0., 2.],
                    ]
                ),
        },
        'target_state': {
            'player_1': np.array([10., -10.]).reshape([1, -1]),
            'player_2': np.array([10., -10.]).reshape([1, -1]),
        },
        'target_dual_diff': {
            'player_1': 10.0,
            'player_2': 10.0,
        },
        'target_policy': {
            'player_1': np.array([1.0, 0.0]).reshape([1, -1]),
            'player_2': np.array([1.0, 0.0]).reshape([1, -1]),
        },
    },
}