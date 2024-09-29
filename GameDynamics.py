import numpy as np
import torch
from copy import deepcopy
from GameInfo import GameInformation
from functools import reduce

class TwoPlayersGame:
    def __init__(self, game):
        # number of states and actions
        self.S = 1
        self.A = {
            'player_1': 2,
            'player_2': 2,
        }
        self.num_players = 2

        # gamma factor; normal form game setting we have self.gamma = 0
        self.gamma = 0

        # reward function with shape: A1xA2
        if game in GameInformation.keys():
            reward_kernel = GameInformation[game]['reward_kernel']
            utility_reward_kernel = GameInformation[game]['utility_reward_kernel']
        else:
            raise NotImplementedError
        
        self.player1_reward = reward_kernel['player1_reward']
        self.player2_reward = reward_kernel['player2_reward']

        self.player1_reward_th = torch.tensor(self.player1_reward, requires_grad=False, dtype=torch.float64)
        self.player2_reward_th = torch.tensor(self.player2_reward, requires_grad=False, dtype=torch.float64)

        self.player_1_utility = utility_reward_kernel['player1_reward']
        self.player_2_utility = utility_reward_kernel['player2_reward']


    # compute the payment w.r.t. the steering reward
    def compute_steering_payments_rates(self, policies, steer_r):
        pi1, pi2 = policies['player_1'], policies['player_2']

        ret1 = pi1.reshape([1, -1]) @ steer_r['player_1'].reshape([-1, 1])
        ret2 = pi2.reshape([1, -1]) @ steer_r['player_2'].reshape([-1, 1])

        return {
            'player_1': ret1,
            'player_2': ret2,
        }
    
    # compute the Q value for player_{player_index}.
    # return a vector with shape [S, A]
    def compute_Q_function(self, policies, steer_r=None):
        pi1, pi2 = policies['player_1'], policies['player_2']
        
        Q1 = pi2.reshape([1, -1]) @ self.player1_reward.T
        Q2 = pi1.reshape([1, -1]) @ self.player2_reward

        if steer_r is not None:
            Q1 += steer_r['player_1'].reshape([1, -1])
            Q2 += steer_r['player_2'].reshape([1, -1])

        return {'player_1': Q1.reshape([self.S, self.A['player_1']]), 'player_2': Q2.reshape([self.S, self.A['player_2']])}


    # compute the return with torch policies
    def compute_total_return_torch(self, policies):
        pi1, pi2 = policies['player_1'], policies['player_2']

        ret1 = pi1.reshape([1, -1]) @ self.player1_reward_th @ pi2.reshape([-1, 1])
        ret2 = pi1.reshape([1, -1]) @ self.player2_reward_th @ pi2.reshape([-1, 1])

        return {
            'player_1': ret1,
            'player_2': ret2,
        }
    
    # compute the return with numpy policies
    def compute_total_return(self, policies):
        pi1, pi2 = policies['player_1'], policies['player_2']

        ret1 = pi1.reshape([1, -1]) @ self.player1_reward @ pi2.reshape([-1, 1])
        ret2 = pi1.reshape([1, -1]) @ self.player2_reward @ pi2.reshape([-1, 1])

        return {
            'player_1': ret1,
            'player_2': ret2,
        }
    
    # the Q returned should have dimension SxA
    def compute_Q_function_torch(self, policies, steer_r):
        pi1, pi2 = policies['player_1'], policies['player_2']
        
        Q1 = pi2.reshape([1, -1]) @ self.player1_reward_th.T + steer_r['player_1'].reshape([1, -1])
        Q2 = pi1.reshape([1, -1]) @ self.player2_reward_th + steer_r['player_2'].reshape([1, -1])

        assert Q1.shape == (self.S, self.A['player_1'])
        assert Q2.shape == (self.S, self.A['player_2'])

        return {'player_1': Q1.reshape([self.S, self.A['player_1']]), 'player_2': Q2.reshape([self.S, self.A['player_2']])}
    
    def compute_total_utility(self, policies):
        pi1, pi2 = policies['player_1'], policies['player_2']

        ret1 = pi1.reshape([1, -1]) @ self.player_1_utility @ pi2.reshape([-1, 1])
        ret2 = pi1.reshape([1, -1]) @ self.player_2_utility @ pi2.reshape([-1, 1])

        return ret1 + ret2


# use matrix multiplication to compute value
class MP_CooperativeGame:
    def __init__(self, num_players=5, 
                 num_acts=2,
                 R_max=0.0,
                 R_min=0.0,
                 eta_max=2.0,
                 eta_min=1.0):
        # number of states and actions
        self.S = 1
        self.num_players = num_players
        self.A = {}
        self.target_state = {}
        for n in range(1, self.num_players + 1):
            self.A['player_{}'.format(n)] = num_acts
            self.target_state['player_{}'.format(n)] = np.array([10., -10.]).reshape([1, -1])

        # gamma factor; normal form game setting we have self.gamma = 0
        self.gamma = 0

        self.R_max = R_max
        self.R_min = R_min

        self.eta_max = eta_max
        self.eta_min = eta_min
        

    # compute the payment w.r.t. the steering reward
    def compute_steering_payments_rates(self, policies, steer_r):
        ret = {}
        for n in range(1, self.num_players + 1):
            name = 'player_{}'.format(n)
            pi_n = policies[name]
            ret[name] = np.sum(pi_n.squeeze() * steer_r[name].squeeze())

        return ret


    # compute the Q value for player_{player_index}.
    # return a vector with shape [S, A]
    def compute_Q_function(self, policies, steer_r=None):
        ret = {}
        piA_array, piB_array = np.zeros([self.num_players]), np.zeros([self.num_players])
        for n in range(1, self.num_players + 1):
            name = 'player_{}'.format(n)
            pi_n = policies[name].squeeze()
            piA_array[n-1] = pi_n[0]
            piB_array[n-1] = pi_n[1]
        
        piA = np.prod(piA_array)
        piB = np.prod(piB_array)
        for n in range(1, self.num_players + 1):
            name = 'player_{}'.format(n)
            pi_n = policies[name].squeeze()
            pi_A_others = piA / pi_n[0]
            pi_B_others = piB / pi_n[1]

            Qn = np.array([pi_A_others * self.R_max, pi_B_others * self.R_min]).reshape([1, 2])

            if steer_r is not None:
                Qn += steer_r[name].reshape([1, 2])
            ret[name] = Qn

        return ret
    

    # compute the return with numpy policies
    def compute_total_return(self, policies):
        ret = {}
        piA, piB = 1.0, 1.0
        for n in range(1, self.num_players + 1):
            name = 'player_{}'.format(n)
            pi_n = policies[name].squeeze()
            piA *= pi_n[0]
            piB *= pi_n[1]

        shared_total_return = piA * self.R_max + piB * self.R_min
        for n in range(1, self.num_players + 1):
            name = 'player_{}'.format(n)
            ret[name] = shared_total_return

        return ret

    # compute the return with numpy policies
    def compute_total_utility(self, policies):
        ret = {}
        piA, piB = 1.0, 1.0
        for n in range(1, self.num_players + 1):
            name = 'player_{}'.format(n)
            pi_n = policies[name].squeeze()
            piA *= pi_n[0]
            piB *= pi_n[1]

        shared_total_return = piA * self.eta_max + piB * self.eta_min
        
        return shared_total_return
