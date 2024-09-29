import numpy as np
import scipy
import torch
from copy import deepcopy
from collections import deque


class PMF_Base:
    def __init__(self, 
                 game, 
                 lr,
                 time_interval,
                 init_clip_threshold=1e-8,
                 reg='neg_ent',
                 obs_dim=4,
                 history_length=None,
                 ):
        self.reg = reg
        self.game = game
        self.lr = lr
        self.time_interval = time_interval

        self.num_players = game.num_players
        self.S = game.S
        self.A = game.A
        self.obs_dim = obs_dim

        self.sampled_acts = {}

        if type(init_clip_threshold) is float:
            self.init_clip_threshold = [init_clip_threshold, 1 - init_clip_threshold]
        elif type(init_clip_threshold) is list:
            if len(init_clip_threshold) == 1:
                self.init_clip_threshold = [init_clip_threshold[0], 1 - init_clip_threshold[0]]
            else:
                self.init_clip_threshold = init_clip_threshold
                assert len(init_clip_threshold) == 2
                assert init_clip_threshold[1] >= init_clip_threshold[0]
        else:
            raise NotImplementedError

    
    def random_initialization(self,):
        x = np.random.rand() * (self.init_clip_threshold[1] - self.init_clip_threshold[0]) + self.init_clip_threshold[0]
        obs = np.array([[np.log(x / (1.0 - x)), 0]])
        obs = obs - np.mean(obs)
        return obs

    def compute_payments_rates(self, steer_r, raw_state=None):
        if raw_state is not None:
            theta = {}
            for i in range(1, self.num_players + 1):
                theta['player_{}'.format(i)] = self.projection(raw_state['player_{}'.format(i)])
        else:
            theta = self.theta
        payments_rates = self.game.compute_steering_payments_rates(theta, steer_r)
        
        return payments_rates
    
    def get_internal_state(self):
        return deepcopy(self.dual_variables)

    def compute_policy_distance(self, theta_star, dual_variables=None):
        distances = {}
        if dual_variables is None:
            for i in range(1, self.num_players + 1):
                theta_star_i = theta_star['player_{}'.format(i)]
                distances['player_{}'.format(i)] = np.linalg.norm(self.theta['player_{}'.format(i)] - theta_star_i)
        else:
            for i in range(1, self.num_players + 1):
                theta_star_i = theta_star['player_{}'.format(i)]
                distances['player_{}'.format(i)] = np.linalg.norm(self.projection(dual_variables['player_{}'.format(i)]) - theta_star_i)
            
        return deepcopy(distances)

    def compute_dual_distance(self, dual_diff_star):
        distances = {}
        for i in range(1, self.num_players + 1):
            dual_diff_star_i = dual_diff_star['player_{}'.format(i)]

            dv = self.dual_variables['player_{}'.format(i)].squeeze()
            dv_diff = dv[0] - dv[1]

            distances['player_{}'.format(i)] = np.abs(dv_diff - dual_diff_star_i)
            
        return deepcopy(distances)
    
    
    def compute_utility(self):
        return self.game.compute_total_utility(self.theta)
    
    def get_policy(self):
        return deepcopy(self.theta)
    
    def projection(self, x):
        if self.reg == 'neg_ent':
            if len(x.shape) == 1:
                if type(x) == torch.Tensor:
                    exp_x = torch.exp(x)
                    return exp_x / torch.sum(exp_x)
                return scipy.special.softmax(x)
            else:
                assert len(x.shape) == 2
                if type(x) == torch.Tensor:
                    exp_x = torch.exp(x)
                    return exp_x / torch.sum(exp_x, dim=1, keepdim=True)
                return scipy.special.softmax(x, axis=1)
        else:
            raise NotImplementedError


    def get_percentile(self, data, num=10):
        ret = []
        for i in range(num):
            ret.append(
                np.percentile(data, i * num + num / 2)
            )
        print('percentile ', ret)


    def sample_observation(self):
        self.sampled_acts = {}
        concatenated_obs = []
        for i in range(1, self.num_players + 1):
            # print(self.A['player_{}'.format(i)])
            sampled_action = np.random.choice(np.arange(self.A['player_{}'.format(i)]), p=self.theta['player_{}'.format(i)].squeeze())
            # print(self.theta['player_{}'.format(i)].squeeze(), sampled_action)
            traj_obs = np.zeros_like(self.theta['player_{}'.format(i)])
            traj_obs[:, sampled_action] = 1.0
            # print('traj_obs ', traj_obs)
            # print(self.theta['player_{}'.format(i)], sampled_action)
            concatenated_obs.append(traj_obs)
            self.sampled_acts['player_{}'.format(i)] = traj_obs

        time_stamp = np.array([(self.T - self.step_num) * self.time_interval])

        concatenated_obs = np.array(concatenated_obs).reshape([1, -1])

        return concatenated_obs, self.sampled_acts, time_stamp