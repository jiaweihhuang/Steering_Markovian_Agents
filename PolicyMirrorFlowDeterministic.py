import numpy as np
import scipy
import torch
from copy import deepcopy
from PolicyMirrorFlowBase import PMF_Base

class PMF_Deterministic(PMF_Base):
    def __init__(self, 
                 game, 
                 lr,
                 T,
                 time_interval=0.01,
                 init_clip_threshold=1e-8,
                 reg='neg_ent',
                 obs_dim=4,
                 ):
        super().__init__(
            game=game, 
            lr=lr,
            time_interval=time_interval,
            init_clip_threshold=init_clip_threshold,
            reg=reg,
            obs_dim=obs_dim,
        )
        self.T = T
        self.reset()
    
    def initialization(self):
        return self.random_initialization()

    def reset(self, options=None, other_info=None):
        self.step_num = 0
        if options is None:
            self.dual_variables = {}
            self.theta = {}
            for i in range(1, self.num_players + 1):
                dual_variable = self.initialization()
                self.dual_variables['player_{}'.format(i)] = dual_variable
                self.theta['player_{}'.format(i)] = self.projection(dual_variable)
        else:
            if type(options) is dict:
                self.dual_variables = deepcopy(options)
            else:
                dual_variables = deepcopy(options)
                assert len(dual_variables.shape) == 1
                self.dual_variables = {}
                start = 0
                for i in range(1, self.num_players + 1):
                    num_params = self.S * self.A['player_{}'.format(i)]
                    self.dual_variables['player_{}'.format(i)] = dual_variables[start:start+num_params].reshape([self.S, -1])
                    start += num_params
                print('self.dual_variables', self.dual_variables)
                
            for i in range(1, self.num_players + 1):
                self.theta['player_{}'.format(i)] = self.projection(self.dual_variables['player_{}'.format(i)])


    # steer_r is a dict with 'player_{i}' as keys, and vector with shape [S, Ai] as values
    def update(self, steer_r):
        self.step_num += 1
        Q = self.game.compute_Q_function(self.theta, steer_r)
        for i in range(1, self.num_players + 1):
            self.dual_variables['player_{}'.format(i)] += self.lr * Q['player_{}'.format(i)]
            self.dual_variables['player_{}'.format(i)] = self.dual_variables['player_{}'.format(i)] - np.mean(self.dual_variables['player_{}'.format(i)])
            self.theta['player_{}'.format(i)] = self.projection(self.dual_variables['player_{}'.format(i)])
        
        return self.get_state(), deepcopy(self.theta), {'all_posterior': None, 'posterior': None, 'likelihood': None}
    

    # the state in this case is the dual variables, which preserve the full information
    def get_state(self):
        state = []

        for i in range(1, self.num_players + 1):
            dual_variable = self.dual_variables['player_{}'.format(i)].reshape([-1])
            state.append(dual_variable)

        state.append(
            np.array([(self.T - self.step_num) * self.time_interval])
        )

        return np.concatenate(state)
    