import numpy as np
from scipy.stats import norm as Normal
from copy import deepcopy
from PolicyMirrorFlowBase import PMF_Base
from utils import compute_uniform_initial_policies_one_player
from collections import deque

class PMF_Avaricious(PMF_Base):
    def __init__(self, 
                 game, 
                 lr,
                 T,
                 time_interval=0.01,
                 init_clip_threshold=1e-8,
                 reg='neg_ent',
                 shift_array=None,
                 obs_dim=4,
                 threshold=0.0,
                 sigma=0.5,
                 ):
        
        super().__init__(
            game=game, 
            lr=lr,
            time_interval=time_interval,
            init_clip_threshold=init_clip_threshold,
            reg=reg,
            obs_dim=obs_dim,
        )
        self.shift_array = shift_array
        self.T = T

        self.threshold = threshold

        self.sigma = sigma

        self.counter = 0

        self.shift_pass_threshold = 0.5

        self.reset()

    def is_success(self):
        for i in range(1, self.num_players + 1):
            if np.argmax(self.all_posterior['player_{}'.format(i)]) != self.sampled_shift_index['player_{}'.format(i)]:
                return False
        return True
    
    def initialization(self):
        x = 1.0 / 3.0
        obs = np.array([[np.log(x / (1.0 - x)), 0]])
        obs = obs - np.mean(obs)
        
        return deepcopy(obs)
            

    def reset(self, options=None, other_info=None):
        self.counter += 1

        self.step_num = 0

        self.sampled_shift = {}
        self.sampled_shift_index = {}

        self.all_posterior = {}

        if other_info is not None and 'fixed_shift' in other_info.keys():
            fixed_shift = other_info['fixed_shift']
        else:
            fixed_shift = None

        for i in range(1, self.num_players + 1):
            name = 'player_{}'.format(i)

            if fixed_shift:
                sampled_shift_index = np.where(self.shift_array == fixed_shift[i-1])[0].squeeze()
            else:
                sampled_shift_index = np.random.randint(len(self.shift_array))
            self.sampled_shift_index[name] = sampled_shift_index
            self.sampled_shift[name] = self.shift_array[sampled_shift_index]

            self.all_posterior[name] = np.ones([len(self.shift_array)]) / len(self.shift_array)

        if options is None:
            self.dual_variables = {}
            self.theta = {}
            for i in range(1, self.num_players + 1):
                dual_variable = self.initialization()
                self.dual_variables['player_{}'.format(i)] = deepcopy(dual_variable)
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
                
            for i in range(1, self.num_players + 1):
                self.theta['player_{}'.format(i)] = self.projection(self.dual_variables['player_{}'.format(i)])

        Q = self.game.compute_Q_function(self.theta, None)


    def update_posterior(self, agent_index, sample, V_val):
        name = 'player_{}'.format(agent_index)
        
        mu = self.compute_mu(self.shift_array, V_val=V_val, agent_index=agent_index)

        if sample == 0.0:
            likelihood = Normal.cdf(0.0, loc=mu, scale=self.sigma)
        else:
            likelihood = Normal.pdf(sample, loc=mu, scale=self.sigma)

        self.all_posterior[name] *= likelihood
        self.all_posterior[name] /= np.sum(self.all_posterior[name])

        return self.all_posterior[name][self.sampled_shift_index[name]], likelihood[self.sampled_shift_index[name]]
        

    def compute_mu(self, shift, V_val, agent_index):
        piA = self.theta['player_{}'.format(agent_index)][0][0]

        abs_shift = np.abs(shift)
        if piA < 0.5:
            mu_natural = 1.5
            mu_avaricious = 1.5 - (V_val >= abs_shift) * (V_val - abs_shift)
            mu = mu_natural * (shift == 0) + mu_avaricious * (shift < 0)
        else:
            mu_natural = 1.5
            mu_avaricious = 1.5 - 10 * (V_val >= abs_shift) * (V_val - abs_shift)
            mu = mu_natural * (shift == 0) + mu_avaricious * (shift < 0)

        return mu


    def sample_learning_rate_scale(self, agent_index, V_val):
        mu = self.compute_mu(self.sampled_shift['player_{}'.format(agent_index)], V_val=V_val, agent_index=agent_index)
        sample = np.random.randn() * self.sigma + mu

        return np.maximum(sample, 0.0)
    

    # steer_r is a dict with 'player_{i}' as keys, and vector with shape [S, Ai] as values
    def update(self, steer_r):
        self.step_num += 1
        Q_with_steer_r = self.game.compute_Q_function(self.theta, steer_r)
        
        self.posterior = {}
        self.likelihood = {}

        for i in range(1, self.num_players + 1):
            # Qi_with_steer_r = Q_with_steer_r['player_{}'.format(i)]
            Vi_steer_r = np.sum(self.theta['player_{}'.format(i)] * steer_r['player_{}'.format(i)])

            sampled_lr_scale = self.sample_learning_rate_scale(i, Vi_steer_r)
            self.posterior['player_{}'.format(i)], self.likelihood['player_{}'.format(i)] = self.update_posterior(i, sampled_lr_scale, Vi_steer_r)

            self.dual_variables['player_{}'.format(i)] += self.lr * sampled_lr_scale * Q_with_steer_r['player_{}'.format(i)]
            self.dual_variables['player_{}'.format(i)] = self.dual_variables['player_{}'.format(i)] - np.mean(self.dual_variables['player_{}'.format(i)])

            self.theta['player_{}'.format(i)] = self.projection(self.dual_variables['player_{}'.format(i)])

        return self.get_state(), deepcopy(self.theta), {'all_posterior': self.all_posterior, 'posterior': self.posterior, 'likelihood': self.likelihood}
    

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