import numpy as np
import scipy
import torch
from copy import deepcopy
from PolicyMirrorFlowBase import PMF_Base
from torch.distributions.normal import Normal

class PMF_Gaussian(PMF_Base):
    def __init__(self,
                 game, 
                 lr,
                 T,
                 time_interval=0.01,
                 init_clip_threshold=1e-8,
                 reg='neg_ent',
                 mu=None,
                 eval_mu=None,
                 sigma=None,
                 obs_dim=4,
                 model_set_size=None,
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

        self.sigma = sigma if sigma is not None else 0.1
        self.model_set_size = model_set_size

        if type(mu) is list:
            self.mu = np.array(mu)
            self.distributions = []
            for mu_ in self.mu:
                self.distributions.append(
                    Normal(loc=mu_, scale=sigma)
                )
        else:
            self.mu = mu
        
        self.eval_mu = eval_mu
        self.dual_variables = None

        self.reset()

    def initialization(self):
        return self.random_initialization()

    def is_success(self):
        return self.sampled_mu_index == np.argmax(self.belief_state)
    
    def get_belief_state(self):
        return self.belief_state
    
    def update_belief_state(self, sample):
        prob = np.array([
            distri.log_prob(torch.tensor(sample)).exp().numpy() for distri in self.distributions
        ])
        p = self.belief_state * prob
        self.belief_state = p / np.sum(p)

    def reset(self, options=None, other_info=None):
        self.step_num = 0

        # sample a mu from the list
        if type(self.mu) is float:
            self.sampled_mu = self.mu
        else:
            if self.eval_mu is not None:
                self.sampled_mu_index = np.argwhere(self.eval_mu == self.mu)[0][0]
            else:
                self.sampled_mu_index = np.random.randint(len(self.mu))
            self.sampled_mu = self.mu[self.sampled_mu_index]
        
        # if there is only one model, we do not need to update the belief state
        if self.model_set_size > 1:
            self.belief_state = np.ones([self.model_set_size]) / self.model_set_size
        else:
            self.belief_state = None

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

    def sample_learning_rate_fixed_lam(self):
        scale = np.random.randn() * self.sigma + self.sampled_mu
        if scale > 0:
            return scale
        else:
            return 0.0
        
    # steer_r is a dict with 'player_{i}' as keys, and vector with shape [S, Ai] as values
    def update(self, steer_r):
        self.step_num += 1

        Q = self.game.compute_Q_function(self.theta, steer_r)

        for i in range(1, self.num_players + 1):
            sampled_scale = self.sample_learning_rate_fixed_lam()
            self.dual_variables['player_{}'.format(i)] += self.lr * sampled_scale * Q['player_{}'.format(i)]
            self.dual_variables['player_{}'.format(i)] = self.dual_variables['player_{}'.format(i)] - np.mean(self.dual_variables['player_{}'.format(i)])
            self.theta['player_{}'.format(i)] = self.projection(self.dual_variables['player_{}'.format(i)])
        
        if self.belief_state is not None:
            self.update_belief_state(sampled_scale)

        return self.get_state(), deepcopy(self.theta), {'all_posterior': None, 'posterior': None, 'likelihood': None}
    

    # the state in this case is the dual variables, which preserve the full information
    def get_state(self):
        state = []

        for i in range(1, self.num_players + 1):
            dual_variable = self.dual_variables['player_{}'.format(i)].reshape([-1])
            state.append(dual_variable)

        if self.model_set_size > 1:
            state.append(self.belief_state)

        state.append(
            np.array([(self.T - self.step_num) * self.time_interval])
        )

        return np.concatenate(state)
    

    def fix_eval_mu(self, mu):
        self.eval_mu = mu