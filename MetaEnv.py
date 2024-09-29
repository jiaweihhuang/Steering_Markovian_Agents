import numpy as np
import gymnasium as gym
import torch
from copy import deepcopy
from gymnasium import spaces
from GameInfo import GameInformation
from GameDynamics import TwoPlayersGame, MP_CooperativeGame
from PolicyMirrorFlowAvaricious import PMF_Avaricious
from PolicyMirrorFlowGaussian import PMF_Gaussian
from PolicyMirrorFlowDeterministic import PMF_Deterministic
from PolicyMirrorFlowAvariciousFixedShift import PMF_Avaricious_FS
from collections import deque

# discrete version of the continuous Markov Decision Process
class Meta_Environment(gym.Env):
    def __init__(self, 
                 game, 
                 obj,
                 max_steer_reward,    # range of steering reward
                 max_obs,               # range of observation
                 T,                     # continuous time of the dynamics
                 lr,                    # step size of the discrete version
                 time_interval,
                 beta,
                 target_state,
                
                 num_players=2,
                 gamma=0.99,
                 init_clip_threshold=1e-8,
                 beta_interval=100,
                 
                 obs_dim=2,
                 act_dim=2,

                 scale=1.0,
                 ada_beta=None,
                 epsilon=None,

                 mu=None,
                 eval_mu=None,
                 sigma=0.5,
                 model_type=None,
                 
                 shift_array=None,
                 random_explore=False,

                 distance_type='policy',
                 eval_mode=False,
                 ):
        

        if game in GameInformation.keys():
            assert num_players == 2
            self.game = TwoPlayersGame(game)
        else:
            if game == 'MP_Cooperative':
                self.game = MP_CooperativeGame(num_players=num_players)
            else:
                raise NotImplementedError
            target_state = self.game.target_state
        obs_dim = obs_dim * num_players + 1     # the additional dimension is for the time
        act_dim = act_dim * num_players
        
        self.obj = obj
        self.debug = False
        self.distance_type = distance_type
        self.save_path = None

        self.scale = scale
        self.ada_beta = ada_beta
        self.epsilon = epsilon if epsilon else 0.01
        self.beta_interval = beta_interval
        
        self.num_players = self.game.num_players
        self.gamma = gamma
        self.max_steer_reward = max_steer_reward
        self.max_obs = max_obs
        self.target_state = target_state

        self.model = None
        self.eval_mode = eval_mode

        self.T = T
        self.lr = lr
        self.time_interval = time_interval
        self.max_ep_len = T
        self.random_explore = random_explore


        self.model_type = model_type
        if self.model_type == 'Normal':
            self.model_set_size = 1
            self.dynamics = PMF_Deterministic(game=self.game, lr=self.lr, T=self.T, time_interval=self.time_interval,reg='neg_ent', 
                                init_clip_threshold=init_clip_threshold, 
                                obs_dim=obs_dim,
                            )
        elif self.model_type == 'Gaussian_lr':
            if type(mu) is list:
                self.model_set_size = len(mu)
            else:
                self.model_set_size = 1
            self.dynamics = PMF_Gaussian(game=self.game, lr=self.lr, T=self.T, time_interval=self.time_interval,reg='neg_ent', 
                                init_clip_threshold=init_clip_threshold, 
                                model_set_size=self.model_set_size,
                                mu=mu,
                                eval_mu=eval_mu,
                                sigma=sigma,
                                obs_dim=obs_dim,
                            )
        elif self.model_type == 'Avaricious':
            self.model_set_size = 1
            self.dynamics = PMF_Avaricious(game=self.game, lr=self.lr, T=self.T, time_interval=self.time_interval,reg='neg_ent',
                            init_clip_threshold=init_clip_threshold, 
                            shift_array=shift_array,
                            obs_dim=obs_dim,
                            sigma=sigma,
                            )
        elif self.model_type == 'Avaricious_FixedShift':
            self.model_set_size = 1
            self.dynamics = PMF_Avaricious_FS(game=self.game, lr=self.lr, T=self.T, time_interval=self.time_interval,reg='neg_ent',
                            init_clip_threshold=init_clip_threshold, 
                            shift_array=shift_array,
                            obs_dim=obs_dim,
                            sigma=sigma,
                            )
        else:
            raise NotImplementedError
        
        self.optimality_gap = [deque(maxlen=beta_interval) for _ in range(self.model_set_size)]

        if self.model_set_size == 1:
            if type(beta) is list:
                self.beta = beta[0]
            else:
                self.beta = beta
        else:
            if type(beta) is float:
                self.beta = [beta for _ in range(self.model_set_size)]
            else:
                assert type(beta) is list
                if len(beta) == 1:
                    self.beta = [beta[0] for _ in range(self.model_set_size)]
                elif len(beta) == self.model_set_size:
                    self.beta = beta
                else:
                    raise NotImplementedError

        self.theta_star = {}
        self.dual_diff_star = {}
        for k in self.target_state:
            self.theta_star[k] = self.dynamics.projection(self.target_state[k])
            target_dual = self.target_state[k].squeeze()
            self.dual_diff_star[k] = target_dual[0] - target_dual[1]

        self.theta_star_th = {}
        self.target_state_th = {}
        for k in self.theta_star.keys():
            self.theta_star_th[k] = torch.tensor(self.theta_star[k], requires_grad=False, dtype=torch.float64)
            self.target_state_th[k] = torch.tensor(self.target_state[k], requires_grad=False, dtype=torch.float64)
        

        self.para_sizes = {}
        self.S = 0
        for i in range(1, self.num_players + 1):
            para_size = self.game.S * self.game.A['player_{}'.format(i)]
            self.para_sizes['player_{}'.format(i)] = para_size
            self.S += para_size
        self.A = self.S

        ### construction obs and action space
        if self.model_set_size > 1:
            obs_dim = obs_dim + self.model_set_size
        obs_low = -np.ones([obs_dim]) * self.max_obs
        obs_high = -obs_low
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.act_dim = act_dim

        if self.act_dim == self.num_players:
            self.act_low = -np.ones([self.act_dim]) * self.max_steer_reward
            self.act_high = np.ones([self.act_dim]) * self.max_steer_reward
        else:
            self.act_low = np.zeros([self.act_dim])
            self.act_high = np.ones([self.act_dim]) * self.max_steer_reward

        self.action_space = spaces.Box(self.act_low, self.act_high, dtype=np.float32)

        self.num_trajs = [0 for _ in range(self.model_set_size)]

        ### other setup for data recording
        self.num_traj = 0
        self.success = []
        self.statistics = {
            'payments': {},
            'total_payments': {},
            'opt_gap': {},
        }

        self.best_accuracy = {}
        self.best_cost = {}
        for index in range(self.model_set_size):
            self.best_accuracy[index] = 0.
            self.best_cost[index] = np.inf

        self.explore_policy = None

        self.max_utility = self.game.compute_total_utility(self.theta_star)

        super().__init__()

    def save_model(self, save_path):
        print('save model to path ', save_path)
        self.model.save(save_path)

    def setup_exploration(self, explore_policy, explore_steps):
        self.explore_policy = explore_policy
        self.explore_steps = explore_steps
        # remove the exploration steps from the total steps
        self.max_ep_len = self.max_ep_len - explore_steps

    def set_model(self, model, save_path=None):
        self.model = model
        self.save_path = save_path

    def reset(
            self,
            *,
            seed=None,
            options=None,
            other_info=None,
        ):
        if options == 'return_statistics':
            return self.statistics
        if options == 'print_bp':
            print('best accuracy ever ', self.best_accuracy)
            print('best cost ever ', self.best_cost)
            return

        self.dynamics.reset(options=options, other_info=other_info)
        self.horizon = 0
        
        self.state = self.dynamics.get_state()
        self.internal_state = self.dynamics.get_policy()

        self.initial_internal_state = deepcopy(self.internal_state)
        self.num_traj += 1

        # self.trajectory is used to record the trajectory, keys include:
        # * Initialization: initial internal_state
        # * horizon: list of horizon
        # * data of each players
        self.trajectory = {
            'Initialization': self.internal_state,
            'horizon': [],
            'posterior': [],
            'total_payment': [],
            'total_return': [],
            'total_distance': [],
            'opt_gap': [],
            'avg_opt_gap': [],
            'utility': [],
        }
        # for data of each players, keys include
        # * policy: list of policies
        # * payment: list of payments
        # * dist: list of dist
        for i in range(1, self.num_players + 1):
            self.trajectory['player_{}'.format(i)] = {
                'policy': [],
                'payment': [],
                'dist': [],
            }

        other_info = {}
        if self.model_type == 'Avaricious':
            other_info['sampled_shift_index'] = self.dynamics.sampled_shift_index


        info = {'policies': deepcopy(self.internal_state), 'dual_variables': deepcopy(self.dynamics.dual_variables), 'other_info': other_info}

        return deepcopy(self.state.astype(np.float32)), info

    def compute_beta(self):
        belief_state = self.dynamics.get_belief_state()
        belief_beta = 0.
        for (bs, b) in zip(belief_state, self.beta):
            belief_beta += bs * b
        return belief_beta
    
    
    # return shape [1, -1]
    def sample_observation(self):
        return self.dynamics.sample_observation()


    '''
    the reward is defined to be the summation over:
    (1) total_payment * self.lr
        here we use self.lr as weights
    (2) distance between current policy and the target policy
    '''
    # action should be a vector with shape [-1]
    def step(self, action):
        action = np.clip(action * self.scale, self.act_low, self.act_high).astype(np.float32)
        assert self.action_space.contains(action), action

        self.horizon += 1

        self.steer_reward = self.action_to_steer_reward(action)
        self.state, policies, info = self.dynamics.update(self.steer_reward)
        payments_rates = self.dynamics.compute_payments_rates(self.steer_reward)
        next_internal_state = self.dynamics.get_policy()

        if self.distance_type == 'policy':
            distances = self.dynamics.compute_policy_distance(self.theta_star)
        else:
            distances = self.dynamics.compute_dual_distance(self.dual_diff_star)

        total_utility = self.dynamics.compute_utility()

        total_payment = 0
        total_distance = 0
        total_posterior = 0
        for i in range(1, self.num_players + 1):
            total_payment += payments_rates['player_{}'.format(i)] * self.time_interval
            total_distance += distances['player_{}'.format(i)]


            if self.model_type == 'Avaricious':
                total_posterior += np.log(info['posterior']['player_{}'.format(i)] + 1e-8)

            # log data into self.trajectory
            self.trajectory['player_{}'.format(i)]['policy'].append(self.internal_state['player_{}'.format(i)])
            self.trajectory['player_{}'.format(i)]['payment'].append(payments_rates['player_{}'.format(i)] * self.time_interval)
            self.trajectory['player_{}'.format(i)]['dist'].append(distances['player_{}'.format(i)])
        
        self.trajectory['utility'].append(total_utility)
        self.trajectory['horizon'].append(self.horizon)
        self.trajectory['posterior'].append(total_posterior)
        self.trajectory['total_payment'].append(total_payment)
        self.trajectory['total_distance'].append(total_distance)


        done = self.horizon >= self.max_ep_len # or total_distance < 1e-3

        rew_total_distance = total_distance

        gap = self.max_utility - total_utility

        if self.model_set_size > 1:
            beta = self.compute_beta()
        else:
            beta = self.beta
        
        reward = -total_payment 

        # if it is the last step, then use the terminal cost
        if done:
            if self.model_set_size > 1 or self.model_type == 'Avaricious':
                self.success.append(self.dynamics.is_success())
                opt_gap = gap
            elif self.obj == 'Explore':
                # reward = reward + 
                reward = total_posterior * self.lr * np.maximum(1.0, beta)
                opt_gap = 0.0
            elif self.obj == 'Nash':
                reward = reward - rew_total_distance * self.lr * np.maximum(1.0, beta)
                opt_gap = total_distance
            elif self.obj == 'MaxUtility':
                reward = reward + total_utility * self.lr * np.maximum(1.0, beta)
                opt_gap = gap
            elif self.obj == 'MinGap':
                reward = reward - gap * self.lr * np.maximum(1.0, beta)
                opt_gap = gap
            else:
                raise NotImplementedError          
            self.trajectory['opt_gap'].append(opt_gap)  
            self.trajectory['avg_opt_gap'].append(opt_gap / self.num_players)  
        # if this is not the last step, then use steering cost (possibly with penalty on distance or utility)
        elif beta > 0.0:
            if self.obj == 'Explore':
                reward = reward + total_posterior * self.lr * np.maximum(1.0, beta)
            elif self.obj == 'Nash':
                 reward = reward - rew_total_distance * self.lr * np.maximum(1.0, beta)
            elif self.obj == 'MaxUtility':
                reward = reward + total_utility * self.lr * np.maximum(1.0, beta)
            elif self.obj == 'MinGap':
                 reward = reward - gap * self.lr * np.maximum(1.0, beta)
            else:
                raise NotImplementedError
            

        self.trajectory['total_return'].append(reward)

        if done:
            if self.model_set_size == 1:
                model_index = 0
            else:
                bs = self.dynamics.get_belief_state()
                model_index = np.argmax(bs)
            self.optimality_gap[model_index].append(opt_gap)
            self.num_trajs[model_index] += 1

            if model_index not in self.statistics['opt_gap'].keys():
                self.statistics['opt_gap'][model_index] = []
                self.statistics['total_payments'][model_index] = []
            else:
                self.statistics['opt_gap'][model_index].append(np.mean(self.optimality_gap[model_index]))
                self.statistics['total_payments'][model_index].append(np.sum(self.trajectory['total_payment']))


            if self.num_traj % 100 == 0 and not self.eval_mode and self.save_path is not None:
                save_model_flag = True
                best_accuracy = deepcopy(self.best_accuracy)
                for model_index in range(self.model_set_size):
                    cur_accuracy = np.mean(np.array(self.optimality_gap[model_index]) < self.epsilon)
                    if cur_accuracy >= self.best_accuracy[model_index]:
                        save_model_flag = True
                        best_accuracy[model_index] = cur_accuracy
                    else:
                        save_model_flag = False
                
                if save_model_flag:
                    self.best_accuracy = best_accuracy

                if save_model_flag:
                    print('best cost', self.best_cost)
                    self.save_model(self.save_path)

            if self.model_type == 'Gaussian_lr' or self.model_type == 'Bernoulli_lr':
                if self.dynamics.sampled_mu not in self.statistics['payments'].keys():
                    self.statistics['payments'][self.dynamics.sampled_mu] = deque(maxlen=200)
                self.statistics['payments'][self.dynamics.sampled_mu].append(np.sum(self.trajectory['total_payment']))

            if self.num_traj % 100 == 0 and self.debug:
                print('Initial ', self.initial_internal_state)
                print('Final ', self.internal_state)
                print('dist', total_distance)
                print('utility', total_utility)
                print('posterior', total_posterior)
                print('steering cost', np.sum(self.trajectory['total_payment']))
                print('beta', self.beta, beta)
                for i in range(len(self.optimality_gap)):
                    print('accurate rate ', np.mean(np.array(self.optimality_gap[i]) < self.epsilon))

                if self.model_type == 'Avaricious':
                    print('Success rate ', np.mean(self.success[-100:]))
                
                if self.model_type == 'Avaricious' or self.model_type == 'Gaussian_lr' or self.model_type == 'Bernoulli_lr':
                    for k in self.statistics['payments'].keys():
                        print(k, np.mean(self.statistics['payments'][k]))

        if self.num_traj % 100 == 0 and self.debug:
            if self.horizon % 20 == 0:
                print('action ', action, 'internal_state ', self.internal_state, 'dist', total_distance, 'utility', total_utility, 'posterior', total_posterior)


        self.internal_state = next_internal_state

        other_info = {}
        if self.model_type == 'Avaricious':
            other_info['all_posterior'] = self.dynamics.all_posterior

        info = {'policies': policies, 'dual_variables': deepcopy(self.dynamics.dual_variables), 'total_payment': total_payment, 'payments_rates': payments_rates, 'posterior': total_posterior, 'optimality_gap': self.optimality_gap, 'other_info': other_info}

        return deepcopy(self.state.astype(np.float32)), deepcopy(reward.squeeze()), done, False, info
    
    
    def action_to_steer_reward(self, action):
        steer_rewards = {}
        if self.act_dim == self.num_players:
            for n in range(self.num_players):
                steer_rewards['player_{}'.format(n+1)] = np.array([np.maximum(action[n], 0), np.maximum(-action[n], 0)])
            return steer_rewards
        
        start = 0
        for i in range(1, self.num_players + 1):
            para_size = self.para_sizes['player_{}'.format(i)]
            
            steer_r_i = action[start: start + para_size]
            steer_rewards['player_{}'.format(i)] = steer_r_i.reshape([self.game.S, self.game.A['player_{}'.format(i)]])

            start += para_size
        return steer_rewards
    
    def fix_eval_mu(self, mu):
        self.dynamics.fix_eval_mu(mu)
    
    def get_trajectory(self):
        return deepcopy(self.trajectory)
    
    def get_dynamics_internal_state(self):
        return self.dynamics.get_internal_state()

    def render(self):
        pass

    def close(self):
        pass
    
    def set_debug(self, debug):
        assert 0 == 1
        self.debug = debug