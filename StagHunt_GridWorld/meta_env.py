import torch
from StagHunt_Env import StagHuntGameGPU
from ppo_agent import PPO, Memory


'''
MetaEnv: agents' parameter updates (PPO) as state transition
'''
class MetaEnv:
    def __init__(self, 
                 state_dim,
                 action_dim,
                 inner_batch_size, 
                 inner_ep_len,
                 num_updates=1,
                 max_ep_len=100,    # this corresponds to the number of training steps of the PPO updates
                 agents_setup=None,
                 beta=1.0,
                 min_action=0.0,
                 max_action=10.0,
                 device=None,
                ):
        self.env = StagHuntGameGPU(max_steps=inner_ep_len, batch_size=inner_batch_size, device=device)
        
        self.inner_ep_len = inner_ep_len
        self.inner_batch_size = inner_batch_size
        self.num_updates = num_updates
        self.max_ep_len = max_ep_len
        self.beta = beta

        assert device is not None
        self.device = device

        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.min_action = min_action
        self.max_action = max_action

        """
        Hyperparams for inner PPO
        """
        assert agents_setup is not None

        self.agent_action_dim = self.env.NUM_ACTIONS
        self.agent_input_shape = agents_setup['input_shape']
        self.agent_n_latent_var = agents_setup['n_latent_var']
        self.agent_lr = agents_setup['lr']
        self.agent_gamma = agents_setup['gamma'] # discount factor
        self.agent_epochs_per_update = agents_setup['epochs_per_update']  # optimize policy for K epochs per 'update'
        self.agent_eps_clip = agents_setup['eps_clip']  # clip parameter for PPO


    def reset(self):
        # set up for two inner agents
        self.inner_agent_1 = PPO(self.agent_input_shape, self.agent_action_dim, self.agent_n_latent_var, self.agent_lr, self.agent_gamma, self.agent_epochs_per_update, self.agent_eps_clip, max_ep_len=self.inner_ep_len, device=self.device)
        self.inner_memory_1 = Memory()

        self.inner_agent_2 = PPO(self.agent_input_shape, self.agent_action_dim, self.agent_n_latent_var, self.agent_lr, self.agent_gamma, self.agent_epochs_per_update, self.agent_eps_clip, max_ep_len=self.inner_ep_len, device=self.device)
        self.inner_memory_2 = Memory()

        self.t = 0

        self.env_states = self.env.reset()
        inner_running_reward_1 = 0
        inner_running_reward_2 = 0

        self.num_stag_list = []
        self.total_utility_list = []
        self.num_rabbit_list = []

        trajectory = []
        for _ in range(self.inner_ep_len):
            trajectory.append(self.env_states)
            with torch.no_grad():
                self.inner_actions_1 = self.inner_agent_1.policy_old.act(self.env_states, self.inner_memory_1)
                self.inner_actions_2 = self.inner_agent_2.policy_old.act(self.env_states, self.inner_memory_2)

                self.env_states, (self.inner_rewards_1, self.inner_rewards_2), self.inner_dones, info = self.env.step([self.inner_actions_1, self.inner_actions_2])
                
                inner_running_reward_1 += self.inner_rewards_1
                inner_running_reward_2 += self.inner_rewards_2

                self.inner_memory_1.rewards.append(self.inner_rewards_1.detach())
                self.inner_memory_2.rewards.append(self.inner_rewards_2.detach())

        self.observation = torch.stack(trajectory, dim=0).reshape([1] + self.state_dim)
        return self.observation
    

    def clip_actions(self, actions):
        return torch.clip(actions, self.min_action, self.max_action)

    def step(self, actions):
        self.t += 1
        actions = self.clip_actions(actions)
        rew_red_stag_together, rew_blue_stag_together, rew_red_stag_alone, rew_blue_stag_alone, \
                rew_red_rabbit_together, rew_blue_rabbit_together, rew_red_rabbit_alone, rew_blue_rabbit_alone = actions.squeeze()

        self.inner_rew_means = []
        observation = []
        # for this MetaEnv, one step corresponds to multiple updates of the two inner agents with PPO, 
        # where each update includes (1) collecting trajectories and (2) updating with PPO
        # we can also consider update once per step, however, it may result in long horizon, since the PPO may takes long time to converge
        for i in range(self.num_updates + 1):
            is_eval_step = i == self.num_updates

            self.inner_memory_1.clear_memory()
            self.inner_memory_2.clear_memory()

            self.env_states = self.env.reset()
            inner_running_reward_1 = 0
            inner_running_reward_2 = 0
            total_steering_cost_1 = 0
            total_steering_cost_2 = 0

            num_stag_together, num_rabbit_together, num_red_rabbit_alone, num_blue_rabbit_alone, num_red_stag_alone, num_blue_stag_alone = 0, 0, 0, 0, 0, 0
            
            total_init = 0

            # Part 1: collect batch of trajectories
            for _ in range(self.inner_ep_len):
                with torch.no_grad():
                    if is_eval_step:
                        observation.append(self.env_states)

                    # in this act function, the memory is updated
                    self.inner_actions_1 = self.inner_agent_1.policy_old.act(self.env_states, self.inner_memory_1)
                    self.inner_actions_2 = self.inner_agent_2.policy_old.act(self.env_states, self.inner_memory_2)

                    self.env_states, (self.inner_rewards_1, self.inner_rewards_2), self.inner_dones, info = self.env.step([self.inner_actions_1, self.inner_actions_2])
                    
                    inner_running_reward_1 += self.inner_rewards_1.detach()
                    inner_running_reward_2 += self.inner_rewards_2.detach()
                    
                    # set up steering rewards
                    stag_together = info['stag_together']
                    red_stag_alone = info['red_stag_alone']
                    blue_stag_alone = info['blue_stag_alone']

                    rabbit_together = info['rabbit_together']
                    red_rabbit_alone = info['red_rabbit_alone']
                    blue_rabbit_alone = info['blue_rabbit_alone']
                    
                    # add the steering rewards on the top of the original rewards
                    steering_rewards_1 = stag_together * rew_red_stag_together + rabbit_together * rew_red_rabbit_together \
                        + red_rabbit_alone * rew_red_rabbit_alone + red_stag_alone * rew_red_stag_alone
                    steering_rewards_2 = stag_together * rew_blue_stag_together + rabbit_together * rew_blue_rabbit_together \
                        + blue_rabbit_alone * rew_blue_rabbit_alone + blue_stag_alone * rew_blue_stag_alone

                    num_stag_together += stag_together.sum()
                    num_red_stag_alone += red_stag_alone.sum()
                    num_blue_stag_alone += blue_stag_alone.sum()

                    num_rabbit_together += rabbit_together.sum()
                    num_red_rabbit_alone += red_rabbit_alone.sum()
                    num_blue_rabbit_alone += blue_rabbit_alone.sum()

                    total_init += info['should_reinit'].sum()


                    self.inner_rewards_1 += steering_rewards_1
                    self.inner_rewards_2 += steering_rewards_2
                    total_steering_cost_1 += steering_rewards_1.detach()
                    total_steering_cost_2 += steering_rewards_2.detach()
                    self.inner_memory_1.rewards.append(self.inner_rewards_1.detach())
                    self.inner_memory_2.rewards.append(self.inner_rewards_2.detach())

            # inner trajectory should have finished before doing the updates
            assert self.inner_dones.mean() > 0

            # Part 2: conduct PPO updates
            # for the last iteration, do not update but just collect data, which will be used as observations for steer strategies to infer agents' hidden states
            if is_eval_step:
                info = {
                    "t": self.t,
                    "rew_1": inner_running_reward_1.mean().item(),
                    "rew_2": inner_running_reward_2.mean().item(),
                    "steer_cost_1": total_steering_cost_1.mean().item(),
                    "steer_cost_2": total_steering_cost_2.mean().item(),

                    "num_stag_together": num_stag_together.float().mean().item(),
                    "num_red_stag_alone": num_red_stag_alone.float().mean().item(),
                    "num_blue_stag_alone": num_blue_stag_alone.float().mean().item(),

                    "num_rabbit_together": num_rabbit_together.float().mean().item(),
                    "num_red_rabbit_alone": num_red_rabbit_alone.float().mean().item(),
                    "num_blue_rabbit_alone": num_blue_rabbit_alone.float().mean().item(),

                    "total_init": total_init.float().mean().item(),
                }
            else:
                self.inner_agent_1.update(self.inner_memory_1)
                self.inner_agent_2.update(self.inner_memory_2)

        steering_cost = total_steering_cost_1.mean().item() + total_steering_cost_2.mean().item()
        total_utility_1 = inner_running_reward_1.detach().mean().item()
        total_utility_2 = inner_running_reward_2.detach().mean().item()
        total_utility = total_utility_1 + total_utility_2

        self.num_stag_list.append(num_stag_together.float().item() / total_init.float().item())
        self.total_utility_list.append(total_utility)
        self.num_rabbit_list.append(num_rabbit_together.float().item() / total_init.float().item())

        done = self.t >= self.max_ep_len
        self.observation = torch.stack(observation).reshape([1] + self.state_dim)

        self.rewards = torch.tensor(total_utility * self.beta - steering_cost)
        
        info['total_utility'] = total_utility
        info['steering_cost'] = steering_cost
        info['rewards'] = self.rewards
        info['stag_ratio'] = num_stag_together / total_init

        info['num_stag_list'] = self.num_stag_list
        info['total_utility_list'] = self.total_utility_list
        info['num_rabbit_list'] = self.num_rabbit_list

        return self.observation, self.rewards, done, info


    def save_states(self, path):
        self.inner_agent_1.save(path + '_agent1.pth')
        self.inner_agent_2.save(path + '_agent2.pth')