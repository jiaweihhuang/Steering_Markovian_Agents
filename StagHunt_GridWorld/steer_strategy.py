import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class SteerStrategyMemory:
    def __init__(self):
        self.actions_traj = []
        self.states_traj = []
        self.logprobs_traj = []
        self.rewards = []

    def clear_memory(self):
        del self.actions_traj[:]
        del self.states_traj[:]
        del self.logprobs_traj[:]
        del self.rewards[:]


class ActorCriticSteerStrategy(nn.Module):
    def __init__(self, input_shape, action_dim, n_out_channels, batch_size):
        super(ActorCriticSteerStrategy, self).__init__()
        self.batch_size = batch_size
        self.n_out_channels = n_out_channels
        self.space = n_out_channels
        self.action_dim = action_dim
        self.input_shape = input_shape

        self.conv_a_0 = nn.Conv2d(input_shape[0], n_out_channels, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.conv_a_1 = nn.Conv2d(n_out_channels, n_out_channels, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.linear_a_0 = nn.Linear(n_out_channels * input_shape[1] * input_shape[2], self.space)

        self.linear_a = nn.Linear(self.space, action_dim * 2)

        self.conv_v_0 = nn.Conv2d(input_shape[0], n_out_channels, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.conv_v_1 = nn.Conv2d(n_out_channels, n_out_channels, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.linear_v_0 = nn.Linear(n_out_channels * input_shape[1] * input_shape[2], self.space)

        self.linear_v = nn.Linear(self.space, 1)

        self.conv_t_0 = nn.Conv2d(input_shape[0], n_out_channels, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.conv_t_1 = nn.Conv2d(n_out_channels, n_out_channels, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.linear_t_0 = nn.Linear(n_out_channels * input_shape[1] * input_shape[2], self.space)
        self.linear_t = nn.Linear(self.space, self.space)
        

    def forward(self):
        raise NotImplementedError

    def infer(self, state_bs):
        x = self.conv_a_0(state_bs)
        x = F.relu(x)
        x = self.conv_a_1(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.linear_a_0(x)
        x = F.relu(x)
        x = self.linear_a(x)
        mu, log_sigma = torch.split(x, self.action_dim, dim=1)

        return mu, log_sigma

    def act(self, state_bs, deterministic=False, print_info=False):
        mu, log_sigma = self.infer(state_bs)

        if deterministic:
            action_b = mu.detach().clone()
        else:
            dist = Normal(loc=mu, scale=log_sigma.exp())
            action_b = dist.sample()

        if print_info:
            print('mean is ', mu, 'log_sigma is ', log_sigma)
        return action_b, dist.log_prob(action_b)

    def evaluate(self, state_Bs, action_Ttb):
        x = self.conv_a_0(state_Bs)
        x = F.relu(x)
        x = self.conv_a_1(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.linear_a_0(x)
        x = F.relu(x)

        x = self.linear_a(x)
        mu, log_sigma = torch.split(x, self.action_dim, dim=1)
        dist = Normal(loc=mu, scale=log_sigma.exp())
        action_logprobs = dist.log_prob(action_Ttb)
        dist_entropy = dist.entropy()

        x = self.conv_v_0(state_Bs)
        x = F.relu(x)
        x = self.conv_v_1(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.linear_v_0(x)
        x = F.relu(x)
        state_value = self.linear_v(x)

        return action_logprobs, state_value, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, batch_size, max_ep_len, hyper_parameters, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_latent_var = n_latent_var
        self.batch_size = batch_size
        self.max_ep_len = max_ep_len

        self.lr = hyper_parameters['lr']
        self.gamma = hyper_parameters['gamma']
        self.eps_clip = hyper_parameters['eps_clip']
        self.epochs_per_update = hyper_parameters['epochs_per_update']
        self.inner_max_ep_len = hyper_parameters['inner_max_ep_len']

        self.device = device

        self.policy = ActorCriticSteerStrategy(state_dim, action_dim, n_latent_var, batch_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.policy_old = ActorCriticSteerStrategy(state_dim, action_dim, n_latent_var, batch_size).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for t, reward in enumerate(reversed(memory.rewards)):
            if t != 0 and t % self.max_ep_len == 0:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.stack(rewards).to(self.device).detach()
        rewards = rewards.reshape([-1, 1])

        # convert list to tensor
        old_states = torch.stack(memory.states_traj).squeeze().to(self.device).detach()  # T,t,b,s
        del memory.states_traj[:]
        old_actions = torch.stack(memory.actions_traj).squeeze().to(self.device).detach()
        del memory.actions_traj[:]
        old_logprobs = torch.stack(memory.logprobs_traj).to(self.device).detach().flatten(end_dim=-2)
        del memory.logprobs_traj[:]

        del memory.rewards[:]
        # Optimize policy for K epochs:
        for i in range(self.epochs_per_update):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach()).squeeze()

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2)
            value_loss = 0.5 * self.MseLoss(state_values, rewards)
            loss = policy_loss + value_loss  # - 0.01*dist_entropy
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            if i % 5 == 0:
                print('loss at iteration {}'.format(i), policy_loss.mean().detach(), value_loss.mean().detach())

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        return policy_loss.mean().detach(), value_loss.mean().detach()

    def save(self, filename):
        torch.save(
            {
                "actor_critic": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            filename,
        )

    def load(self, filename):
        import os
        assert os.path.exists(filename), filename
        checkpoint = torch.load(filename)
        self.policy.load_state_dict(checkpoint["actor_critic"])
        self.policy_old.load_state_dict(checkpoint["actor_critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
