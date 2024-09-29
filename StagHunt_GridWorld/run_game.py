import torch
import os
import json
from meta_env import MetaEnv
from steer_strategy import SteerStrategyMemory, PPO
import argparse
import pickle
import numpy as np
import random


parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", type=str, default="exp")
parser.add_argument("--game", type=str, default='StagHunt')
parser.add_argument('--device', default='cpu', type=str, help='which device to use?')
parser.add_argument('--seed', type = int, default = 0, help='which seed to use?')

parser.add_argument("--max-episodes", type=int, default=80, help='training episodes for steering strategy')
parser.add_argument("--max-ep-len", type=int, default=15, help='trajectory length of the steering dynamics')
parser.add_argument("--steer-epochs-per-update", type=int, default=80, help='number of optimization for steering strategy per PPO update')
parser.add_argument("--train-batch-size", type=int, default=128, help='batch size used for training steering strategy')

parser.add_argument("--beta", type=float, default=1.0, help='regularizatoin weights')
parser.add_argument("--max-steer-rew", type=float, default=10.0, help='maximal steering reward')
parser.add_argument("--lr", type=float, default=0.001, help='learning rate for steering strategy')

parser.add_argument("--save-freq", type=int, default=5, help='save frequency')

# set ups for agents
parser.add_argument("--agent-lr", type=float, default=0.005, help='agents learning rate')
parser.add_argument("--agents-epochs-per-update", type=int, default=20, help='number of optimization for inner agents per MetaEnv.step()')
parser.add_argument("--inner-batch-size", type=int, default=256, help='batch size used for training agents')
parser.add_argument("--inner-max-ep-len", type=int, default=16, help='trajectory length of StagHunt games')


args = parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


if __name__ == "__main__":
    ############## Hyperparameters ##############
    set_seed(args.seed)

    assert args.beta > 0

    # here we only use the trajectory information as the state dimension
    state_dim = [4 * args.inner_max_ep_len * args.inner_batch_size, 3, 3]

    # rew_stag_together, rew_red_stag_alone, rew_blue_stag_alone, rew_rabbit_together, rew_red_rabbit_alone, rew_blue_rabbit_alone
    action_dim = 8
    n_latent_var = 16   # number of variables in hidden layer

    num_steps = args.train_batch_size * args.inner_max_ep_len

    name = args.exp_name + '_StagHunt/seed_{}/BS{}_weight{}_updates{}_MaxEpLen{}_lr{}'.format(args.seed, args.train_batch_size, args.beta, args.steer_epochs_per_update, args.max_ep_len, args.lr)
    print(f"RUNNING NAME: {name}")
    if not os.path.isdir(name):
        os.makedirs(name)
        with open(os.path.join(name, "commandline_args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)
    #############################################

    if not torch.cuda.is_available():
        assert args.device == 'cpu'

    memory = SteerStrategyMemory()
    steering_strategy = PPO(
                  state_dim, 
                  action_dim, 
                  n_latent_var,
                  args.train_batch_size, 
                  max_ep_len=args.max_ep_len,
                  hyper_parameters = {
                        'lr': args.lr,
                        'gamma': 0.99,  # discount factor
                        'epochs_per_update': args.steer_epochs_per_update, # optimize policy for K epochs per update (i.e. MetaEnv.step)
                        'eps_clip': 0.2,  # clip parameter for PPO
                        'inner_max_ep_len': args.inner_max_ep_len,
                        # 'betas': (0.9, 0.999),
                        # 'tau': 0.3,  # GAE
                },
                device=args.device,
                )
    
    print(sum(p.numel() for p in steering_strategy.policy_old.parameters() if p.requires_grad))
    
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    records = []

    agents_setup = {
        'input_shape': [4, 3, 3],
        'n_latent_var': 8,
        'lr': args.agent_lr,
        'gamma': 0.96,  # discount factor
        'epochs_per_update': args.agents_epochs_per_update,  # update policy for K epochs
        'eps_clip': 0.2,  # clip parameter for PPO
    }
    env = MetaEnv(state_dim,
                  action_dim,
                  args.inner_batch_size, 
                  args.inner_max_ep_len, 
                  num_updates=5,
                  max_ep_len=args.max_ep_len,
                  agents_setup=agents_setup,
                  beta=args.beta,
                  max_action=args.max_steer_rew,
                  device=args.device,
                  )


    episode_returns = []
    total_utilities = []
    steering_costs = []
    stag_ratios = []
    # training loop
    for i_episode in range(1, args.max_episodes + 1):
        torch.cuda.empty_cache()
        memory.clear_memory()
        running_reward = 0
        total_utility = 0
        steering_cost = 0
        stag_ratio = 0
        p1_num_opp, p2_num_opp, p1_num_self, p2_num_self = 0, 0, 0, 0

        num_cooperate_final = 0

        # collect batch of data
        for i in range(args.train_batch_size):
            if i % 10 == 0:
                print('i_episode ', i_episode)
                print('training iteration ', i)
            # here state is a batch of trajectories with shape (1, 16, 512, 4, 3, 3)
            state = env.reset()
            done = False

            counter = 0
            while not done:
                memory.states_traj.append(state)
                
                # here action is the steering reward
                # in the one-step transition, the red & blue agents will conduct several PPO updates in the environment with modified reward functions
                # reward returned here is the reward for steering agents

                with torch.no_grad():
                    action, log_prob = steering_strategy.policy_old.act(state.detach(), print_info=False)
                    memory.actions_traj.append(action)
                    memory.logprobs_traj.append(log_prob)
                    
                    action = env.clip_actions(action)

                state, reward, done, info = env.step(action.detach())

                # normalize reward
                reward = reward / args.max_ep_len


                running_reward += reward
                total_utility += info['total_utility']
                steering_cost += info['steering_cost']
                memory.rewards.append(reward)

                counter += 1
                if counter % 10 == 0 or done:
                    print(action)

                # when a trajectory complete, we record the behavior of agents
                if done:
                    break
                
            
            # only record the stag ratio at the last step of training
            stag_ratio += info['stag_ratio']
            # if do not use steering, just generate one training trajectory and terminates

            if info['num_stag_list'][-1] > 0.9:
                num_cooperate_final += 1

        episode_returns.append(running_reward.item() / args.train_batch_size)
        total_utilities.append(total_utility / args.train_batch_size)
        steering_costs.append(steering_cost / args.train_batch_size)
        stag_ratios.append(num_cooperate_final / args.train_batch_size)
        print('episode_returns ', episode_returns)
        print('total_utilities ', total_utilities)
        print('steering_costs ', steering_costs)
        print('stag_ratios', stag_ratios)

        # run PPO to train steering_strategy
        pi_loss, val_loss = steering_strategy.update(memory)

        print("=" * 50)

        record = {
                "episode": i_episode,
                "rew": running_reward,
                "total_utility": total_utilities[-1],
                "episode_return": episode_returns[-1],
                'steering_cost': steering_costs[-1],
                "pi_loss": pi_loss,
                "val_loss": val_loss,
                "stag_ratios": stag_ratios[-1],
            }

        for k in info.keys():
            if k not in record.keys():
                record[k] = info[k]

        records.append(record)

        if i_episode % args.save_freq == 0 or i_episode == args.max_episodes:
            steering_strategy.save(os.path.join(name, f"{i_episode}.pth"))
            with open(os.path.join(name, "records.pickle"), "wb") as f:
                pickle.dump(records, f)
            print(f"SAVING! {i_episode}")