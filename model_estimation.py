import numpy as np
import argparse
import gymnasium as gym
from GameInfo import GameInformation
import pickle
from stable_baselines3 import SAC, PPO


def main(args):
    # args.explore_policy_path = './model/MP_Cooperative_Explore_ValueAware_Greedy/seed_500/PPO_steering_K5000000_beta100.0_T30_lr0.01_shift[0.0, -0.25, -0.75]_sigma0.5'
    if args.explore_policy_path:
        explore_policy = PPO.load(args.explore_policy_path)
    else:
        explore_policy = None

    if args.game in GameInformation.keys():
        target_state = GameInformation[args.game]['target_state']
    else:
        target_state = None

    num_players = args.num_players
    env_list = []
    fixed_shift = args.fixed_shift
    
    for T in args.T:
        env_id = 'MetaEnv_T{}'.format(T)
        gym.envs.register(
            id = env_id,
            entry_point='MetaEnv:Meta_Environment',
            kwargs={
                'game': args.game,
                'num_players': args.num_players,
                'obj': 'Explore',
                'max_steer_reward': args.max_steer_reward,
                'max_obs': 1000,
                'T': T,
                'lr': args.lr,
                'time_interval': args.time_interval,
                'beta': args.beta,
                'target_state': target_state,
                'init_clip_threshold': args.init_clip_threshold,
                'gamma':0.99,
                'obs_dim': args.obs_dim,
                'act_dim': args.act_dim,  

                'shift_array': np.array(args.shift),
                'model_type': 'Avaricious',
                'sigma': args.sigma,
            },
        )

        env = gym.make(env_id)
        env_list.append(env)

        predicted_results = []
        for i in range(args.num_eval):
            obs, info = env.reset(other_info={'fixed_shift': fixed_shift})
            sampled_shift_index = info['other_info']['sampled_shift_index']

            done = False
            while not done:
                if explore_policy:
                    action, _state = explore_policy.predict(obs, deterministic=True)
                else:
                    # uniform noise
                    action = env.action_space.sample()

                obs, rew, done, _, info = env.step(action)

            all_posterior = info['other_info']['all_posterior']
            predicted_result = {}
            for n in range(1, num_players + 1):
                posterior = all_posterior['player_{}'.format(n)]
                winner_index = np.argwhere(posterior == np.amax(posterior))
                if len(winner_index) > 1:
                    winner_index = np.array(winner_index).squeeze()
                    index = np.random.choice(winner_index)
                else:
                    index = np.array(winner_index).squeeze()
                predicted_result['player_{}'.format(n)] = index
            predicted_results.append(predicted_result)
            print('Trial {}:\nPrediction is {}; \nThe ground truth is {}'.format(i, predicted_result, sampled_shift_index))


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--game', type = str, default = '', help='what game to run?')
    parser.add_argument('--num-eval', type = int, default = 100, help='training iteration')
    parser.add_argument('--seed', type = int, default = 0, help='random seed')
    parser.add_argument('-T', type = int, default = 500, nargs='+', help='total time step')
    parser.add_argument('--lr', type = float, default = 0.01, help='agent learning rate')
    parser.add_argument('--time-interval', type = float, default = 0.01, help='the actual time each steering step corresponding to')
    parser.add_argument('--init-clip-threshold', type = float, nargs='+', default = 0.01, help='clipping threshold for initialization')
    parser.add_argument('--num-players', type = int, default = 10, help='number of players')

    parser.add_argument('--beta', type = float, default = 0.0, nargs='+', help='weights on distance loss')
    parser.add_argument('--max-steer-reward', type = float, default = 1., help='the maximal steering reward')
    
    parser.add_argument('--shift', nargs='+', type = float, default = [0., -0.25, -0.75], help='the shift term')
    parser.add_argument('--fixed-shift', default = [0.0] * 10, type = float, nargs='+', help='the hidden shift by all agents; if provided, we will use it as the true model to fix the ')

    parser.add_argument('--act-dim', type = int, default = 2, help='action dimension')
    parser.add_argument('--obs-dim', type = int, default = 2, help='obs dimension per agent')

    parser.add_argument('--mu', default = None, type = float, nargs='+', help='the mu list')
    parser.add_argument('--sigma', default = 0.5, type = float, help='the sigma list')

    parser.add_argument('--explore-policy-path', type = str, default = None, help='exploration policy path')

    args = parser.parse_args()
 
    return args


if __name__ == '__main__':
    args = get_parser()
    main(args)