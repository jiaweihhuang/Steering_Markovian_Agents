import numpy as np
import os
from stable_baselines3 import PPO
import gymnasium as gym
from GameInfo import GameInformation
import argparse
from utils import set_random_seed


def main(args):
    set_random_seed(args.seed)

    if args.game in GameInformation.keys():
        target_state = GameInformation[args.game]['target_state']
    else:
        target_state = None

    # Register Env
    assert args.model_type == 'Normal'
    # normal learning setting, correspond to the known model setting
    gym.envs.register(
        id = 'MetaEnv',
        entry_point='MetaEnv:Meta_Environment',
        kwargs={
            'game': args.game,
            'num_players': args.num_players,
            'obj': args.obj,
            'distance_type': args.distance_type,
            'max_steer_reward': args.max_steer_reward,
            'max_obs': 1000,
            'T': args.T,
            'lr': args.lr,
            'time_interval': args.time_interval,
            'beta': args.beta,
            'target_state': target_state,
            'init_clip_threshold': args.init_clip_threshold,
            'gamma':0.99,
            'act_dim': 2,
            'obs_dim': 2,       # obs_dim_per_player

            'model_type': args.model_type,
        },
    )
    env = gym.make('MetaEnv')

    prefix = './model/{}_{}_{}'.format(args.game, args.obj, args.model_type)
    prefix += '/seed_{}'.format(args.seed)

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    model_save_path = prefix + '/PPO_steering_K{}_beta{}_T{}_lr{}'.format(args.K, args.beta, args.T, args.lr)

    # when building the model, the model automatically clip the actions to env.action_space
    # although the dtype may be np.float64 instead of np.float32
    model = PPO("MlpPolicy", env, policy_kwargs={'net_arch': dict(pi=[256, 256], vf=[256, 256])}, verbose=1)
    env.set_model(model)

    print('start train')
    model.learn(total_timesteps=args.K, log_interval=4) 
    print('finished train')
    model.save(model_save_path)


def get_parser():
    parser = argparse.ArgumentParser()
    # 'StagHunt', 'ZeroSum', 'Cooperative', 'PD' 
    parser.add_argument('--game', type = str, default = '', help='what game to run?')
    
    parser.add_argument('--obj', type = str, default = 'Nash', choices=['Nash', 'MaxUtility', 'Explore', 'MinGap', 'Explore_Success'], 
                        help='what is the objective? "Nash" means steering to a fixed (Nash) policy; "MaxUtility" means steering to maximize the total utility')
    
    parser.add_argument('--distance-type', type = str, default = 'policy', choices=['policy', 'dual_variable'], help='how to compute the distance')
    
    parser.add_argument('-K', type = int, default = 3000000, help='training iteration')
    parser.add_argument('--seed', type = int, default = 0, help='random seed')
    parser.add_argument('-T', type = int, default = 500, help='total time step')
    parser.add_argument('--lr', type = float, default = 0.01, help='agent learning rate')
    parser.add_argument('--time-interval', type = float, default = 0.01, help='the actual time each steering step corresponding to')
    parser.add_argument('--init-clip-threshold', type = float, nargs='+', default = 0.01, help='clipping threshold for initialization')
    parser.add_argument('--num-players', type = int, default = 2, help='number of players')

    parser.add_argument('--algo', type = str, default = 'PPO', help='training algorithm')
    parser.add_argument('--beta', type = float, default = 0.0, help='weights on distance loss')
    parser.add_argument('--max-steer-reward', type = float, default = 10., help='the maximal steering reward')
    
    parser.add_argument('--model-type', default='Normal', choices=['Normal'], type=str)

    parser.add_argument('--shift', type = float, default = 5., help='total time')

    args = parser.parse_args()
 
    return args


if __name__ == '__main__':
    args = get_parser()
    main(args)