import numpy as np
import os
import argparse
import pickle
from stable_baselines3 import SAC, PPO
import gymnasium as gym
from LogClass import Logger
# from visualize import plot_in_one_figure, plot_total_payment
from GameInfo import TwoPlayerGames
import matplotlib.pyplot as plt
from copy import deepcopy
from utils import compute_uniform_initial_policies

def main(args):
    game = args.game
    obj = args.obj
    T = args.T
    lr = args.lr

    if args.game in TwoPlayerGames.keys():
        target_state = TwoPlayerGames[args.game]['target_state']
    else:
        raise NotImplementedError

    # Register Env
    gym.envs.register(
        id = 'MetaEnv',
        entry_point='MetaEnv:Meta_Environment',
        kwargs={
            'game': game,
            'obj': obj,
            'max_steer_reward': args.max_steer_reward,
            'max_obs': 1000,
            'T': T,
            'lr': lr,
            'beta': args.beta,
            'target_state': target_state,
            'gamma':0.99,
            'init_clip_threshold': args.init_clip_threshold,
            'act_dim': args.act_dim,

            'model_type': 'Normal',
            'eval_mode': True,
        },
    )

    env = gym.make('MetaEnv')

    log_prefix = './data/{}_{}_{}/beta_{}/trajectory_seed{}'.format(game, obj, args.model_type, args.beta, args.model_seed)
    if not os.path.exists(log_prefix):
        os.makedirs(log_prefix)

    initializations = compute_uniform_initial_policies(grid_number=args.grid_number)

    if args.game == 'StagHunt40':
        xlabel = r'$\pi^1(\text{H})$'
        ylabel = r'$\pi^2(\text{H})$'
    else:
        xlabel = r'$\pi^1(\text{Head})$'
        ylabel = r'$\pi^2(\text{Head})$'

    '''
    Plots for without steering
    '''
    # without steering
    log_path = log_prefix + '/{}_withoutSteering.pickle'.format(game)
    if not os.path.exists(log_path):
        log = Logger(num_players=2)
        # for i in range(args.num_eval):
        for i in range(len(initializations)):
            obs = env.reset(options=initializations[i])
            while True:
                obs, rew, done, _, _ = env.step(np.zeros([args.act_dim * 2]))
                if done:
                    log.record(env.get_trajectory())
                    break
        log.save(log_path)
    plt.figure(figsize=(10, 10))
    plot_trajectory(log_prefix + '/{}_withoutSteering.pickle'.format(game), title='No Steering', xlabel=xlabel, ylabel=ylabel)
    plt.savefig(log_prefix + '/{}_withoutSteering.pdf'.format(game), bbox_inches='tight')
    plt.close()

    prefix = './model/{}_{}_{}/seed_{}'.format(game, obj, args.model_type, args.model_seed)
    model_name = prefix + "/{}_steering_K{}_beta{}_T{}_lr{}".format(args.algo, args.K, args.beta, T, lr)

    model_name += '_TD'

    log_path = log_prefix + "/{}_{}_{}_steering_K{}_beta{}_T{}_lr{}_seed{}.pickle".format(args.game, args.obj, args.algo, args.K, args.beta, T, lr, args.model_seed)
    figure_path = log_prefix + '/{}_withSteering.pdf'.format(game)
    if not os.path.exists(log_path):
        evaluate(env, model_name=model_name, log_path=log_path, initializations=initializations)
    plt.figure(figsize=(10, 10))

    plot_trajectory(log_path, title='With Steering', xlabel=xlabel, ylabel=ylabel)

    # plt.savefig(figure_path + '.png', bbox_inches='tight')
    plt.savefig(figure_path, bbox_inches='tight')
    plt.close()

    print('save to ', log_path)


def evaluate(env, model_name, log_path, initializations):
    log = Logger(num_players=2)
    if 'PPO' in model_name:
        model = PPO.load(model_name)
    elif 'SAC' in model_name:
        model = SAC.load(model_name)
    else:
        raise NotImplementedError
    
    for i in range(len(initializations)):
        obs = env.reset(options=initializations[i])[0]
        while True:
            action, _state = model.predict(obs, deterministic=True)
            prev_obs = obs
            obs, reward, done, _, info = env.step(action)
            # print(obs)
            if done:
                log.record(env.get_trajectory())
                break
    log.save(log_path)


def plot_trajectory(path, title=None, xlabel=None, ylabel=None):
    with open(path, 'rb') as f:
        log = pickle.load(f)
    
    data = log.data
    init_x, init_y = [], []
    total_return = []
    for i in range(len(data)):
        d = data[i]
        traj_x, traj_y = [], []
        for h in d['horizon']:
            traj_x.append(d['player_1']['policy'][h-1][0][0])
            traj_y.append(d['player_2']['policy'][h-1][0][0])
        init_x.append(traj_x[0])
        init_y.append(traj_y[0])
        plt.plot(traj_x, traj_y, color='r', linewidth=0.1, alpha=0.3)

        total_return.append(sum(d['player_1']['payment']) + sum(d['player_2']['payment']))

    s = 10.0
    plt.scatter(init_x, init_y, color='black', s=s)
    plt.xlim([0,1])
    plt.ylim([0,1])

    fontsize = 50

    if xlabel is None:
        xlabel = r'$\pi^1(a_1)$'
    if ylabel is None:
        ylabel = r'$\pi^2(a_1)$'
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize, ticks=[0.0, 0.5, 1.0], labels=[0.0, 0.5, 1.0])
    plt.yticks(fontsize=fontsize, ticks=[0.5, 1.0], labels=[0.5, 1.0])
    if title:
        plt.title(title, fontsize=fontsize)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type = str, nargs='+', help='what game to run?')
    parser.add_argument('--obj', type = str, default = 'Nash', choices=['Nash', 'MaxUtility', 'MinGap'], 
                        help='what is the objective? "Nash" means steering to a fixed (Nash) policy; "MaxUtility" means steering to maximize the total utility')
    parser.add_argument('--grid-number', type = int, default = 5, help='number of grids to plot')

    parser.add_argument('-K', type = int, default=None, help='training iteration')
    parser.add_argument('--num-eval', type = int, default = 100, help='number of evaluations')
    parser.add_argument('--model-seed', type = int, default = 0, help='random seed for trained models')

    parser.add_argument('-T', type = int, default = 500, help='total time step')
    parser.add_argument('--lr', type = float, default = 0.01, help='agent learning rate')
    parser.add_argument('--time-interval', type = float, default = 0.01, help='the actual time each steering step corresponding to')
    parser.add_argument('--init-clip-threshold', type = float, nargs='+', default = 1e-8, help='clipping threshold for initialization')

    parser.add_argument('--algo', type = str, default = 'PPO', choices=['PPO', 'SAC', 'Pontryagin'], help='training algorithm')
    parser.add_argument('--beta', type = float, nargs='+', default = 0.0, help='weights on distance loss')
    
    parser.add_argument('--max-steer-reward', type = float, default = 10., help='the maximal steering reward')
    
    parser.add_argument('--act-dim', type = int, default = 2, help='action dimension')
    parser.add_argument('--obs-dim', type = int, default = 2, help='obs dimension')
    parser.add_argument('--model-type', default='Normal', type=str)

    args = parser.parse_args()
 
    return args


if __name__ == '__main__':
    args = get_parser()

    for game in args.game:
        for beta in args.beta:
            args_copy = deepcopy(args)
            args_copy.beta = beta
            args_copy.game = game
            main(args_copy)