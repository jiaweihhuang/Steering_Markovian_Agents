import numpy as np
import argparse
import gymnasium as gym
from GameInfo import TwoPlayerGames
import pickle

def main(args):
    if args.game in TwoPlayerGames.keys():
        target_state = TwoPlayerGames[args.game]['target_state']
    else:
        target_state = None

    num_players = args.num_players
    env_list = []
    avg_sr_list = []
    conf_interval_list = []
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
                'act_dim': 1,   # Here we set action dim = 1, so that the steer reward for one action will not offest the other, to increase the success rate for random exploration

                'shift_array': np.array(args.shift),
                'model_type': args.model_type,
                'sigma': args.sigma,
            },
        )

        env = gym.make(env_id)
        env_list.append(env)


        num_failure = 0
        for i in range(args.num_eval):
            obs, info = env.reset()
            sampled_shift_index = info['other_info']['sampled_shift_index']

            done = False
            while not done:
                action = env.action_space.sample()
                obs, rew, done, _, info = env.step(action)

            all_posterior = info['other_info']['all_posterior']
            success = True
            for i in range(1, num_players + 1):
                index = np.argmax(all_posterior['player_{}'.format(i)])
                # print(i, index, sampled_shift_index['player_{}'.format(i)])
                if index != sampled_shift_index['player_{}'.format(i)]:
                    num_failure += 1
                    break
        
        avg_sr = 1.0 - num_failure / args.num_eval
        conf_interval = avg_sr * (1 - avg_sr) / np.sqrt(args.num_eval - 1)

        avg_sr_list.append(avg_sr)
        conf_interval_list.append(conf_interval)


    log = {}
    for i in range(len(args.T)):
        print('T = {}, Avg Success Rate is {} \pm {}'.format(args.T[i], avg_sr_list[i], conf_interval_list[i]))
        log[args.T[i]] = (avg_sr_list[i], conf_interval_list[i])

    with open('Explore_Random_Data.pickle', 'wb') as f:
        pickle.dump(log, f)

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--game', type = str, default = '', help='what game to run?')
    parser.add_argument('--num-eval', type = int, default = 100, help='training iteration')
    parser.add_argument('--seed', type = int, default = 0, help='random seed')
    parser.add_argument('-T', type = int, default = 500, help='total time step')
    parser.add_argument('--lr', type = float, default = 0.01, help='agent learning rate')
    parser.add_argument('--time-interval', type = float, default = 0.01, help='the actual time each steering step corresponding to')
    parser.add_argument('--init-clip-threshold', type = float, nargs='+', default = 0.01, help='clipping threshold for initialization')
    parser.add_argument('--num-players', type = int, default = 2, help='number of players')

    parser.add_argument('--beta', type = float, default = 0.0, nargs='+', help='weights on distance loss')
    parser.add_argument('--max-steer-reward', type = float, default = 2., help='the maximal steering reward')
    
    parser.add_argument('--shift', nargs='+', type = float, default = [0.], help='the shift term')

    parser.add_argument('--act-dim', type = int, default = 1, help='action dimension')
    parser.add_argument('--obs-dim', type = int, default = 2, help='obs dimension per agent')

    parser.add_argument('--model-type', default=None, type=str) #, choices=['Normal', 'Gaussian_lr', 'ValueAware'], help='model type')
    parser.add_argument('--mu', default = None, type = float, nargs='+', help='the mu list')
    parser.add_argument('--sigma', default = 0.5, type = float, help='the sigma list')

    args = parser.parse_args()
 
    return args


if __name__ == '__main__':
    args = get_parser()
    main(args)