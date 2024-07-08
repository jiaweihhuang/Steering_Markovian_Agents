import torch
import numpy as np

'''
player 1: red
player 2: blue

Initialization: 
two players, stag, rabbit are initialized at four corner, respectively

stag and rabbit are fixed, but players can move

In the following four cases, the environment will reinitialize:
(1) if two players reach the stag block at the same time, each of them receive reward = 0.25;  
(2) if only one player reach the stag block, it receives reward 0;
(3) if two players reach the rabbit block at the same time, each of them receive reward = 0.1;
(4) if only one player reach the rabbit block, it receives reward 0.2;
'''
class StagHuntGameGPU:
    """
    Vectorized Coin Game environment.
    Note: slightly deviates from the Gym API.
    """

    NUM_AGENTS = 2
    MOVES = torch.stack(
        [
            torch.LongTensor([0, 1]),
            torch.LongTensor([0, -1]),
            torch.LongTensor([1, 0]),
            torch.LongTensor([-1, 0]),
            torch.LongTensor([0, 0]),
        ],
        dim=0,
    )
    NUM_ACTIONS = MOVES.shape[0]

    def __init__(self, max_steps, batch_size, grid_size=3, device=None, scale=1.0):
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.batch_size = batch_size
        # The 4 channels stand for 2 players and 2 coin positions
        self.ob_space_shape = [4, grid_size, grid_size]
        self.NUM_STATES = np.prod(self.ob_space_shape)
        self.available_actions = self.MOVES.shape[0]
        self.step_count = None

        assert device is not None
        self.device = device
        self.MOVES = self.MOVES.to(self.device)

        self.state_min = torch.tensor(0, dtype=torch.int32).to(self.device)
        self.state_max = torch.tensor(grid_size - 1, dtype=torch.int32).to(self.device)

        # normalized return
        self.scale = scale
        self.rew_stag = 0.25 * self.scale
        self.rew_rabbit_together = 0.1 * self.scale
        self.rew_rabbit_alone = 0.2 * self.scale

    def get_init_position(self, num_samples):
        init = torch.tensor([0, 3, 1, 2]).expand(num_samples, -1).to(self.device)

        return init

    def reset(self):
        self.step_count = 0
        init = self.get_init_position(self.batch_size)

        self.red_pos = torch.stack((init[:,0] // 2 * (self.grid_size - 1), init[:,0] % 2 * (self.grid_size - 1)), dim=-1)
        self.blue_pos = torch.stack((init[:,1] // 2 * (self.grid_size - 1), init[:,1] % 2 * (self.grid_size - 1)), dim=-1)
        self.rabbit_pos = torch.stack((init[:,2] // 2 * (self.grid_size - 1), init[:,2] % 2 * (self.grid_size - 1)), dim=-1)
        self.stag_pos = torch.stack((init[:,3] // 2 * (self.grid_size - 1), init[:,3] % 2 * (self.grid_size - 1)), dim=-1)

        state = self._generate_state()
        return state

    def _reset_positions(self, num_reinit):
        # reset positions for those batches at least one player reach the stag or rabbit
        init = self.get_init_position(num_reinit)

        self.red_pos[self.should_reinit] = torch.stack((init[:,0] // 2 * (self.grid_size - 1), init[:,0] % 2 * (self.grid_size - 1)), dim=-1)
        self.blue_pos[self.should_reinit] = torch.stack((init[:,1] // 2 * (self.grid_size - 1), init[:,1] % 2 * (self.grid_size - 1)), dim=-1)
        self.rabbit_pos[self.should_reinit] = torch.stack((init[:,2] // 2 * (self.grid_size - 1), init[:,2] % 2 * (self.grid_size - 1)), dim=-1)
        self.stag_pos[self.should_reinit] = torch.stack((init[:,3] // 2 * (self.grid_size - 1), init[:,3] % 2 * (self.grid_size - 1)), dim=-1)


    def _same_pos(self, x, y):
        return torch.all(x == y, dim=-1)

    def _generate_state(self):
        red_pos_flat = self.red_pos[:, 0] * self.grid_size + self.red_pos[:, 1]
        blue_pos_flat = self.blue_pos[:, 0] * self.grid_size + self.blue_pos[:, 1]

        rabbit_pos_flat = self.rabbit_pos[:, 0] * self.grid_size + self.rabbit_pos[:, 1]
        stag_pos_flat = self.stag_pos[:, 0] * self.grid_size + self.stag_pos[:, 1]

        state = torch.zeros((self.batch_size, 4, self.grid_size * self.grid_size)).to(self.device)

        state[:, 0].scatter_(1, red_pos_flat[:, None], 1)
        state[:, 1].scatter_(1, blue_pos_flat[:, None], 1)
        state[:, 2].scatter_(1, rabbit_pos_flat[:, None], 1)
        state[:, 3].scatter_(1, stag_pos_flat[:, None], 1)

        return state.view(self.batch_size, 4, self.grid_size, self.grid_size)

    def step(self, actions):
        ac0, ac1 = actions

        self.step_count += 1

        self.red_pos = torch.clip(self.red_pos + self.MOVES[ac0], min=self.state_min, max=self.state_max)
        self.blue_pos = torch.clip(self.blue_pos + self.MOVES[ac1], min=self.state_min, max=self.state_max)

        # Compute rewards
        # note that this the a batch/vectorized environment
        red_reward = torch.zeros(self.batch_size).to(self.device)
        blue_reward = torch.zeros(self.batch_size).to(self.device)

        red_rabbit_matches = self._same_pos(self.red_pos, self.rabbit_pos)
        red_stag_matches = self._same_pos(self.red_pos, self.stag_pos)

        blue_rabbit_matches = self._same_pos(self.blue_pos, self.rabbit_pos)
        blue_stag_matches = self._same_pos(self.blue_pos, self.stag_pos)

        rabbit_together = torch.logical_and(red_rabbit_matches, blue_rabbit_matches)
        red_rabbit_alone = torch.logical_and(red_rabbit_matches, torch.logical_not(blue_rabbit_matches))
        blue_rabbit_alone = torch.logical_and(blue_rabbit_matches, torch.logical_not(red_rabbit_matches))

        stag_together = torch.logical_and(red_stag_matches, blue_stag_matches)
        red_stag_alone = torch.logical_and(red_stag_matches, torch.logical_not(blue_stag_matches))
        blue_stag_alone = torch.logical_and(blue_stag_matches, torch.logical_not(red_stag_matches))
        
        red_reward[stag_together] += self.rew_stag
        red_reward[red_rabbit_alone] += self.rew_rabbit_alone
        red_reward[rabbit_together] += self.rew_rabbit_together

        blue_reward[stag_together] += self.rew_stag
        blue_reward[blue_rabbit_alone] += self.rew_rabbit_alone
        blue_reward[rabbit_together] += self.rew_rabbit_together

        red_matches = torch.logical_or(red_rabbit_matches, red_stag_matches)
        blue_matches = torch.logical_or(blue_rabbit_matches, blue_stag_matches)
        self.should_reinit = torch.logical_or(red_matches, blue_matches)

        num_reinit = torch.sum(self.should_reinit)
        if num_reinit > 0:
            self._reset_positions(num_reinit)

        reward = [red_reward.float(), blue_reward.float()]
        state = self._generate_state()
        if self.step_count >= self.max_steps:
            done = torch.ones(self.batch_size).to(self.device)
        else:
            done = torch.zeros(self.batch_size).to(self.device)

        info = {
            'stag_together': stag_together,
            'red_stag_alone': red_stag_alone,
            'blue_stag_alone': blue_stag_alone,

            'rabbit_together': rabbit_together,
            'red_rabbit_alone': red_rabbit_alone,
            'blue_rabbit_alone': blue_rabbit_alone,
            'should_reinit': self.should_reinit,
        }

        return state, reward, done, info