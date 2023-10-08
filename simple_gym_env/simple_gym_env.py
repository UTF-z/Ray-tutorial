from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gymnasium as gym
from gymnasium.spaces import Discrete
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms import Algorithm
from ray.rllib.utils import check_env
import os

class SimpleMaze(gym.Env):

    def __init__(self, env_config):
        super().__init__()
        size = env_config['size']
        self.size = size
        self.action_space = Discrete(4)
        self.observation_space = Discrete(size*size)
        self.agent = (0, 0)
        self.goal = (size-1, size-1)
        self.info = {'obs': self.agent}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent = (0, 0)

        return self.get_observation(), self.info

    def get_observation(self):
        seeker = self.agent
        return self.size * seeker[0] + seeker[1]

    def get_reward(self):
        return 1 if self.agent == self.goal else 0

    def is_done(self):
        return self.agent == self.goal

    def step(self, action):

        seeker = self.agent
        if action == 0:  # move down
            seeker = (min(seeker[0] + 1, self.size-1), seeker[1])
        elif action == 1:  # move left
            seeker = (seeker[0], max(seeker[1] - 1, 0))
        elif action == 2:  # move up
            seeker = (max(seeker[0] - 1, 0), seeker[1])
        elif action == 3:  # move right
            seeker = (seeker[0], min(seeker[1] + 1, self.size-1))
        else:
            raise ValueError("Invalid action")
        self.agent = seeker

        observations = self.get_observation()
        rewards = self.get_reward()
        done = self.is_done()

        return observations, rewards, done, False, self.info

    # def render(self, *args, **kwargs):
    #     grid = [['| ' for _ in range(self.size)] + ["|\n"] for _ in range(self.size)]
    #     grid[self.goal[0]][self.goal[1]] = '|G'
    #     grid[self.agent[0]][self.agent[1]] = '|1'
    #     print(''.join([''.join(grid_row) for grid_row in grid]))


def train_simple_maze():
    config = DQNConfig()
    config.environment(
        env=SimpleMaze,
        env_config={'size': 4},
        observation_space=None,
        action_space=None,
        render_env=False
    )
    config.rollouts( 
        num_rollout_workers=2, 
        create_env_on_local_worker=True
    )
    config.resources(
        
    )
    pretty_print(config.to_dict())
    algo = config.build()
    for i in range(50):
        result = algo.train()
        print(pretty_print(result))
    evaluation = algo.evaluate()
    print(pretty_print(evaluation))
    checkpoint = algo.save_checkpoint('./ckpt_simple_env_4')
    print(checkpoint)

def evaluate_simple_maze():
    algo = Algorithm.from_checkpoint('./ckpt_simple_env_4')
    env_config = {
        'size': 4
    }
    env = SimpleMaze(env_config)
    s, _ = env.reset()
    env.render()
    done = False
    total_reward = 0
    while not done:
        a = algo.compute_single_action(s)
        ns, r, done, trunc, info = env.step(a)
        total_reward += r
        env.render()
        s = ns
        done = done or trunc
    print(total_reward)

if __name__ == '__main__':
    train_simple_maze()
    