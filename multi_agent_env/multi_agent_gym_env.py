import os
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.utils import check_env
from ray.rllib.algorithms import Algorithm
from ray.rllib.policy.policy import PolicySpec

class MultiAgentMaze(MultiAgentEnv):

    def __init__(self, env_config):
        super().__init__()
        self.size = env_config['size']
        self.action_space = Discrete(4)
        self.observation_space = Discrete(self.size * self.size)
        self.agents = {1: (self.size - 1, 0), 2: (0, self.size - 1)}
        self.goal = (self.size - 1, self.size - 1)
        
    
    @property
    def info(self):
        return {1: {'obs': self.agents[1]}, 2: {'obs': self.agents[2]}}


    def reset(self, *, seed=None, options=None):
        self.agents = {1: (self.size - 1, 0), 2: (0, self.size - 1)}

        observations =  {1: self.get_observation(1), 2: self.get_observation(2)}
        return observations, self.info

    def get_observation(self, agent_id):
        seeker = self.agents[agent_id]
        return self.size * seeker[0] + seeker[1]

    def get_reward(self, agent_id):
        return 1 if self.agents[agent_id] == self.goal else 0

    def is_done(self, agent_id):
        return self.agents[agent_id] == self.goal

    def step(self, action):
        agent_ids = action.keys()

        for agent_id in agent_ids:
            seeker = self.agents[agent_id]
            if action[agent_id] == 0:  # move down
                seeker = (min(seeker[0] + 1, self.size - 1), seeker[1])
            elif action[agent_id] == 1:  # move left
                seeker = (seeker[0], max(seeker[1] - 1, 0))
            elif action[agent_id] == 2:  # move up
                seeker = (max(seeker[0] - 1, 0), seeker[1])
            elif action[agent_id] == 3:  # move right
                seeker = (seeker[0], min(seeker[1] + 1, self.size - 1))
            else:
                raise ValueError("Invalid action")
            self.agents[agent_id] = seeker

        observations = {i: self.get_observation(i) for i in agent_ids}
        rewards = {i: self.get_reward(i) for i in agent_ids}
        terminateds = {i: self.is_done(i) for i in agent_ids}
        truncs = {i: False for i in agent_ids}
        info = {i: {'obs': self.agents[i]} for i in agent_ids}

        terminateds["__all__"] = all(terminateds.values())
        truncs['__all__'] = all(truncs.values())

        return observations, rewards, terminateds, truncs, info

    def render(self, *args, **kwargs):
        """We override this method here so clear the output in Jupyter notebooks.
        The previous implementation works well in the terminal, but does not clear
        the screen in interactive environments.
        """
        grid = [['| ' for _ in range(self.size)] + ["|\n"] for _ in range(self.size)]
        grid[self.goal[0]][self.goal[1]] = '|G'
        grid[self.agents[1][0]][self.agents[1][1]] = '|1'
        grid[self.agents[2][0]][self.agents[2][1]] = '|2'
        grid[self.agents[2][0]][self.agents[2][1]] = '|2'
        print(''.join([''.join(grid_row) for grid_row in grid]))

def train_algo():
    config = DQNConfig()
    config.environment(
        env=MultiAgentMaze,
        env_config={'size': 4},
        observation_space=None,
        action_space=None,
        render_env=False
    )
    config.multi_agent(
        policies={
            'policy_1': PolicySpec(
                policy_class=None,
                observation_space=None,
                action_space=None,
                config={
                    'gamma': 0.85
                }
            ),
            'policy_2': PolicySpec(
                policy_class=None,
                observation_space=None,
                action_space=None,
                config={
                    'gamma': 0.99
                }
            )
        },
        policy_mapping_fn= lambda agent_id, episode, worker, **kwargs:
            f'policy_{agent_id}'
    )
    algo = config.build()
    for i in range(20):
        algo.train()
    algo.save_checkpoint('./ckpt_ma_env_4') # mkdir first
    evaluation = algo.evaluate()
    print(pretty_print(evaluation))

def test_algo():
    algo = Algorithm.from_checkpoint('./ckpt_ma_env_4')
    policy_mapping_fn = lambda agent_id: f'policy_{agent_id}'
    env_config = {'size': 4}
    env = MultiAgentMaze(env_config)
    s, _ = env.reset()
    env.render()
    done = False
    reward = {1: 0, 2: 0}
    while not done:
        actions = {i: algo.compute_single_action(s[i], policy_id=policy_mapping_fn(i)) for i in s.keys()}
        ns, r, done, trunc, _ = env.step(actions)
        done = done['__all__'] or trunc['__all__']
        for id in reward.keys():
            reward[id] += r[id]
        s = ns
        env.render()

if __name__ == '__main__':
    env_config = {'size': 4}
    train_algo()