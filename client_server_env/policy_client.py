import gymnasium as gym
from ray.rllib.env.policy_client import PolicyClient
from gymnasium.spaces import Discrete

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

    def render(self, *args, **kwargs):
        grid = [['| ' for _ in range(self.size)] + ["|\n"] for _ in range(self.size)]
        grid[self.goal[0]][self.goal[1]] = '|G'
        grid[self.agent[0]][self.agent[1]] = '|1'
        print(''.join([''.join(grid_row) for grid_row in grid]))

if __name__ == '__main__':
    env_config = {'size': 4}
    env = SimpleMaze(env_config)
    client = PolicyClient('http://localhost:9900', inference_mode='remote')
    s, _ = env.reset()
    episode_id = client.start_episode(training_enabled=True)
    rewards = 0.0
    while True:
        action = client.get_action(episode_id, s)
        ns, r, done, trunc, info = env.step(action)
        rewards += r
        client.log_returns(episode_id, r, info=info)
        s = ns
        if done or trunc:
            print('Total reward', rewards)
            rewards = 0.0
            client.end_episode(episode_id, s)
            s, _ = env.reset()
            episode_id = client.start_episode(training_enabled=True)