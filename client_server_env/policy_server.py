import ray
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env.policy_server_input import PolicyServerInput
import gymnasium as gym
from gymnasium.spaces import Discrete


def policy_input(ioctx):
    return PolicyServerInput(ioctx, 'localhost', 9900)


if __name__ ==  '__main__':
    ray.init()
    env_size = 4
    config = DQNConfig()
    config.environment(
        env=None,
        action_space=Discrete(4),
        observation_space=Discrete(env_size * env_size)
    )
    config.debugging(log_level='INFO')
    config.rollouts(
        num_rollout_workers=0,
        enable_connectors=False)
    config.offline_data(
        input_=policy_input,
    )
    algo = config.build()
    time_steps = 0
    for _ in range(10):
        results = algo.train()
        ckpt = algo.save('./ckpt_cs')
        if time_steps >= 1000:
            break
        time_steps += results['timesteps_total']