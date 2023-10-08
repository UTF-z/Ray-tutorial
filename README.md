# 如何将自定义环境集成进RLlib

## 1. 自定义单智能体环境

### 1.1 环境定义的规范

想要将自定义单智能体环境集成到RLlib中,  在定义环境时只需要让自定义环境继承自 `gymnasium.Env` 并按照OpenAI gymnasium的Env API 定义环境即可. 具体而言, 需要让环境类对象包含两个属性和三个方法. 方法的定义规范如下所示:

```python
class SingleAgentEnv(gymnasium.Env):

	def __init__(self, env_config):
		super().__init__()
		self.observation_space = <Space>
		self.action_space = <Space>
		...
	def reset(self, *, seed=None, options=None):
		super().reset(seed=seed)
		...
		return observation, info
	def step(self, action):
		...
		return observation, reward, terminated, truncated, info
```

两个属性是指状态空间 `observation_space` 和 动作空间 `action_space` 一般来讲可以使用 `gymnasium.spaces` 的 `Discrete` 或者 `Box`.

三个方法分别是构造函数, reset方法和step方法.

**构造函数**的参数是一个名为 `env_config` 的字典,其中包含了构造环境所需要的配置变量.

**reset方法**的参数按照代码中所示的方式定义, 其中seed是随机数种子, options包含了自定义环境reset可能需要的其他变量.

**step方法**仅包含一个动作作为参数, 负责实现MDP的状态转移, 重点在于它的五个返回值. 其中observation是下一个状态观测值, reward是这一步的奖励信号, terminated和truncated都是布尔值, 前者表示agent是否完成任务, 比如到达终点什么的, 后者表示当前episode是否因为达到了最大步数等原因被提前截断了. 最后的info包含一些调试信息或者日志信息.

### 1.2 如何在RLlib中使用环境

使用RLlib的核心在于它的 `Algorithm`, 而一般而言每个Algorithm都是从相应的Config里面build出来的. 所以使用RLlib进行训练的流程是

- 实例化一个对应算法的Config对象
- 在Config对象中进行相应的配置.
- 调用Config对象的build方法生成一个Algorithm对象
- 使用Algorithm对象的train方法进行一轮训练.
- 再合适的时候使用ALgorithm对象的save/save_checkpoint方法保存训练结果.
- 使用Algorithm的evaluate方法或者自定义测试方法进行测试.

其中第二步设置Config对象是精髓, RLlib的Algorithm对象配置的灵活度是很高的, 算法的超参数, 训练参数, 分布式训练的设置等等都可以配置. 我们的自定义环境也是在这一步整合到RLlib平台中去的. 具体而言, 我们通过Config对象的environment方法进行设置.

```python
config = DQNConfig()
config.environment(
    env=SimpleMaze,
    env_config={'size': 4},
    observation_space=None,
    action_space=None,
    render_env=False
)
```

其中env字段直接传入自定义类(type), env_config则是之前说的环境构造函数的参数, 是一个字典. observation_space和action_space传入None, 这是因为环境里面有, RLlib可以自动推断出来. 但如果RLlib访问不到环境, 我们需要在这里设置, 这种情况一般是使用外部环境(ExternalAgentEnv)的时候发生, 后面会提及.

这一部分的例程我提供在'simple_gym_env'文件夹中, 包含一个保存的checkpoint文件夹和一个python脚本, 脚本里定义了一个简单的GridWorld, 使用RLlib的DQN Algorithm进行训练.

## 2. 自定义多智能体环境

### 2.1 环境定义的规范

自定义多智能体环境的方式和之前单智能体大同小异, 仍然需要实现"两个属性, 三个方法". 但它们都略需调整.

- 首先, 自定义多智能体环境要继承自RLlib的MultiAgentEnv类.
- **构造函数**中, 要为每个agent生成一个固定的agent_id, 可以是数字或字符串.
- 对于**observation_space**和**action_space**, 我们需要为每个agent_id定义好它自己的动作和状态空间. 所以现在它们都是字典, 不过这个字典需要用 `gym.spaces.Dict()`封装一下, 像这样:

```python
from gymnasium.spaces import Dict
...
class MAEnv(MultiAgentEnv):
	...
	self.observation_space = Dict({
		'agent_0': Box(low=-1.0, high=1.0, shape=(10,)),
		'agent_1': Box(low=-1.0, high=1.0, shape=(20,))
	})
	self.action_space = Dict({
		'agent_0': Discrete(2),
		'agent_1': Discrete(3)
	})
	...
```

当然, 如果所有agent都有相同的动作和状态空间, 那么 `observation_space`和 `action_space`直接定义成相应的space即可.

- **reset方法**返回的observations是一个字典, 包含了每一个agent_id对应的observation. 如下所示

```python
observations = {
	'agent_1': <obs1>,
	'agent_2': <obs2>,
	...,
	'agent_n': <obsn>
}
```

- **step方法**要变的地方稍微多一点.

  首先, 传入的参数actions现在也是一个字典, 包含这**这一步要采取行动的**的agents和它们对应的动作.

  所有返回值现在都是字典, 从agent_id映射到它们各自的信息.

  返回的observations和reset方法返回的observation形式一样, 但里面不一定包含全部的agents_id, 而只需要包含**下一步需要采取行动的**agents的obs.

  rewards字典里面可以包含任意agent对应的reward.

  terminateds和truncateds包含的agents_id和actions中要一致.同时都还要多一条'__all__'字段, 表示所有agents(整个episode)是否终止或截断.

  infos中包含的agent_id需要是返回的observations包含的agent_id的子集.

### 2.2 如何在RLlib中使用环境

在config里面配置一个多智能体环境除了像单智能体环境那样通过 `config.environment`配置相关字段之外, 还要通过 `config.multi_agent`配置多智能体相关的字段, 包括

- policies: 传入一个字典, 将每个policy的名称映射到对应的PolicySpec上去, 在PolicySpec可以覆写observation_space, action_space和algorithm的一些超参.
- policy_mapping_fn: 传一个函数, 参数接口是(agent_id, episode, worker, **kwargs), 返回一个policies中policy的名称. 这个函数负责把agent_id映射到它专属的policy上面去.

下面是一个例子:

```python
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
```

完整的例程在multi_agent_env文件夹中, 里面定义了一个多智能体的GridWorld. 可以注意一下里面是如何写一个简单的测试函数来测试多智能体策略.

## 3. 外部环境

如果RLlib不能访问到环境, 比如环境在另一台机器上, 上面无法进行RLlib的训练. 那么我们可以RLlib提供的Client-Server训练模式.

在CS训练模式下, Server端部署RLlib训练模块, Client端部署环境. Client每一步向Server发送observation, 获取actions, 然后自己step自己的环境并向Server报告这一步的reward, done-dict, trunc-dict和info.

### 3.1. 如何写一个client.

- client端要实例化一个PolicyClient(ray.rllib.env.policy_client.PolicyClient), 构造函数提供server端的套接字和inference_mode. inference_mode可以设置为remote或者loca. 后者会从server端拉一份policy下来自己跑, 会比remote快一些. client端就利用这个PolicyClient对象(下称clt)和server通信.
- 在每个episode开始或重置的时候, 调用 `clt.start_episode(training_enabled=True)`拿到一个当前episode的id.
- 每次需要获取action的时候, 使用 `clt.get_action(episode_id, observation)`请求server给一个action.
- step之后,  通过 `clt.log_returns(episode, reward, info=info)`来向server汇报reward情况.
- 每个episode结束之后, 通过 `clt.end_episode(episode_id, s)`来向server报告episode已经结束.

下面是一个例子:

```python
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
```

### 3.2. 如何写一个server.

server端通过PolicyServerInput (ray.rllib.env.policy_server_input)来接收client传进来的信息. 我们先上例子:

```python
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
```

可以看到, 首先要用 `ray.init()` 来初始化server. Config配置environment的时候, env字段设为None, 因为server上面没有环境. 但接下来action_space和observation_space就要设置成和环境一致的了, 因为Algorithm无法从client的环境中推断(是否有更好的方法?).

在server上, 要额外通过rollouts方法设置一下关于rollout worker的内容. num_rollout_workers如果大于1, 那么不同的worker要监听不同的端口. 注意enable_connectors要设置为False, 否则会报错(大概是目前connector和ExternalEnv不适配).

接下来就是最关键的一步, 通过offline_data方法设置input_字段, 传入一个函数, 这个函数的定义如下

```python
def policy_input(ioctx):
    return PolicyServerInput(ioctx, <SERVER_IP_ADDRESS>, <PORT>)
```

设置好input_字段之后, 照常build出一个algorithm进行训练就好了. server会阻塞到train上面, 等待client接入.

例程放在了client_server_env里面.

> 参考资料:
>
> [4. Reinforcement Learning with Ray RLlib - Learning Ray [Book] (oreilly.com)](https://www.oreilly.com/library/view/learning-ray/9781098117214/ch04.html#idm45752868452800)
>
> [Environments — Ray 2.4.0](https://docs.ray.io/en/latest/rllib/rllib-env.html#)
>
> [Gymnasium Documentation (farama.org)](https://gymnasium.farama.org/)
