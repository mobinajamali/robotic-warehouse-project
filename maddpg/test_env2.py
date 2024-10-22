import gymnasium as gym
import rware

env = gym.make("rware-tiny-2ag-v2")
_, _ = env.reset()
for agent in range(env.unwrapped.n_agents):
    print(f'agent observation space {env.observation_space[agent]}')
obs, info = env.reset()
print(f'initial observation: {obs}, debug info: {info}')
done = False
while not done:
    actions = {}
    for agent in range(env.unwrapped.n_agents):
        actions[agent] = env.action_space[agent].sample()
    obs_, reward, done, trunc, info = env.step(actions)
    print(f'actions taken{actions}')
obs = list(obs)
print(f'obs as a list {obs}')