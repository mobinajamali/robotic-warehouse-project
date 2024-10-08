import rware
import gymnasium as gym

env = gym.make("rware-tiny-2ag-v2")
obs, info = env.reset()
print(f'obs: {obs}')
print(f'info: {info}')
action = env.action_space.sample()
print(f'action: {action}')
print(f'n_agents: {env.unwrapped.n_agents}')