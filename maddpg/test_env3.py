import gymnasium as gym
import rware

env = gym.make("rware-tiny-2ag-v2")
_, _ = env.reset()
for agent in range(env.unwrapped.n_agents):
    obs_space = env.observation_space[agent]  # 71
    action_space = env.action_space[agent]  # 5
    print(obs_space.shape[0])
    print(action_space.n)
    #print(action_space.shape[0])  