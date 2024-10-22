import rware
import gymnasium as gym

env = gym.make("rware-tiny-2ag-v2")
obs, info = env.reset()
print(f"Observation shape: {obs[0].shape}")

# Reset the environment and get the initial observations
obs, _ = env.reset()

# Loop through the agents and print their observation and action spaces along with actual observations
for idx, agent in enumerate(env.unwrapped.agents):
    # Use the index to access the observation and action spaces
    obs_space = env.observation_space[idx]
    obs_space_shape = env.observation_space[idx].shape[0]
    action_space = env.action_space[idx]
    action_space_shape = env.action_space[idx].n
    print(f'idx: {idx}')
    print(f'agent: {agent}')
    
    # Get the observation for the current agent
    agent_obs = obs[idx]  # Access the observation for the current agent
    
    # Print the values for each agent
    print(f'agent {agent}:')
    print(f'  obs_space: {obs_space}')  # Print the observation space (shape and bounds)
    print(f'  obs_space_shape: {obs_space_shape}')
    print(f'  action_space: {action_space}')  # Print the action space (type and number of actions)
    print(f'  action_space_shape: {action_space_shape}') 
    print(f'  initial observation: {agent_obs}')  # Print the actual initial observation for this agent
    print('------------------------')
# Print the full initial observations for reference
print(f'Initial observations (all agents): {obs}')


#print("Agents:", env.unwrapped.agents)
#print("Number of Agents:", env.unwrapped.n_agents)
#print(f'obs: {obs}')
#print(f'info: {info}')
#action = env.action_space.sample()
#print(f'action: {action}')
#print(f'n_agents: {env.unwrapped.n_agents}')
#done = False
#while not done:
#    action = env.action_space.sample()
#    obs_, reward, done, trunc, info = env.step(action)
#    print('************************')
#    print(f'reward: {reward}')
#    print(f'obs_: {obs_}')
#    print(f'done: {done}')
#    print(f'trunc: {trunc}')
#    print(f'action: {action}')