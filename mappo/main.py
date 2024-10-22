import numpy as np
import gymnasium as gym
import rware
from algo import MAPPO, PPOMemory


def obs_list_to_state_vector(obseravtion):
    '''
    flattening all the obs for all the agents
    '''
    state = []
    for row in obseravtion:
        obs = np.array([])
        for o in row:
            obs = np.concatenate([obs, o])
        state.append(obs)
    return np.array(state)

def run():
    env = gym.make("rware-tiny-2ag-v2", render_mode='human')
    N = 2048  # can try 200for first debugging tries
    batch_size = 64
    n_epochs = 10
    alpha = 3e-4
    scenario = 'rware-tiny-2ag-v2'
    n_agents = 2

    actor_dims = []
    n_actions = []

    for agent in range(n_agents):
        obs_space = env.observation_space[agent]
        action_space = env.action_space[agent]

        actor_dims.append(obs_space.shape[0])
        n_actions.append(action_space.n)
    critic_dims = sum(actor_dims)

    mappo_agents = MAPPO(actor_dims=actor_dims, critic_dims=critic_dims,
                         n_agents=n_agents, n_actions=n_actions,
                         env=env, gamma=0.95, alpha=alpha,
                         scenario=scenario)

    memory = PPOMemory(batch_size, N, n_agents, env.unwrapped.agents,
                       critic_dims, actor_dims, n_actions)

    MAX_STEPS = 1_000_000
    total_steps = 0
    evaluate = False
    episode = 1
    traj_length = 0
    score_history, steps_history = [], []

    if evaluate:
        mappo_agents.load_checkpoint()

    while total_steps < MAX_STEPS:
        observation, _ = env.reset()
        terminal = False
        score = 0

        while not terminal:
            if evaluate:
                env.render()
            
            obs = {i: observation[i] for i in range(len(observation))}
            # action-probs
            a_p = list(mappo_agents.choose_action(obs))
            #print(f'ap: {a_p}')
            action = a_p[0]
            prob = a_p[1]
            #print(f'actions {action}')
            observation_, reward, done, trunc, info = env.step(action)
            obs_ = {i: observation_[i] for i in range(len(observation_))}
            #print(f'action {action}')
            #print(f'probs {prob}')
            #print(f'observation_ {obs_}')
            #print(f'reward {reward}')
            #print(f'done {done}')
            #print(f'trunc {trunc}')

            total_steps += 1
            traj_length += 1

            obs_arr = [list(obs.values())]
            new_obs_arr = [list(observation_)]

            state = obs_list_to_state_vector(obs_arr)
            state_ = obs_list_to_state_vector(new_obs_arr)

            score += sum(reward)
            #print(f'score: {score}')

            terminal = done or trunc
            #mask = [0.0 if t else 1.0 for t in terminal]  # list of floating points instead of boolean
            memory.store_memory(obs_arr, state, action,
                                prob, reward,
                                new_obs_arr, state_, terminal)
            #print('-------------------')
            if traj_length % N == 0:
                mappo_agents.learn(memory)
                traj_length = 0
                memory.clear_memory()
            obs = observation_

        score_history.append(score)
        steps_history.append(total_steps)
        avg_score = np.mean(score_history[-100:])
        print(f'Episode {episode} total steps {total_steps}'
              f' avg score {avg_score :.1f}')

        episode += 1

    np.save('data/mappo_scores.npy', np.array(score_history))
    np.save('data/mappo_steps.npy', np.array(steps_history))
    env.close()


if __name__ == '__main__':
    run()