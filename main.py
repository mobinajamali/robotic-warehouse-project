import numpy as np
import gymnasium as gym
import rware
from multiagent import MultiAgent
from replay_buffer import Replay


def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

def main():
    env = gym.make("rware-tiny-2ag-v2", render_mode='human')
    obs, _ = env.reset()
    #print(f"State shape: {obs}")
    n_agents = env.n_agents
    actor_dims = []
    n_actions = []
    for agent in range(n_agents):
        obs_space = env.observation_space[agent]
        action_space = env.action_space[agent]

        actor_dims.append(obs_space.shape[0])
        n_actions.append(action_space.n)  

    critic_dims = sum(actor_dims)
    agents = MultiAgent(actor_dims=actor_dims, critic_dims=critic_dims, n_agents=n_agents, n_actions=n_actions, env=env, ckp_dir='tmp/', gamma=0.95, lr_actor=1e-4, lr_critic=1e-3)
    memory = Replay(mem_size=1_000_000, critic_dims=critic_dims, actor_dims=actor_dims, n_actions=n_actions, n_agents=n_agents, batch_size=1024)

    MAX_STEPS = 1_000_000
    PRINT_INTERVAL = 500
    evaluate = False
    N_GAMES = 50000
    score_history = []
    total_steps = 0
    episode = 0
    best_score = 0

    if evaluate:
        agents.load_checkpoint()

    for i in range(N_GAMES):
        obs, _ = env.reset()
        done, trunc = False, False
        episode_step = 0
        score = 0
        while not (done or trunc):
            if evaluate:
                env.render()

            actions = agents.choose_action(obs)
            obs_, reward, done, trunc, info = env.step(actions)
            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            if episode_step >= MAX_STEPS:
                done = True

            memory.store_transition(obs, state, actions, reward,
                                    obs_, state_, done)

            if total_steps % 100 == 0 and not evaluate:
                agents.learn(memory)

            obs = obs_
            total_steps += 1
            score += sum(reward) 
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not evaluate:
            if avg_score > best_score:
                agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print(f'episode {i}, average score {avg_score:.1f}')
        episode += 1

if __name__ == '__main__':
    main()

    