import numpy as np
import gymnasium as gym
import rware
from multiagent import MultiAgent
from replay_buffer import Replay
from torch.utils.tensorboard import SummaryWriter
import datetime


def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

def main():
    env = gym.make("rware-tiny-2ag-v2", render_mode='human')

    summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_' 
    writer = SummaryWriter(summary_writer_name)

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
 
    ###critic_dims = sum(actor_dims)
    critic_dims = sum(actor_dims) + sum(n_actions)
    agents = MultiAgent(actor_dims=actor_dims, critic_dims=critic_dims, n_agents=n_agents, n_actions=n_actions, env=env, ckp_dir='tmp/', gamma=0.95, lr_actor=1e-4, lr_critic=1e-3)
    critic_dims = sum(actor_dims)
    memory = Replay(mem_size=1_000_000, critic_dims=critic_dims, actor_dims=actor_dims, n_actions=n_actions, n_agents=n_agents, batch_size=1024)

    MAX_STEPS = 1_000
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
        #print(f"Starting episode {i}")
        obs, _ = env.reset()
        done, trunc = False, False
        episode_step = 0
        score = 0
        while not (done or trunc):
            if evaluate:
                env.render()
            obs = {i: obs[i] for i in range(len(obs))}
            #print(obs)
            actions = agents.choose_action(obs)
            actions = list(actions.values())
            obs_, reward, done, trunc, info = env.step(actions)
            
            obs = list(obs.values())
            obs_ = list(obs_)
            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)
            #print(f'obs: {obs}')
            #print(f'obs_: {obs_}')
            #print(f'state: {state}')
            #print(f'state_: {state_}')

            #terminal = [d or t for d, t in zip(done, trunc)]
            if episode_step >= MAX_STEPS:
                print(f"Reached MAX_STEPS: {episode_step}")
                done = True

            memory.store_transition(obs, state, actions, reward,
                                    obs_, state_, done)

            if total_steps % 100 == 0 and not evaluate:
                #print("Learning...")
                agents.learn(memory)
            
            obs = obs_
            total_steps += 1
            episode_step += 1
            score += sum(reward) 
            agents.decrement_eps()
        
        writer.add_scalar('reward/train', score, episode)
        score_history.append(score)
        #print(f'score history {score_history}')
        avg_score = np.mean(score_history[-100:])
        #print(f'avg_score {avg_score}')
        print(f"Episode: {i}, total numsteps: {total_steps}, episode steps: {episode_step}, reward: {score}")
        
        if not evaluate:
            #if avg_score > best_score:
            if episode % 10 == 0:
                agents.save_checkpoint()
                #best_score = avg_score
        #if i % PRINT_INTERVAL == 0 and i > 0:
        #    print(f'Episode {i}, average score {avg_score:.1f}')
        episode += 1

if __name__ == '__main__':
    main()

    