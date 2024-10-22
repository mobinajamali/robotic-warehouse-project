import os
import gymnasium as gym
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class PPOMemory:
    def __init__(self, batch_size, mem_size, n_agents, agents, 
                 critic_dims, actor_dims, n_actions):
        # use combination of numpy arrays and dict
        self.mem_size = mem_size
        self.mem_ctr = 0
        self.batch_size = batch_size
        self.critic_dims = critic_dims
        self.actor_dims = actor_dims
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.agents = agents

        self.states = np.zeros((self.mem_size, critic_dims), dtype=np.float32)
        self.rewards = np.zeros((self.mem_size, n_agents), dtype=np.float32)
        self.dones = np.zeros(self.mem_size, dtype=np.float32)  # one done for all of the agents
        self.new_states = np.zeros((self.mem_size, critic_dims), dtype=np.float32)

        self.actor_states = {a: np.zeros((self.mem_size, actor_dims[a])) for a in range(self.n_agents)}
        self.actor_new_states = {a: np.zeros((self.mem_size, actor_dims[a])) for a in range(self.n_agents)}
        self.actions = {a: np.zeros((self.mem_size, n_actions[a])) for a in range(self.n_agents)}
        self.probs = {a: np.zeros((self.mem_size, n_actions[a])) for a in range(self.n_agents)}

    def recall(self):
        return self.actor_states, \
        self.states, \
        self.actions, \
        self.probs, \
        self.rewards, \
        self.actor_new_states, \
        self.new_states, \
        self.dones

    def generate_batches(self):
        n_batches = int(self.mem_size // self.batch_size)
        indices = np.arange(self.mem_size, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i*self.batch_size:(i+1)*self.batch_size]
                   for i in range(n_batches)]
        return batches

    def store_memory(self, raw_obs, state, action, probs, reward, 
                     raw_obs_, state_, done):
        '''
        raw_obs: obs of each individual agent
        state: concate obs of all the agents
        '''
        index = self.mem_ctr % self.mem_size
        self.states[index] = state
        self.new_states[index] = state_
        self.dones[index] = done
        self.rewards[index] = reward

        for agent in self.agents:
            self.actions[agent][index] = action[agent]
            self.actor_states[agent][index] = raw_obs[agent]
            self.actor_new_states[agent][index] = raw_obs_[agent]
            self.probs[agent][index] = probs[agent]
        self.mem_ctr += 1

    def clear_memory(self):
        '''
        clear up memory at the end of each trajectory
        '''
        self.states = np.zeros((self.mem_size, self.critic_dims), dtype=np.float32)
        self.rewards = np.zeros((self.mem_size, self.n_agents), dtype=np.float32)
        self.dones = np.zeros(self.mem_size, dtype=np.float32) 
        self.new_states = np.zeros((self.mem_size, self.critic_dims), dtype=np.float32)
        self.actor_states = {a: np.zeros((self.mem_size, self.actor_dims[a])) for a in range(self.n_agents)}
        self.actor_new_states = {a: np.zeros((self.mem_size, self.actor_dims[a])) for a in range(self.n_agents)}
        self.actions = {a: np.zeros((self.mem_size, self.n_actions[a])) for a in range(self.n_agents)}
        self.probs = {a: np.zeros((self.mem_size, self.n_actions[a])) for a in range(self.n_agents)}


class MAPPO:
    def __init__(self, actor_dims, critic_dims, n_actions, env, n_agents=2, 
                 alpha=0.0005, gamma=0.95, ckp_dir='tmp/mappo/', 
                 scenario='rware-tiny-2ag-v2'):
        self.n_agents = n_agents
        self.agents = []
        ckp_dir += scenario
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims=actor_dims[agent_idx], critic_dims=critic_dims, n_actions=n_actions[agent_idx], agent_idx=agent_idx,
                                     alpha=alpha, ckp_dir=ckp_dir, gamma=gamma, scenario=scenario))


    def choose_action(self, raw_obs):
        #print(f"Raw observations: {raw_obs}")
        actions = {}
        probs = {}
        
        for idx, agent in zip(raw_obs, self.agents):
            #print(f'Agent {idx} received observation: {raw_obs[idx]}') 
            action, prob = agent.choose_action(raw_obs[idx])
            actions[idx] = action
            probs[idx] = prob
    
        return actions, probs
    

    def learn(self, memory):
        for agent in self.agents:
            agent.learn(memory)

    def save_checkpoint(self):
        print("Saving models...")
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print("Loading models...")
        for agent in self.agents:
            agent.load_models()
        print("Successfully loaded models")


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=128,
                 fc2_dims=128, ckp_dir='models/', scenario=None):
        super(ActorNetwork, self).__init__()

        ckp_dir += scenario
        if not os.path.exists(ckp_dir):
            os.makedirs(ckp_dir)
        self.ckp_file = os.path.join(ckp_dir, 'actor_torch_ppo')
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        dist = Categorical(action_probs)
        return dist
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.ckp_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.ckp_file))
    
class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=128,
                 fc2_dims=128, ckp_dir='models/', scenario=None):
        super(CriticNetwork, self).__init__()

        ckp_dir += scenario
        if not os.path.exists(ckp_dir):
            os.makedirs(ckp_dir)
        self.ckp_file = os.path.join(ckp_dir, 'critic_torch_ppo')
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.v = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.v(x)
        return v
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.ckp_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.ckp_file))
    

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, agent_idx, 
                 gamma=0.99, alpha=0.0005, mem_size=2048, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10,
                 ckp_dir = None, scenario = None):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = 0.001
        self.agent_idx = agent_idx
        self.n_actions = n_actions
        self.mem_size = mem_size

        self.actor = ActorNetwork(n_actions, actor_dims, alpha, ckp_dir=ckp_dir, scenario=scenario)
        self.critic = CriticNetwork(critic_dims, alpha, ckp_dir=ckp_dir, scenario=scenario)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        #print(f"Agent {self} received observation: {observation}")
        with T.no_grad():
            #state = T.tensor([observation], dtype=T.float, device=self.actor.device)
            state = T.tensor(observation, dtype=T.float, device=self.actor.device)
            dist = self.actor(state)
            action = dist.sample()
            prob = dist.log_prob(action)
            actions = T.squeeze(action).item()
            probs = T.squeeze(prob).item()
            #print(f"Chosen action: {actions}, with probability: {probs}")
            #print(f'action: {action}')
            #print(f'probs: {probs}')

        #return actions.cpu().numpy(), probs.cpu().numpy()
        return actions, probs
    

    def calc_adv_and_returns(self, memories: PPOMemory):
        states, new_states, r, dones = memories
        with T.no_grad():
            values = self.critic(states).squeeze()
            values_ = self.critic(new_states).squeeze()
            deltas = r[:, self.agent_idx] + self.gamma * values_ - values
            deltas = deltas.cpu().numpy()
            adv = [0]
            for step in reversed(range(deltas.shape[0])):
                advantage = deltas[step] +\
                    self.gamma*self.gae_lambda*adv[-1]*np.array(dones[step])
                adv.append(advantage)
            adv.reverse()
            adv = np.array(adv[:-1])
            #print(f"adv shape before unsqueeze: {adv.shape}")
            adv = T.tensor(adv, device=self.critic.device).unsqueeze(1)
            returns = adv + values.unsqueeze(1)
            adv = (adv - adv.mean()) / (adv.std()+1e-4) # normalization
        return adv, returns
    

    def learn(self, memory: PPOMemory):
        actor_states, states, actions, old_probs, rewards, actor_new_states, \
        states_, dones = memory.recall()

        state_arr = T.tensor(states, dtype=T.float, device=self.critic.device)
        states__arr = T.tensor(states_, dtype=T.float, device=self.critic.device)
        r = T.tensor(rewards, dtype=T.float, device=self.critic.device)
        action_arr = T.tensor(actions[self.agent_idx], dtype=T.float, device=self.critic.device)
        old_probs_arr = T.tensor(old_probs[self.agent_idx], dtype=T.float, device=self.critic.device)
        actor_states_arr = T.tensor(actor_states[self.agent_idx], dtype=T.float, device=self.critic.device)
        adv, returns = self.calc_adv_and_returns((state_arr, states__arr, r, dones))
        
        for epoch in range(self.n_epochs):
            batches = memory.generate_batches()
            for batch in batches:
                old_probs = old_probs_arr[batch]
                actions = action_arr[batch]
                actor_states = actor_states_arr[batch]
                dist = self.actor(actor_states)
                #new_probs = dist.log_prob(actions)
                new_probs = dist.log_prob(actions.unsqueeze(-1))
                prob_ratio = T.exp(new_probs.sum(1, keepdims=True) - old_probs.sum(1, keepdims=True))

                #prob_ratio = T.exp(new_probs.sum(2, keepdims=True) - old_probs.sum(2, keepdims=True))
                weighted_probs = adv[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(
                    prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * adv[batch]
                #print(f'dist.entropy().shape: {dist.entropy().shape}')
                entropy = dist.entropy().unsqueeze(1)
                #entropy = dist.entropy().sum(1, keepdims=True)
                #entropy = dist.entropy().sum(2, keepdims=True)
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs)
                actor_loss -= self.entropy_coefficient * entropy
                self.actor.optimizer.zero_grad()
                actor_loss.mean().backward()
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor.optimizer.step()

                states = state_arr[batch]
                critic_value = self.critic(states).squeeze()
                critic_loss = (critic_value - returns[batch].squeeze()).pow(2).mean()
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

