#from agent import Agent
from agent2 import Agent
import torch as T
import numpy as np

class MultiAgent:
    def __init__(self, actor_dims, critic_dims, n_actions, env, n_agents,
                 lr_actor=1e-4, lr_critic=1e-3, fc1_dim=64, fc2_dim=64, gamma=0.95, tau=0.01,
                 ckp_dir='tmp/', scenario='rware-tiny-2ag-v2'):
        self.agents = []
        ckp_dir += scenario
        for i in range(n_agents):
            ###action = list(env.action_space.keys())[i]
            self.agents.append(Agent(actor_dim=actor_dims[i], critic_dim=critic_dims, n_actions=n_actions[i], n_agents=n_agents, agent_id=i, epsilon=1.0,  eps_min=0.05, eps_dec=5e-4,
                                     lr_actor=lr_actor, lr_critic=lr_critic, tau=tau, fc1_dim=fc1_dim, fc2_dim=fc2_dim, ckp_dir=ckp_dir, gamma=gamma))

    def choose_action(self, obs):
        ###actions = []
        actions = {}
        ### for agent_id, agent in enumerate(self.agents):
        for agent_id, agent in zip(obs, self.agents):
            action = agent.choose_action(obs[agent_id])
            actions[agent_id] = action
            #print(f"Chosen action for agent {agent_id}: {action}")
        return actions

    def decrement_eps(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def learn(self, memory):
        for agent in self.agents:
            agent.learn(memory, self.agents)

    def save_checkpoint(self):
        print("Saving models...")
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print("Loading models...")
        for agent in self.agents:
            agent.load_models()
        print("Successfully loaded models")

