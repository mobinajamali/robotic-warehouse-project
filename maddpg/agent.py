import numpy as np
import datetime
import torch as T
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from rware.warehouse import Action
from networks import CriticNetwork, ActorNetwork
from replay_buffer import Replay

class Agent:
    def __init__(self, n_actions, actor_dim, critic_dim, agent_id, n_agents=2, ckp_dir='tmp/', 
                 lr_critic=1e-3, lr_actor=1e-4, fc1_dim=64, fc2_dim=64, gamma=0.95, tau=0.01):
        super(Agent, self).__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        agent_name = 'agent_%s' % agent_id
        self.agent_id = agent_id
        self.gamma = gamma
        self.tau = tau

        ###self.critic = CriticNetwork(lr_critic, critic_dim, fc1_dim, fc2_dim, ckp_dir, name=agent_name+'_critic_')
        self.critic = CriticNetwork(lr_critic, n_agents, critic_dim,  n_actions, fc1_dim, fc2_dim, ckp_dir, name=agent_name+'_critic_')
        self.target_critic = CriticNetwork(lr_critic, n_agents, critic_dim, n_actions, fc1_dim, fc2_dim, ckp_dir, name=agent_name+'_target_critic_')
        self.actor = ActorNetwork(lr_actor, actor_dim, fc1_dim, fc2_dim, n_actions, ckp_dir, name=agent_name+'_actor_')
        self.target_actor = ActorNetwork(lr_actor, actor_dim, fc1_dim, fc2_dim, n_actions, ckp_dir, name=agent_name+'_target_actor_')

        # initial hard copy (tau=1)
        self.network_update(self.actor, self.target_actor, tau=1)
        self.network_update(self.critic, self.target_critic, tau=1)


    def network_update(self, src, dest, tau=None):
        tau = tau or self.tau
        for param, target in zip(src.parameters(), dest.parameters()):
            target.data.copy_(tau * param.data + (1 - tau) * target.data)

    #def choose_action(self, obs):
    #    #print(f"Input obs_list: {obs}")
    #    state = T.tensor(obs[np.newaxis, :], dtype=T.float, device=self.actor.device)
    #    ###state = T.FloatTensor(obs).unsqueeze(0).to(self.actor.device)
    #    #print(f"Final converted state vector: {state}")
    #    
    #    actions = self.actor.forward(state)
    #    action = T.argmax(actions).item() % 5  
    #    return action  
    #    #return action.data.cpu().numpy()[0]

    def choose_action(self, state):
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action_probs = self.actor.forward(state)  # Get action probabilities
        action_distribution = T.distributions.Categorical(action_probs)  # Create a categorical distribution
        action = action_distribution.sample()  # Sample an action
        return action.item()  # Return the action index
            
    def learn(self, memory: Replay, agent_list):
        if not memory.ready():
            return 
        
        obs, state, action, reward, obs_, state_, done= memory.sample_buffer()
        states = T.tensor(state, dtype=T.float, device=self.actor.device)
        states_ = T.tensor(state_, dtype=T.float, device=self.actor.device)
        rewards = T.tensor(np.array(reward), dtype=T.float, device=self.actor.device)
        dones = T.tensor(np.array(done), device=self.actor.device)

        obss = [T.tensor(obs[idx], device=self.actor.device, dtype=T.float) for idx in range(len(agent_list))]
        obss_ = [T.tensor(obs_[idx], device=self.actor.device, dtype=T.float) for idx in range(len(agent_list))]
        actions = [T.tensor(action[idx], device=self.actor.device, dtype=T.float) for idx in range(len(agent_list))]

        with T.no_grad():
            # calculate target critic value using target networks with no gradients
            new_actions = T.cat([agent.target_actor(obss_[i])
                                 for i, agent in enumerate(agent_list)], dim=1)
            critic_value_ = self.target_critic.forward(
                                states_, new_actions).squeeze() 
            ###critic_value_[dones[:, self.agent_id]] = 0.0
            critic_value_[dones] = 0.0
            target = rewards[:, self.agent_id] + self.gamma * critic_value_

        old_actions = T.cat([actions[i] for i in range(len(agent_list))], dim=1)
        critic_value = self.critic.forward(states, old_actions).squeeze()
        critic_loss = F.mse_loss(target, critic_value) # update

        # update critic network
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0) # gradient clip to prevent explod
        self.critic.optimizer.step()

        # calculate actor loss and update actor network
        actions[self.agent_id] = self.actor.forward(obss[self.agent_id])
        actions = T.cat([actions[i] for i in range(len(agent_list))], dim=1)
        actor_loss = -self.critic.forward(states, actions).mean() 
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor.optimizer.step()

        # soft update target networks
        self.network_update(self.actor, self.target_actor, tau=self.tau)
        self.network_update(self.critic, self.target_critic, tau=self.tau)
    

    def save_models(self):
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()

    def load_models(self):
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()




