import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CriticNetwork(nn.Module):
    def __init__(self, lr_critic, n_agents, input_dims, n_actions, fc1_dim, fc2_dim, ckp_dir, name):
        super(CriticNetwork, self).__init__()
        self.ckp_file = os.path.join(ckp_dir, name)
        ###self.fc1 = nn.Linear(input_dims+n_agents*n_actions, fc1_dim)
        self.fc1 = nn.Linear(input_dims, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.q = nn.Linear(fc2_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_critic)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        #print("State shape:", state.shape) 
        #print("Action shape:", action.shape) 
        
        layer1 = F.relu(self.fc1(T.cat([state, action], dim=1)))
        ###layer1 = F.relu(self.fc1(T.cat((state, action.squeeze(1)), dim=1)))
        layer2 = F.relu(self.fc2(layer1))
        layer3 = self.q(layer2)
        return layer3


    def save_checkpoint(self):
        T.save(self.state_dict(), self.ckp_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.ckp_file))
    
class ActorNetwork(nn.Module):
    def __init__(self, lr_actor, input_dims, fc1_dim, fc2_dim, n_actions, ckp_dir, name):
        super(ActorNetwork, self).__init__()
        self.ckp_file = os.path.join(ckp_dir, name)
        self.fc1 = nn.Linear(input_dims, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.pi = nn.Linear(fc2_dim, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_actor)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        #print(f"State shape: {state}") 
        ###state = T.tensor(state, dtype=T.float, device=self.device)
        layer1  = F.relu(self.fc1(state))
        layer2 = F.relu(self.fc2(layer1))
        ###layer3 = T.tanh(self.pi(layer2))
        layer3 = F.softmax(self.pi(layer2), dim=-1)
        return layer3
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.ckp_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.ckp_file))






