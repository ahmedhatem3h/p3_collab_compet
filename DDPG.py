# individual network settings for each actor + critic pair

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn as nn

import random
import copy
from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OUNoise:
    ''' Ornstein-Uhlenbeck process '''
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
        
    def reset(self):
        ''' reset to internal state to initial mu '''
        self.state = copy.copy(self.mu)
        
    def sample(self):
        ''' update internal state and return it as a noise sample '''
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=512, fc2_units=256):
       
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=512, fc2_units=256, fc3_units=256):
        
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size*2 , fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size*2, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = F.relu(self.fc1(states))
        x = torch.cat((x, actions), dim=1).to(device)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return self.fc4(x)

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, batch_size, random_seed, tau, lr_a, lr_c, wd, actor_id):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            tau (float) : for soft update of target network
            lr_a (float) : learning rate for actor network
            lr_c (float) : learning rate for critic network
            wd (float) :  L2 weight decay
            actor_id (int) : agent id
        """
        self.actor_id = actor_id
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.seed = random.seed(random_seed)

        self.tau = tau
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.wd = wd

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_a)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_c, weight_decay=self.wd)

        self.soft_update(self.critic_local, self.critic_target, 1)
        self.soft_update(self.actor_local, self.actor_target, 1)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)


    def reset(self):
        self.noise.reset()

    def act(self, state, noise = 1, add_noise=True):
        """Returns actions with noise for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += (noise * self.noise.sample())
        return np.clip(action, -1, 1)

    def local_act(self, state, add_noise=False):
        """Returns actions without noise for given states as per current policy."""
        
        state = torch.FloatTensor(state).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().numpy()
        self.actor_local.train()

        return action

    def target_act(self, state, add_noise=False):
        """Returns actions without noise for given states as per current target policy."""
        
        state = torch.FloatTensor(state).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_target(state).cpu().numpy()
        self.actor_local.train()
        return action

    def learn(self, experiences, target_actions_next, local_actions_current, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        
        states, actions, rewards, next_states, dones = experiences
        states = torch.from_numpy(np.array(states)).float().to(device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(device)

        n_states = torch.reshape(states, (self.batch_size,-1))
        n_next_states = torch.reshape(next_states, (self.batch_size,-1))

      

        # ---------------------------- update critic ---------------------------- #
        
        # combine all the next target actions and next observations for input to target critic
        Q_targets_next = self.critic_target(n_next_states, target_actions_next)

        Q_targets = rewards[:,self.actor_id].unsqueeze(1).to(device) + gamma * Q_targets_next * (1 - dones[:, self.actor_id].unsqueeze(1).to(device)) 
    
        # Compute critic loss
        n_actions = torch.reshape(actions, (self.batch_size,-1)).to(device)

        Q_expected = self.critic_local(n_states, n_actions)
        
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        # combine all the actions and observations for input to critic
        actor_loss = -self.critic_local(n_states, local_actions_current).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
    
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
