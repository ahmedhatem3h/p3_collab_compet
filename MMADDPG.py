from DDPG import Agent

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn as nn

import random
import copy
from collections import namedtuple, deque

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
#episode_counter = 0
MIN_EPS_COUNTER = 50

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = [e.state for e in experiences if e is not None]
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = [e.next_state for e in experiences if e is not None]
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class MADDPG:
	def __init__(self, state_size, action_size, random_seed):
		super(MADDPG, self).__init__()

		self.state_size = state_size
		self.action_size = action_size
		self.random_seed = random_seed

		self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, self.random_seed)

		
		self.maddpg_agent = [Agent(self.state_size, self.action_size, BATCH_SIZE, self.random_seed, TAU, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY, 0), 
							 Agent(self.state_size, self.action_size, BATCH_SIZE, self.random_seed, TAU, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY, 1)]
		
		self.iter = 0
		self.episode_counter = 0

		self.eps = 2
		self.eps_decay = 0.9999

	
	def reset_agents(self):
		for agent in self.maddpg_agent:
			agent.reset()

	def act(self, obs_all_agents):
		actions = [agent.act(obs, noise = self.eps) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
		return actions

	def local_act(self, obs_all_agents):
		"""get actions from all agents in the MADDPG object"""

		target_actions = []
		for obs in obs_all_agents:
			target_actions.append([agent.local_act(ob) for agent, ob in zip(self.maddpg_agent, obs)])
		
		return target_actions
	def target_act(self, obs_all_agents):
		"""get target network actions from all the agents in the MADDPG object """
		
		target_actions = []
		for obs in obs_all_agents:
			target_actions.append([agent.target_act(ob) for agent, ob in zip(self.maddpg_agent, obs)])
		
		return target_actions

	def step(self, state, action, reward, next_state, done):
		"""Save experience in replay memory, and use random sample from buffer to learn."""
		# Save experience / reward
		self.memory.add(state, action, reward, next_state, done)

		self.iter +=1

		if np.any(done):                                    
			self.episode_counter +=1
			self.eps = self.eps*self.eps_decay
		# Learn, if enough samples are available in memory
		if len(self.memory) > BATCH_SIZE and (self.episode_counter > MIN_EPS_COUNTER) and (self.iter%2 == 0):
			for i in range(2):
				experiences = self.memory.sample()
				self.update(experiences,i)

	def update(self, experiences, actor_id):
		"""update the critics and actors of all the agents """


		states, actions, rewards, next_states, dones = experiences

		#get target network actions from all the agents in the MADDPG object
		target_actions_next = self.target_act(next_states)
		target_actions_next = torch.FloatTensor(target_actions_next)
		target_actions_next = torch.reshape(target_actions_next, (BATCH_SIZE,4)).to(device)

		local_actions_current = self.local_act(states)
		local_actions_current = torch.FloatTensor(local_actions_current)
		local_actions_current = torch.reshape(local_actions_current, (BATCH_SIZE,4)).to(device)

		self.maddpg_agent[actor_id].learn(experiences, target_actions_next, local_actions_current, GAMMA)

