import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory
from model import DQN
from utils import find_max_lives, check_live, get_frame, get_init_state
from config import *
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, action_size):
        self.action_size = action_size
        
        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 500000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 1000

        # Generate the memory
        self.memory = ReplayMemory()

        # Create the policy net and the target net
        self.policy_net = DQN(action_size)
        self.policy_net.to(device)
        
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

        # Initialize a target network and initialize the target network to the policy net
        self.target_net = DQN(action_size).to(device)
        self.target_net.eval()

    def load_policy_net(self, path):
        self.policy_net = torch.load(path)           

    # after some time interval update the target net to be same with policy net
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = torch.tensor(random.randrange(self.action_size)).to(device)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            action = self.policy_net(state).max(1)[1].unsqueeze(0)
        return action


    # pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch, dtype=object).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :4, :, :]) / 255.
        states = torch.from_numpy(states).cuda()
        actions = list(mini_batch[1])
        actions = torch.LongTensor(actions).cuda()
        rewards = list(mini_batch[2])
        rewards = torch.FloatTensor(rewards).cuda()
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        next_states = torch.from_numpy(next_states).to(device)
        dones = mini_batch[3] # checks if the game is over
        mask = torch.tensor(list(map(int, dones==False)),dtype=torch.uint8).cuda()
        
        # Your agent.py code here with double DQN modifications
        ### CODE ###
        curr_state_actions = self.policy_net(states)
        curr_state_values = curr_state_actions.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_curr_actions = self.policy_net(next_states)
        next_curr_actions = next_curr_actions.max(1)[1]
        # print("hi")
        # next_curr_actions = next_curr_actions.detach()
        # Compute Q function of next state using the same policy network
        next_state_actions = self.target_net(next_states).detach()
        next_state_values = next_state_actions.gather(1, next_curr_actions.unsqueeze(1)).squeeze(1)
        next_state_values = next_state_values * mask.float()

        # Compute the expected Q values using the Bellman equation
        expected_state_values = rewards + self.discount_factor * next_state_values

        # Compute the Huber Loss between current and expected Q values
        criterion = nn.SmoothL1Loss()
        loss = criterion(curr_state_values, expected_state_values)

        # Optimize the model: perform backpropagation and update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()  
        