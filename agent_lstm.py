import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory, ReplayMemoryLSTM
from model import DQN, DQN_LSTM
from utils import find_max_lives, check_live, get_frame, get_init_state
from config import *
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent_LSTM(Agent):
    def __init__(self, action_size):
        super().__init__(action_size)
        self.memory = ReplayMemoryLSTM()
        self.policy_net = DQN_LSTM(action_size)
        self.policy_net.to(device)
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    def get_action(self, state, hidden=None):
        state = torch.from_numpy(state).float().to(device)
        if np.random.rand() <= self.epsilon:
            action = torch.tensor(np.random.randint(self.action_size)).to(device)
            _, hidden = self.policy_net(state.unsqueeze(0), hidden, train=False)
        else:
            q_values, hidden = self.policy_net(state.unsqueeze(0), hidden, train=False)
            action = q_values.max(1)[1]
        return action, hidden

    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        
        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch, dtype=object).transpose()
        
        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :lstm_seq_length, :, :]) / 255.
        states = torch.from_numpy(states).to(device)
        actions = list(mini_batch[1])
        actions = torch.LongTensor(actions).to(device)
        rewards = list(mini_batch[2])
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        next_states = torch.from_numpy(next_states).to(device)
        dones = mini_batch[3]
        mask = torch.tensor(list(map(int, dones == False)), dtype=torch.uint8).to(device)

        self.optimizer.zero_grad()
        q_values, _ = self.policy_net(states, None)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values, _ = self.policy_net(next_states, None)
        next_state_values = next_q_values.max(1)[0]
        next_state_values = next_state_values[mask]
        expected_state_action_values = (next_state_values.detach() * self.discount_factor) + rewards[mask]
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
        
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
