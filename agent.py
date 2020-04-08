import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import math
import sys
import pickle

device = torch.device("cuda")


class Network(nn.Module):

    def __init__(self, n_actions):
        super(Network, self).__init__()
        self.cnn = CNN(n_actions)
        self.rnn = nn.LSTM(1447, 256, 1)
        self.out = nn.Linear(256, n_actions)
        self.to(device)

    def forward(self, state, next_piece, hidden=None):
        self.rnn.flatten_parameters()
        features = torch.cat([self.cnn(state), next_piece], dim=2)
        output, hidden = self.rnn(features, hidden if hidden else self.init_hidden())
        return self.out(output), hidden

    def init_hidden(self, sz=256):
        return [torch.zeros((1,1,sz), device=device, dtype=torch.float)]*2


class CNN(nn.Module):
    
    def __init__(self, out_size):
        super(CNN, self).__init__()
        in_channels = 8
        self.convR = nn.Conv2d(in_channels, 8, (20,1))
        self.convC = nn.Conv2d(in_channels, 8, (1,10))
        self.conv1 = nn.Conv2d(in_channels, 16, 5, padding=2)
        self.block1 = nn.Sequential(
            nn.Conv2d(16, 24, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

    def forward(self, state):
        xR = torch.relu(self.convR(state)).view(state.size(0), 1, -1)
        xC = torch.relu(self.convC(state)).view(state.size(0), 1, -1)
        x = torch.relu(self.conv1(state))
        x = self.block1(x).view(state.size(0), 1, -1)
        return torch.cat([xR, xC, x], dim=2)


class Agent():
    
    def __init__(self, num_actions, eps_start=1.0, eps_end=0, eps_decay=0.996, gamma=0.992, alpha=5e-4, tau=1e-3):
        self.local_Q = Network(num_actions)
        self.target_Q = Network(num_actions)
        self.target_Q.load_state_dict(self.local_Q.state_dict())
        self.target_Q.eval()
        self.optimizer = optim.Adam(self.local_Q.parameters(), lr=alpha)
        self.loss = nn.SmoothL1Loss()
        self.num_actions = num_actions
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.gamma = gamma
        self.scores = []
        self.episodes = []
        self.durations = []
        self.start = 1

    def init_hidden(self):
        self.hidden = self.local_Q.init_hidden()
        
    def select_action(self, state, next_piece):
        if np.random.random() > self.eps_start:
            self.local_Q.eval()
            with torch.no_grad():
                obs = torch.tensor(state, device=device, dtype=torch.float).unsqueeze(0)
                next_piece = torch.tensor(next_piece, device=device, dtype=torch.float).view(1,1,7)
                output, self.hidden = self.local_Q(obs, next_piece, self.hidden)
                action = torch.argmax(output).item()
            self.local_Q.train()
        else:
            action = np.random.randint(self.num_actions)
        return action

    def learn(self, trajectory):
        batch = list(zip(*trajectory))

        state_batch = torch.tensor(batch[0], device=device, dtype=torch.float)
        next_piece_batch = torch.tensor(batch[1], device=device, dtype=torch.float).unsqueeze(1)
        action_batch = torch.tensor(batch[2], device=device)
        reward_batch = torch.tensor(batch[3], device=device)

        indexes = np.arange(state_batch.size(0)-1)
        prediction = self.local_Q(state_batch, next_piece_batch)[0][indexes,0,action_batch[:-1]]
        max_actions = torch.argmax(self.local_Q(state_batch[1:], next_piece_batch[1:])[0], dim=2).squeeze(1)

        with torch.no_grad():
            evaluated = self.target_Q(state_batch[1:], next_piece_batch[1:])[0][indexes,0,max_actions]
            evaluated = reward_batch[:-1] + self.gamma * evaluated
            evaluated[-1] = reward_batch[-2] # if done, there will be no next state
        
        self.optimizer.zero_grad()
        self.loss(prediction, evaluated).to(device).backward()
        self.optimizer.step()

        for target_param, local_param in zip(self.target_Q.parameters(), self.local_Q.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)

        self.eps_start = max(self.eps_end, self.eps_decay * self.eps_start)
        
    def save(self, filename):
        pickle_out = open(filename+".tt","wb")
        pickle.dump(self, pickle_out)
        pickle_out.close()


def load_agent(filename):
    pickle_in = open(filename, mode="rb")
    agent = pickle.load(pickle_in)
    pickle_in.close()
    return agent

"""
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import math
import sys
import pickle

device = torch.device("cuda")


class Network(nn.Module):

    def __init__(self, n_actions):
        super(Network, self).__init__()
        self.cnn = CNN(n_actions)
        self.rnn = nn.LSTM(1447, 256, 1)
        self.out = nn.Linear(256, n_actions)
        self.to(device)

    def forward(self, state, next_piece):
        self.rnn.flatten_parameters()
        features = torch.cat([self.cnn(state), next_piece], dim=2)
        output, _ = self.rnn(features, self.init_hidden())
        return self.out(output)

    def init_hidden(self, sz=256):
        return [torch.zeros((1,1,sz), device=device, dtype=torch.float)]*2


class CNN(nn.Module):
    
    def __init__(self, out_size):
        super(CNN, self).__init__()
        in_channels = 8
        self.convR = nn.Conv2d(in_channels, 8, (20,1))
        self.convC = nn.Conv2d(in_channels, 8, (1,10))
        self.conv1 = nn.Conv2d(in_channels, 16, 5, padding=2)
        self.block1 = nn.Sequential(
            nn.Conv2d(16, 24, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

    def forward(self, state):
        xR = torch.relu(self.convR(state)).view(state.size(0), 1, -1)
        xC = torch.relu(self.convC(state)).view(state.size(0), 1, -1)
        x = torch.relu(self.conv1(state))
        x = self.block1(x).view(state.size(0), 1, -1)
        return torch.cat([xR, xC, x], dim=2)


class Agent():
    
    def __init__(self, num_actions, gamma=0.992, alpha=5e-4, tau=1e-3):
        self.local_Q = Network(num_actions)
        self.target_Q = Network(num_actions)
        self.target_Q.load_state_dict(self.local_Q.state_dict())
        self.target_Q.eval()
        self.optimizer = optim.Adam(self.local_Q.parameters(), lr=alpha)
        self.loss = nn.SmoothL1Loss()
        self.num_actions = num_actions
        self.tau = tau
        self.gamma = gamma
        self.scores = []
        self.episodes = []
        self.durations = []
        self.start = 1
        self.state_memory = []
        self.next_piece_memory = []
        self.action_memory = []
        self.reward_memory = []

    def select_action(self, state, next_piece):
        obs = torch.tensor(state, device=device, dtype=torch.float).unsqueeze(0)
        next_piece = torch.tensor(next_piece, device=device, dtype=torch.float).view(1,1,7)
        output = self.local_Q(obs, next_piece)
        action = torch.argmax(output).item()
        self.state
        self.next_piece_memory.append(next_piece)
        self.action_memory.append(action)
        return action

    def learn(self, trajectory):
        batch = list(zip(*trajectory))

        state_batch = torch.tensor(self.state_memory, device=device, dtype=torch.float)
        next_piece_batch = torch.tensor(batch[1], device=device, dtype=torch.float).unsqueeze(1)
        action_batch = torch.tensor(batch[2], device=device)
        reward_batch = torch.tensor(batch[3], device=device)
        done_batch = torch.tensor(batch[4], device=device)

        indexes = np.arange(state_batch.size(0)-1)
        max_actions = torch.argmax(self.local_Q(state_batch[1:], next_piece_batch[1:]), dim=2).squeeze(1)
        prediction = self.local_Q(state_batch[:-1], next_piece_batch[:-1])[indexes,0,action_batch[:-1]]

        with torch.no_grad():
            evaluated = self.target_Q(state_batch[1:], next_piece_batch[1:])[indexes,0,max_actions]
            evaluated = reward_batch[:-1] + self.gamma * evaluated * done_batch[:-1]

        self.optimizer.zero_grad()
        self.loss(prediction, evaluated).to(device).backward()
        self.optimizer.step()

        for target_param, local_param in zip(self.target_Q.parameters(), self.local_Q.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        pickle_out = open(filename+".tt","wb")
        pickle.dump(self, pickle_out)
        pickle_out.close()


def load_agent(filename):
    pickle_in = open(filename, mode="rb")
    agent = pickle.load(pickle_in)
    pickle_in.close()
    return agent


    def learn(self, trajectory):
        batch = list(zip(*trajectory))

        state_batch = torch.tensor(batch[0], device=device, dtype=torch.float)
        next_piece_batch = torch.tensor(batch[1], device=device, dtype=torch.float).unsqueeze(1)
        action_batch = torch.tensor(batch[2], device=device)
        reward_batch = torch.tensor(batch[3], device=device)

        indexes = np.arange(state_batch.size(0))
        prediction = self.local_Q(state_batch, next_piece_batch)[0][indexes,0,action_batch]

        print(action_batch.shape)
        print(reward_batch.shape)
        print(prediction.shape)
        print()
        max_actions = torch.argmax(self.local_Q(state_batch[1:], next_piece_batch[1:])[0], dim=2).squeeze(1)

        with torch.no_grad():
            evaluated = torch.zeros((len(batch[0]),1,self.num_actions), device=device)
            evaluated[indexes[:-1],0,max_actions] = self.target_Q(state_batch[1:], next_piece_batch[1:])[0][indexes[:-1],0,max_actions]
            evaluated[indexes[:-1],0,max_actions] = reward_batch[:-1] + self.gamma * evaluated
            evaluated[-1] = reward_batch[-1]
            #evaluated[-1] = reward_batch[-1] # if done, there will be no next state

        print(evaluated)
        print(reward_batch)
        sys.exit()

"""