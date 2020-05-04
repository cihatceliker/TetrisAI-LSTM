import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"


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

    # returns two zero tensors for the initial state of the lstm
    def init_hidden(self, sz=256):
        return [torch.zeros((1,1,sz), device=device, dtype=torch.float)]*2


class CNN(nn.Module):
    
    def __init__(self, out_size):
        super(CNN, self).__init__()
        in_channels = 4
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
    
    def __init__(self, num_actions, gamma=0.997, alpha=1e-3, tau=1e-3):
        global device
        print("runs on %s." % device)
        device = torch.device(device)
        
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

    def init_hidden(self):
        self.hidden = self.local_Q.init_hidden()
        
    def select_action(self, state, next_piece):
        self.local_Q.eval()
        with torch.no_grad():
            obs = torch.tensor(state, device=device, dtype=torch.float).unsqueeze(0)
            next_piece = torch.tensor(next_piece, device=device, dtype=torch.float).view(1,1,7)
            output, self.hidden = self.local_Q(obs, next_piece, self.hidden)
            action = torch.argmax(output).item()
        self.local_Q.train()
        return action

    def learn(self, trajectory):
        # trajectory is an array of [state, next_piece, action, reward]
        #                           [state, next_piece, action, reward]
        #                                            .
        #                                            .
        #                                            .
        # transposing the trajectory gives an array of 
        #                           [state,      ..., state     ]
        #                           [next_piece, ..., next_piece]
        #                           [action,     ..., action    ]
        #                           [reward,     ..., reward    ]
        batch = list(zip(*trajectory))

        state_batch = torch.tensor(batch[0], device=device, dtype=torch.float)
        next_piece_batch = torch.tensor(batch[1], device=device, dtype=torch.float).unsqueeze(1)
        action_batch = torch.tensor(batch[2], device=device)
        reward_batch = torch.tensor(batch[3], device=device)
        
        # implementation of the DoubleDQN algorithm.
        indexes = np.arange(state_batch.size(0))
        prediction = self.local_Q(state_batch, next_piece_batch)[0][indexes,0,action_batch]
        evaluated = torch.zeros_like(prediction)
        max_actions = torch.argmax(self.local_Q(state_batch[1:], next_piece_batch[1:])[0], dim=2).squeeze(1)

        with torch.no_grad():
            evaluated[:-1] = self.target_Q(state_batch[1:], next_piece_batch[1:])[0][indexes[:-1],0,max_actions] * self.gamma
            evaluated += reward_batch

        self.optimizer.zero_grad()
        self.loss(prediction, evaluated).to(device).backward()
        self.optimizer.step()

        # smoothly updating the target network
        for target_param, local_param in zip(self.target_Q.parameters(), self.local_Q.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)
        
    def save(self, filename):
        pickle_out = open(filename+".tt","wb")
        pickle.dump(self, pickle_out)
        pickle_out.close()
    
    def save_brain(self, filename):
        d = {}
        d["local"] = self.local_Q.state_dict()
        d["target"] = self.target_Q.state_dict()
        torch.save(d, filename)    

    def load_brain(self, filename):
        d = torch.load(filename, map_location=device)
        self.local_Q.load_state_dict(d["local"])
        self.target_Q.load_state_dict(d["target"])


def load_agent(filename):
    pickle_in = open(filename, mode="rb")
    agent = pickle.load(pickle_in)
    pickle_in.close()
    return agent
