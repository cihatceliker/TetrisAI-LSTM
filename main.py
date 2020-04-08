from environment import Environment
from agent import Agent, load_agent
import torch
import numpy as np
import math
import pickle
import sys

num_actions = 6
num_iter = 50000000
print_interval = 10
save_interval = 100

env = Environment()
agent = Agent(num_actions) if len(sys.argv) == 1 else load_agent(sys.argv[1])  

def count(tetrises):
    s = {}
    for i in range(1,5):
        s[i] = len([j for j in tetrises if j == i])
    return str(s)

#agent.loss = torch.nn.MSELoss()
agent.optimizer = torch.optim.Adam(agent.local_Q.parameters(), 1e-4)
print(agent.optimizer)
print(agent.loss)

all_tetrises = []
for episode in range(agent.start, num_iter):
    done = False
    score = 0
    ep_duration = 0
    state, next_piece = env.reset()
    trajectory = []
    agent.init_hidden()

    while not done:
        action = agent.select_action(state, next_piece)
        next_state, reward, done, next_next_piece = env.step(action)
        trajectory.append([state, next_piece, action, reward])
        state = next_state
        next_piece = next_next_piece
        score += reward
        ep_duration += 1

    agent.learn(trajectory)
    all_tetrises += env.tetrises

    agent.episodes.append(episode)
    agent.scores.append(score)
    agent.durations.append(ep_duration)
    agent.start = episode

    if ep_duration > 1500 or 4 in env.tetrises:
        pickle_out = open(str(ep_duration)+str(4 in env.tetrises)+".ep","wb")
        pickle.dump(trajectory, pickle_out)
        pickle_out.close()
        print("trajectory saved to : " + str(ep_duration)+str(4 in env.tetrises)+".ep")

    if episode % print_interval == 0 or ep_duration > 1500:
        avg_score = np.mean(agent.scores[max(0, episode-print_interval):(episode+1)])
        avg_duration = np.mean(agent.durations[max(0, episode-print_interval):(episode+1)])
        if episode % save_interval == 0:
            agent.start = episode + 1
            agent.save(str(episode))
        print("Episode: %d - Avg. Duration: %d - Avg. Score: %3.3f - %s" % (episode, avg_duration, avg_score, count(all_tetrises)))
        all_tetrises = []
        #$print("Episode: %d - Avg. Duration: %d - Avg. Score: %3.3f" % (episode, avg_duration, avg_score))
