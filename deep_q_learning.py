import numpy as np
import torch
import random
import gym
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import pickle
import math


ENV = "BipedalWalker-v3"
MODEL_FILE = "./dqn_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
N_GAMES = 1000
MEM_SIZE = 1000000
BATCH_SIZE = 64
TARGET_UPDATE = 2
GAMMA = 0.99
EPSILON = 1
EPSILON_DEC = 1e-3
EPSILON_MIN = 0.05
LR = 1e-4

steps_taken = 0

class ExperienceReplay:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def store_transition(self, state, action, reward, new_state, done):
        self.buffer.append((state, action, reward, new_state, done))

    def sample(self):
        sample = random.sample(self.buffer, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*sample)
        states = torch.tensor(states).float().to(DEVICE)
        actions = torch.stack(actions).long().to(DEVICE)
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32).reshape(-1, 1)).to(DEVICE)
        next_states = torch.tensor(next_states).float().to(DEVICE)
        dones = torch.from_numpy(np.array(dones, dtype=np.uint8).reshape(-1, 1)).float().to(DEVICE)
        return (states, actions, rewards, next_states, dones)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(QNetwork, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = self.l1(state)
        x = F.relu(x)
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x

class Agent():
    def __init__(self, state_space, action_space):
        self.memory = ExperienceReplay(MEM_SIZE)
        self.action_space = action_space
        self.main_model = QNetwork(state_space.shape[0], action_space.shape[0], action_space.high[0]).to(DEVICE)
        self.target_model = QNetwork(state_space.shape[0], action_space.shape[0], action_space.high[0]).to(DEVICE)
        self.optimizer = optim.Adam(self.main_model.parameters(), lr=LR)
        self.target_model.load_state_dict(self.main_model.state_dict())
        self.target_model.eval()

    def step(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        if len(self.memory) > BATCH_SIZE:
            self.learn()

    def learn(self):
        state, action, reward, new_state, done = self.memory.sample()
        q_eval = self.main_model(state)
        q_next = self.target_model(new_state)
        q_target = reward + GAMMA * (q_next) * (1 - done)
        loss = F.mse_loss(q_eval, q_target.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.main_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def choose_action(self, state):
        global steps_taken 
        eps_threshold = EPSILON_MIN + (EPSILON - EPSILON_MIN) * math.exp(-1 * steps_taken / EPSILON_DEC)
        steps_taken += 1
        if np.random.random() < eps_threshold:
            return torch.from_numpy(self.action_space.sample())
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(DEVICE)
            with torch.no_grad():
                return self.main_model(state).flatten().cpu().data

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def main():
    env = gym.make(ENV)
    state_space = env.observation_space
    action_space = env.action_space
    agent = Agent(state_space, action_space)

    visual = input("visualize? [y/n]: ")
    load = input("\nload from model? [y/n]: ")

    if load == "y":
        load_path = input("\npath to load from: ")
        agent.main_model = load_model(agent.main_model, load_path)
        agent.target_model = load_model(agent.main_model, load_path)

    max_score = -10000
    max_game = 0
    scores = []
    start = datetime.datetime.now()

    for game in range(N_GAMES):
        done = False
        score = 0
        observation = env.reset()
        episode_start = datetime.datetime.now()

        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            agent.step(observation, action, reward, next_observation, done)
            score += float(reward)
            observation = next_observation

        if game % TARGET_UPDATE == 0:
            agent.target_model.load_state_dict(agent.main_model.state_dict())

        episode_end = datetime.datetime.now()
        elapsed = episode_end - episode_start
        scores.append(score)
        avg_score = np.mean(scores[-100:])

        if score > max_score:
            max_score = score
            max_game = game
            save_model(agent.main_model, MODEL_FILE)

            if visual == 'y':
                # Visualize the best game (No replay of actions)
                pass

        print(f'game: {game}, reward: {score}, max reward: {max_score} at game {max_game}')
        print(f'average score for the last 100 games: {avg_score}')
        print(f'time: {elapsed.total_seconds()} seconds')

    end = datetime.datetime.now()
    elapsed = end - start
    print(f'Total time: {elapsed.total_seconds()} seconds')

    with open(MODEL_FILE + '_scores', 'wb') as scores_file:
        pickle.dump(scores, scores_file)

main()
