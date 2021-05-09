import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy.ndimage.filters import uniform_filter1d
from environment import Environment

EPISODES = 1200
INPUT_SIZE = 25
BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 1
EPS_END = 0.01
EPS_DECAY = EPISODES*3
TARGET_UPDATE = 10
LR = 1e-4

class Agent:
    def __init__(self, policy_net, target_net):
        self.step_count = 0
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optim.Adam(self.policy_net.parameters(), LR)

    def choose_action(self, state):
        threshold = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * self.step_count / EPS_DECAY)
        random_sample = np.random.random_sample()
        self.step_count += 1
        # print(threshold)

        if random_sample > threshold:
            # get the action from the policy network
            with torch.no_grad():
                return self.policy_net(state).argmax().item()
        else:
            # get the action by random decision
            return np.random.choice([0, 1])

    def optimize_model(self, replay_memory):
        states, actions, \
                    rewards, next_states, dones = replay_memory
        states = states.reshape((BATCH_SIZE, INPUT_SIZE))
        next_states = next_states.reshape((BATCH_SIZE, INPUT_SIZE))
        # get the q-values for the selected actions
        current_Q = self.policy_net(states).gather(1,
                        actions.unsqueeze(1).type(torch.int64))
        next_Q = torch.zeros(BATCH_SIZE)
        # if the episode is done we want the next_Q value to remain zero
        # and not let the target network predict the max-Q value
        mask = torch.where(dones == 0)[0]
        next_Q[mask] = self.target_net(next_states[mask]).max(1)[0].detach()
        next_Q = (next_Q * GAMMA) + rewards
        
        # calculate the loss between the current Q values and the next Q values
        loss_function = nn.MSELoss()
        loss = loss_function(current_Q, next_Q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self):
        torch.save(self.policy_net.state_dict(), 'policy_weights.pth')


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.store_count = 0

    def store_experience(self, experience):
        """Stores an experience into the replay memory. If the replay memory
        is full, overwrite the previous memory."""
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.store_count % self.capacity] = experience
        self.store_count += 1

    def sample_memory(self):
        """Returns a random sample of experiences from memory."""
        batch = random.sample(self.memory, BATCH_SIZE)
        batch = Experience(*zip(*batch))

        s = torch.cat(batch.State)
        a = torch.cat(batch.Action)
        r = torch.cat(batch.Reward)
        n_s = torch.cat(batch.Next_State)
        d = torch.cat(batch.Done)

        return (s, a, r, n_s, d)


def get_model():
    """Returns neural network model with a binary sigmoid output. Could've
    used a class here with inheritcance of nn.module, but this works fine."""
    model = nn.Sequential(
        nn.Linear(INPUT_SIZE, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 2, False),
        nn.Sigmoid()
    )
    return model

if __name__ == '__main__':
    env = Environment()
    # create a policy network and clone its weights into a target network
    policy_net = get_model()
    target_net = get_model()
    target_net.load_state_dict(policy_net.state_dict())

    # this network will not be trained
    target_net.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(policy_net.to(device), target_net.to(device))

    # initialize initial state after an episode has started
    state = torch.zeros(INPUT_SIZE)
    # create a namedtuple for easy storage of replay memory and create 10000
    # storage spaces for experiences
    replay = ReplayMemory(10000)
    Experience = namedtuple('Experience',
                    ('State', 'Action', 'Reward', 'Next_State', 'Done'))
    rewards_per_episode = []
    rewards_episode = 0
    for episode in range(EPISODES):
        env.__init__()
        alive = True
        print('Episode: {}, Rewards Last Episode: {}'.format(
            episode, rewards_episode))
        rewards_episode = 0
        while alive:
            # choose an action and perform it
            # action = 0
            action = agent.choose_action(state)
            done, next_state, reward = env.step(action)

            # however, for the 4 frames thereafter we don't do anything and
            # observe our action. This gives more valuable input to the
            # network as to what the action leads to in the game. We
            # concatenate this all into one state.
            for frame in range(4):
                if not done:
                    done, temp_state, reward_temp = env.step(0)
                    next_state = np.vstack((next_state, temp_state))
                    reward += reward_temp
                else:
                    next_state = np.vstack((next_state, np.zeros(5)))
            # flatten and convert to tensor so we can use it as input
            next_state = torch.from_numpy(next_state.flatten()).float()
            # store the state, action, reward and next state into replay memory
            replay.store_experience(
                Experience(state, torch.Tensor([action]),
                    torch.Tensor([reward]), next_state, torch.Tensor([done]))
            )
            rewards_episode += reward
            state = next_state

            if len(replay.memory) >= BATCH_SIZE:
                agent.optimize_model(replay.sample_memory())

            if done:
                alive = False
        rewards_per_episode.append(rewards_episode)
        # update the target network
        if episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
    agent.save_model()
    running_mean = uniform_filter1d(rewards_per_episode, 100)
    plt.plot(rewards_per_episode, label='Rewards')
    plt.plot(running_mean, label='Rewards running mean')
    plt.title('Rewards per episode')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend()
    plt.show()
