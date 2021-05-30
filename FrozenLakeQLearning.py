import gym
import numpy as np
import math
import matplotlib.pyplot as plt

num_episodes = 1000

env = gym.make('FrozenLake-v0')
holes = (5,7,11,12)

min_lr=0.1
min_epsilon=0.1
discount=1.0
decay=25

Q_table = np.zeros((env.observation_space.n, env.action_space.n))

ep_rewards = []
aggr_ep_rewards = {'ep' : [], 'avg' : [], 'min' : [], 'max' : []}
sequence = {'states' : [], 'actions' : [], 'rewards' : []}

def choose_action(state):
    if (np.random.random() < epsilon):
        return env.action_space.sample()
    else:
        return np.argmax(Q_table[state])

def update_q(state, action, reward, new_state):
    Q_table[state][action] +=  learning_rate * (reward +  discount * np.max(Q_table[new_state]) - Q_table[state][action])

def get_epsilon(t):
    return max(min_epsilon, min(1., 1. - math.log10((t + 1) / decay)))

def get_learning_rate(t):
    return max(min_lr, min(1., 1. - math.log10((t + 1) / decay)))



########### TRAINING PROCESS ###############

for episode in range(num_episodes):

    episode_reward = 0

    render = True

    current_state = env.reset()
    learning_rate = get_learning_rate(episode)
    epsilon = get_epsilon(episode)

    done = False

    while not done:

        action = choose_action(current_state)

        obs, reward, done, _ = env.step(action)

        episode_reward += reward
        new_state = obs

        if new_state == 15:
            print("Goal reached")

        if new_state in holes:
            print(f"Fell in hole {new_state}")
        update_q(current_state, action, reward, new_state)
        current_state = new_state


    print(f"Agent received a reward of {episode_reward} in episode {episode}")

env.close()

np.save(f"qtables/qtable.npy", Q_table)
