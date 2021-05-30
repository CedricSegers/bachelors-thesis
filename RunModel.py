# Let CartPole run with existing model
import gym
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime

#env = gym.make('CartPole-v0')
env = gym.make('CartPole-v1')

Q_table = np.load('/Users/cedricsegers/Desktop/ComputerScience/3Bachelor/Bachelproef/CartPole/SavedModels/9300-qtable.npy')

buckets=(1, 1, 6, 12)
upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50) / 1.]
lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50) / 1.]

sequence = {'states' : [], 'actions' : [], 'rewards' : []}

def discretize_state(obs):
    discretized = list()
    for i in range(len(obs)):
        scaling = (obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i])
        new_obs = int(round((buckets[i] - 1) * scaling))
        new_obs = min(buckets[i] - 1, max(0, new_obs))
        discretized.append(new_obs)
    return tuple(discretized)

def run():
    t = 0
    done = False
    current_state = discretize_state(env.reset())

    sequence['states'].append(current_state)

    while not done:
        sequence['states'].append(current_state)
        env.render()
        t = t+1
        action = np.argmax(Q_table[current_state])
        sequence['actions'].append(action)
        obs, reward, done, _ = env.step(action)
        sequence['rewards'].append(reward)
        new_state = discretize_state(obs)
        current_state = new_state

    now = datetime.now()
    date_time = now.strftime("%H%M%S%m%d%Y")
    np.save(f"sequences/{date_time}-states.npy", sequence['states'])
    np.save(f"sequences/{date_time}-actions.npy", sequence['actions'])
    np.save(f"sequences/{date_time}-rewards.npy", sequence['rewards'])
    return t

print(run())
