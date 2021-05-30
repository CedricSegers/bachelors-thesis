import gym
import numpy as np
import math
import matplotlib.pyplot as plt


buckets=(1, 1, 6, 12)
num_episodes=10000

min_lr=0.1
min_epsilon=0.1
discount=1.0
decay=25

SHOW_EVERY = num_episodes / 10
STATS_EVERY = 100
SAVE_MODEL_EVERY = 50

#env = gym.make('CartPole-v0')
# Try with v1?
env = gym.make('CartPole-v1')

upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50) / 1.]
lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50) / 1.]

Q_table = np.zeros(buckets + (env.action_space.n,))

ep_rewards = []
aggr_ep_rewards = {'ep' : [], 'avg' : [], 'min' : [], 'max' : []}

def discretize_state(obs):
    discretized = list()
    for i in range(len(obs)):
        scaling = (obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i])
        new_obs = int(round((buckets[i] - 1) * scaling))
        new_obs = min(buckets[i] - 1, max(0, new_obs))
        discretized.append(new_obs)
    return tuple(discretized)

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


for episode in range(num_episodes):

    episode_reward = 0

    if not episode % SHOW_EVERY:
        render = True
    else:
        render = False

    current_state = discretize_state(env.reset())
    learning_rate = get_learning_rate(episode)
    epsilon = get_epsilon(episode)

    done = False

    while not done:

        action = choose_action(current_state)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
        new_state = discretize_state(obs)
        update_q(current_state, action, reward, new_state)
        current_state = new_state

        if render:
            env.render()

    ep_rewards.append(episode_reward)
    # Save the Q-table
    # This way after training we can use the model == the Q-Table at the episode we want
    if not episode % SAVE_MODEL_EVERY:
        np.save(f"qtables/{episode}-qtable.npy", Q_table)

    if not episode % STATS_EVERY:
        average_reward = sum(ep_rewards[-STATS_EVERY:])/len(ep_rewards[-STATS_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
        print(f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-STATS_EVERY:])}, max: {max(ep_rewards[-STATS_EVERY:])}, current epsilon: {epsilon:>1.2f}" )

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
plt.legend(loc=4)
plt.show()
