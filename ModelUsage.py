# Use the saved model to operate MountainCar

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")

EPISODES = 1000
SHOW_EVERY = 1000
SAVE_EVERY = 1

print("Observation space High: ", env.observation_space.high)
print("Observation space Low: ", env.observation_space.low)
print("Action space: ", env.action_space.n)


DISCRETE_OS_SIZE = [40] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/ DISCRETE_OS_SIZE

q_table = np.load('/Users/cedricsegers/Desktop/ComputerScience/3Bachelor/Bachelproef/qtables/87550-qtable.npy')

success = 0
ep_rewards = []
aggr_ep_rewards = {'ep' : [], 'avg' : [], 'min' : [], 'max' : []}

# Helper function to discretize states
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


# Loop for each episode
for episode in range(EPISODES):
    episode_reward = 0

    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())
    done = False

    while not done:

        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward

        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()

        if new_state[0] >= env.goal_position:
            print(f"We made it on episode {episode}")
            success += 1

        discrete_state = new_discrete_state

    ep_rewards.append(episode_reward)

    if not episode % SAVE_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        print(f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])}, max: {max(ep_rewards[-SHOW_EVERY:])}" )

env.close()

print(f"Number of sucesses: {success}")
percentage = success // EPISODES
print(f"{percentage} succes rate")

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
plt.legend(loc=4)
plt.show()
