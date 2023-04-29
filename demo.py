import numpy as np
import gymnasium as gym
from policy import Policy


# Initialize agent.
agent = [None] * 9
resume = True
if resume:
    for i in range(9):
        agent[i] = Policy(resume, i)
else:
    for i in range(9):
        agent[i] = Policy(resume, i)

# Initialize environment.
env = gym.make("ALE/MsPacman-ram-v5", render_mode='human')
state, info = env.reset()
states = []
episode = 1
reward_sum = 0
avg_reward = 0
high_score = 0

# Training loop.
while True:

    prob = 0
    action = 0
    for i in range(9):
        p = agent[i].policy_forward(state)
        if p > prob:
            prob = p
            action = i
    if np.random.uniform() < 0.95:
        action = env.action_space.sample()
    agent[action].probs[-1] += 1

    state, reward, terminated, truncated, info = env.step(action)
    states.append(state)
    reward_sum += reward

    if terminated or truncated:
        if reward_sum > 400:
            for i in range(9):
                agent[i].policy_backward(reward_sum, states, episode)
        else:
            for i in range(9):
                agent[i].policy_backward(-1, states, episode)

        # Measure performance.
        if reward_sum > high_score:
            high_score = reward_sum
        avg_reward += reward_sum / 100

        # Save model if a large number of episodes has passed.
        if episode % 100 == 0:
            for i in range(9):
                agent[i].save_model(i)
            print(f'{episode - 99} - {episode}\t| AVG rewards: {int(avg_reward)}\t| High score: {int(high_score)}')
            print('----------------------------------------------------------')
            avg_reward = 0
            high_score = 0

        # Reset environment.
        episode += 1
        reward_sum = 0
        states = []
        state, info = env.reset()
