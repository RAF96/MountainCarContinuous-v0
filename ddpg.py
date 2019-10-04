from collections import deque

import gym
import numpy as np

from agent import Agent
from settings import ENV_NAME, GAMMA, POTENTIAL_FUNCTION_COEF

n_episodes = 150
max_t = 900
rate_of_print = 1


def ddpg():
    scores = []
    env = gym.make(ENV_NAME)
    agent = Agent(state_size=2, action_size=1)

    for i_episode in range(n_episodes):
        score = 0
        done = False
        state = env.reset()

        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            modified_reward = reward + \
                              POTENTIAL_FUNCTION_COEF * (GAMMA * abs(next_state[1]) - abs(state[1]))

            agent.step(state, action, modified_reward, next_state, done)
            score += reward
            state = next_state
            if done:
                break

        scores.append(score)

        if i_episode % rate_of_print == 0:
            print("Episode: {}. Score: {}, Done: {}".format(i_episode / rate_of_print, score, done))


    return agent, scores
