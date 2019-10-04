import time

import gym
import matplotlib.pyplot as plt
import numpy as np

from agent import Agent
from ddpg import ddpg
from settings import *

if __name__ == '__main__':

    def train():
        agent, scores = ddpg()

        plt.plot(np.arange(1, len(scores) + 1), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()

        return agent


    def show(agent: Agent):
        env = gym.make(ENV_NAME)

        state = env.reset()
        for t in range(900):
            action = agent.act(state)
            env.render()
            state, reward, done, _ = env.step(action)
            time.sleep(0.01)

            if done:
                print("Win!!! episode: {}".format(t))
                break
        else:
            print("Fail!!!")
        env.close()


    def estimate(agent: Agent):
        env = gym.make(ENV_NAME)
        reward_sum = 0
        for _ in range(100):
            state = env.reset()
            for t in range(900):
                action = agent.act(state)
                state, reward, done, _ = env.step(action)
                reward_sum += reward

                if done:
                    break
        env.close()
        return reward_sum / 100


    def load():
        agent = Agent()
        agent.load()
        return agent


    agent = train()
    print(estimate(agent))
    show(agent)
