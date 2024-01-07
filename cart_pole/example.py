#!/usr/bin/env python3
"""Copied from the gymnasium documentation page
https://pypi.org/project/gymnasium/
"""
import gymnasium as gym


# Don't forget the render_mode
env = gym.make('CartPole-v1', render_mode='human')
observation, info = env.reset()


class bcolors:
    OKCYAN = '\033[96m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


for _ in range(1000):
    # agent policy that uses the observation and info
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"terminated: {bcolors.FAIL if terminated else bcolors.OKCYAN}{terminated}{bcolors.ENDC}:\n"
          f"   cart position: {observation[0]}\n"
          f"   cart velocity: {observation[1]}\n"
          f"   pole angle: {observation[2]*180/3.1415}\n"
          f"   pole angular velocity: {observation[3]}\n"
          )

    if terminated or truncated:
        observation, info = env.reset()

env.close()
