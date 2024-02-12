#!/usr/bin/env python3

import tensorflow as tf
import keras
from keras import __version__  # noqa
tf.keras.__version__ = __version__  # noqa
import gymnasium as gym  # noqa
import numpy as np  # noqa

from keras import __version__  # noqa
tf.keras.__version__ = __version__  # noqa
from rl.agents import DQNAgent  # noqa

from rl.memory import SequentialMemory  # noqa
from rl.policy import BoltzmannQPolicy  # noqa

from tensorflow.keras.optimizers import Adam  # noqa
from tensorflow.keras.layers import Dense, Flatten  # noqa
from tensorflow.keras.models import Sequential  # noqa

Adam._name = "sldkfjsldkfjlsdkjf"
# import keras


# import random

env = gym.make('CartPole-v1', render_mode="human")
states = env.observation_space.shape[0]
actions = env.action_space.n

# # Running environment with random choices:
# episodes = 5
# for episode in range(1, episodes+1):
#     state = env.reset()
#     print(f"state: {state}")
#     done = False
#     score = 0

#     while not done:
#         env.render()
#         action = random.choice([0, 1])
#         n_state, reward, done, _, info = env.step(action)
#         score += reward
#     print(f"Episode:{episode} Score:{score}")


def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


model = build_model(states, actions)
# model.summary()


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50_000, window_length=1)
    _dqn = DQNAgent(model=model, memory=memory, policy=policy,
                    nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return _dqn


# dqn = build_agent(model, actions)
# dqn.compile(tf.keras.optimizers.legacy.Adam(
#     learning_rate=1e-3), metrics=['mae'])
# dqn.fit(env, nb_steps=50_000, visualize=False, verbose=1)


# # Running environment with random choices:
# episodes = 5
# for episode in range(1, episodes+1):
#     state = env.reset()
#     print(f"state: {state}")
#     done = False
#     score = 0

#     while not done:
#         env.render()
#         # action = random.choice([0, 1])
#         obs_tensor = tf.convert_to_tensor([state])
#         # print(observation)
#         action_probabilities = model(obs_tensor)

#         action = tf.argmax(action_probabilities[0]).numpy()

#         n_state, reward, done, _, info = env.step(action)
#         score += reward
#     print(f"Episode:{episode} Score:{score}")
