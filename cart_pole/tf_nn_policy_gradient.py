#!/usr/bin/env python3
import sys  # Used for getting command line arguments

from math import degrees
import gymnasium as gym
import tensorflow as tf
import keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential


# ALPHA is the learning rate, how much we nudge weights at each update.
# A larger alpha might result in the model never converging on a solution while a
# smaller one might result in the model taking a while to converge.
ALPHA = 1E-3
# GAMMA is the discount, which is used to calculate the return for each
# state + action update.
# Setting this close to 1 (usually between 0.99 and 0.9) rewards the model for
# actions made earlier in the simulation. The idea behind this is to promote
# actions that kept the model 'alive' rather than actions that could have led
# to termination of the environment.
GAMMA = 0.98
HALFWAY_TO_DEATH_ANGLE = degrees(0.2095/2)


class bcolors:
    """Copied from a stackoverflow post, used for printing strings with colors :)"""
    OKCYAN = '\033[96m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


class GradientAscent(tf.Module):
    """A custom optimizer for training the model with gradient ascent rather than
    descent. Can probably delete this and just apply the negative of the gradient
    descent - would use less code and this kind of looks confusing."""

    def __init__(self, learning_rate=1E-2):
        # Initialize parameters
        self.learning_rate = learning_rate
        self.title = f'Gradient descent optimizer: learning rate={self.learning_rate}'

    def apply_gradients(self, grads, vars, coeff=1):
        # Update variables
        for grad, var in zip(grads, vars):
            var.assign_add(self.learning_rate*grad*coeff)


def train_model_in_environment(model, env, weights_save_filename):
    """Given a model and an environment, train the model"""
    episodes_per_save = 100.0
    optimizer = GradientAscent()  # Used for applying the gradients to our model

    average_lifetime_per_save = 0
    episode_num = 1

    while True:  # train forever
        # Initialize all of the loops variables
        terminated = False
        truncated = False
        episode_lifetime = 0  # So we can keep track of how long the model lived in the environment
        observation, info = env.reset()  # Get the initial observation

        while not terminated and not truncated:  # One lifetime of the simulation
            with tf.GradientTape() as tape:
                obs_tensor = tf.convert_to_tensor([observation])
                # print(observation)
                action_probabilities = model(obs_tensor)

                # TODO: why are we taking the log?
                log_probs = tf.math.log(action_probabilities)

                action = tf.argmax(action_probabilities[0]).numpy()

                # TODO: look at the info and other variables to see if they can help improve our training
                observation, _, terminated, truncated, info = env.step(
                    action)

            angle = degrees(observation[2])
            # # reward = HALFWAY_TO_DEATH_ANGLE - abs(angle)
            reward = GAMMA ** (episode_lifetime)

            action_direction = action if action == 1 else -1
            reward2 = 1 if (angle * action_direction > 0) else -1
            # print(
            #     f"pole angle: \t{angle}\n"
            #     f"action: \t{action}\n"
            #     f"action dir: \t{action_direction}\n"
            #     f"reward: \t{reward2}\n"
            # )
            # # TODO: look at including the pole angular velocity in the reward calculation
            # quit()

            # TODO: we eventually want to base the reward off of observation[2]**2
            # print('probs:')
            # print(action_probabilities)
            # print("weights before:")
            # print(model.trainable_weights)

            grads = tape.gradient(
                log_probs, model.trainable_weights)

            # G = GAMMA**episode_lifetime
            optimizer.apply_gradients(
                grads, model.trainable_weights, reward*reward2)
            # print("\nweights after:")
            # print(model.trainable_weights)
            # quit()
            episode_lifetime += 1

        # For useful print output
        average_lifetime_per_save += episode_lifetime/episodes_per_save

        # Save the weights if we've iterated over episodes_per_save episodes
        if episode_num % episodes_per_save == 0:
            # model.save_weights(weights_save_filename)
            print(
                f'did NOT save weights after episode {episode_num}. Average lifetime per {episodes_per_save} episodes is {average_lifetime_per_save}.')
            average_lifetime_per_save = 0

        episode_num += 1


def get_neural_network(weights_load_filename=None):
    """Construct a neural network and load weights if possible.

    The input layer has four nodes to match
    CartPole's observation.
    Two hidden layers, each with 10 nodes and sigmoid activation functions.
    The output layer has two nodes, representing the probability we should
    choose action 0 (move cart left) or action 1 (move cart right). Using a
    softmax activation here to make these accurate probabilities (using
    softmax ensures these add to 1.0)
    """
    neural_network = Sequential()
    neural_network.add(Flatten(input_shape=(4, 1)))
    neural_network.add(keras.layers.Dense(10, activation='sigmoid'))
    neural_network.add(keras.layers.Dense(10, activation='sigmoid'))
    neural_network.add(keras.layers.Dense(2, activation='softmax'))
    # neural_network = keras.Sequential(
    #     [
    #         # keras.layers.Input(shape=(4,)),
    #         Flatten(input_shape=(1, 4)),
    #         keras.layers.Dense(10, activation='sigmoid'),
    #         keras.layers.Dense(10, activation='sigmoid'),
    #         # Sigmoid activation on output to get probability
    #         keras.layers.Dense(2, activation='softmax')
    #     ]
    # )

    # Try to load the weights we passed in a filename
    # if weights_load_filename:
    #     try:
    #         neural_network.load_weights(weights_load_filename)
    #         print(f"Successfully loaded weights from {weights_load_filename}")
    #     except tf.errors.NotFoundError:
    #         print(
    #             f"{bcolors.FAIL}Error loading weights from {weights_load_filename}{bcolors.ENDC}")

    return neural_network


def main():
    """All of the high level code to create and train a model for the CartPole environment.
    Using a main function instead of putting this code in `__name__ == '__main__'`
    block to prevent any variable scope conflicts."""
    # Get optional command line arguments
    num_args = len(sys.argv) - 1  # first element is always the filename
    weights_filename = sys.argv[1] if num_args >= 1 else None
    render_environment = sys.argv[2] if num_args >= 2 else None

    model = get_neural_network(weights_filename)
    # model.summary()
    # quit()

    env = gym.make(
        'CartPole-v1', render_mode='human' if render_environment else None)

    train_model_in_environment(
        model, env=env, weights_save_filename=weights_filename)


if __name__ == '__main__':
    main()
