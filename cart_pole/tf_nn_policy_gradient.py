#!/usr/bin/env python3
import sys  # Used for getting command line arguments

import gymnasium as gym
import tensorflow as tf
import keras


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


class bcolors:
    """Copied from a stackoverflow post, used for printing strings with colors :)"""
    OKCYAN = '\033[96m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


class GradientAscent(tf.Module):
    """A custom optimizer for training the model with gradient ascent rather than
    descent. Can probably delete this and just apply the negative of the gradient
    descent - would use less code and this kind of looks confusing."""

    def __init__(self, learning_rate=GAMMA):
        # Initialize parameters
        self.learning_rate = learning_rate
        self.title = f'Gradient descent optimizer: learning rate={self.learning_rate}'

    def apply_gradients(self, grads, vars, coeff=1):
        # Update variables
        for grad, var in zip(grads, vars):
            # I couldn't find where I copied this custom implementation from,
            # but in that person's code they used assign_sub to apply gradient
            # descent.
            # Also I think technically multiplying by the learning rate here
            # isn't needed since we already calculated the returns with the
            # learning rate, but I guess this is how the equation is written,
            # and the operation is being performed for all states so it shouldn't
            # hurt.
            var.assign_add(self.learning_rate*grad*coeff)


def train_model_in_environment(model, env, weights_save_filename, verbose=True):
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
        observation, _ = env.reset()  # Get the initial observation

        # Keeping track of the log probabilities, rewards, and returns at each state in the
        # model's lifetime so we can update the model at the end of its lifetime.
        # It would be fine to do this at each state, but at each state we can't calculate
        # the return, since we don't know how long that instance of the model is going to
        # live.
        log_probs = []
        rewards = []  # the CartPole environment returns 1 for every observation
        # Returns will eventually look something like:
        # `[8.312611893492505, 7.461848870910719, 6.593723337664, 5.7078809567999995,
        # 4.80396016, 3.881592, 2.9404, 1.98, 1.0]`.
        # Which considers actions made earlier in the episode more "important"
        returns = []

        # Wrapping the following code like this allows `tape` to watch the variables
        # we update on our model, which allows it to compute the gradients later on.
        with tf.GradientTape() as tape:
            # Run the simulation with the model until terminated
            while not terminated and not truncated:
                # Forward pass of the neural network
                action_probabilities = model(
                    tf.convert_to_tensor([observation]))

                # Keep track of the log of action probabilities to take the gradient
                # of later on
                log_probs.append(tf.math.log(action_probabilities))

                # Get the action with the higher probability
                action = tf.argmax(action_probabilities[0]).numpy()

                # Take that action and get info for the next iteration
                # (but use the reward from that action in a second)
                observation, reward, terminated, truncated, _ = env.step(
                    action)
                rewards.append(reward)
                episode_lifetime += 1

            # Calculate `G` for this action/state and keep track of it for later
            cumulative_return = 0
            for r in rewards[::-1]:
                cumulative_return = r + GAMMA * cumulative_return
                returns.insert(0, cumulative_return)

            # Multiply the returns by the log probabilities of the actions that resulted
            # in those returns.
            # The equation for policy gradient update has the return outside of the gradient,
            # but this should be doing the same thing
            log_probs = tf.concat(log_probs, axis=0)
            returns = tf.convert_to_tensor(returns, dtype=tf.float32)
            policy_loss = tf.reduce_sum(
                log_probs * tf.expand_dims(returns, axis=1))

        # Finally use the GradientTape to get the gradient of the policy loss with
        # respect to the models weights.
        # Using the word `loss` in the variable that we take the gradient of is
        # unintuitive, because policy_loss here is more a measure of how good the
        # model did.
        # So we are getting the gradients to move the weights in directions that
        # increase the returns.
        grads = tape.gradient(policy_loss, model.trainable_weights)
        # Apply the gradients and update the model :)
        optimizer.apply_gradients(grads, model.trainable_weights)

        # For useful print output
        average_lifetime_per_save += episode_lifetime/episodes_per_save

        # Save the weights if we've iterated over episodes_per_save episodes
        if episode_num % episodes_per_save == 0:
            model.save_weights(weights_save_filename)
            print(
                f'Saved weights after episode {episode_num}. Average lifetime per {episodes_per_save} episodes is {average_lifetime_per_save}.')
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
    neural_network = keras.Sequential([
        keras.layers.Input(shape=(4,)),
        keras.layers.Dense(10, activation='sigmoid'),
        keras.layers.Dense(10, activation='sigmoid'),
        # Sigmoid activation on output to get probability
        keras.layers.Dense(2, activation='softmax')
    ])

    # Try to load the weights we passed in a filename
    if weights_load_filename:
        try:
            neural_network.load_weights(weights_load_filename)
            print(f"Successfully loaded weights from {weights_load_filename}")
        except:
            print(
                f"{bcolors.FAIL}Error loading weights from {weights_load_filename}{bcolors.ENDC}")

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

    env = gym.make(
        'CartPole-v1', render_mode='human' if render_environment else None)

    train_model_in_environment(
        model, env=env, weights_save_filename=weights_filename)


if __name__ == '__main__':
    main()
