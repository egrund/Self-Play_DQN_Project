from model import MyMLP
import tensorflow as tf
import numpy as np
import random as rand

class DQNAgent:
    """ Implements a basic DQN Algorithm """

    def __init__(self, env, buffer, reward_function = lambda d,r: r):

        # create an initialize model and target_model
        self.model = MyMLP(hidden_units = [128,64,32], output_units = len(env.available_actions))
        self.target_model = MyMLP(hidden_units = [128,64,32], output_units = len(env.available_actions))
        obs = env.reset()
        self.model(obs)
        self.target_model(obs)
        self.target_model.set_weights(np.array(self.model.get_weights(),dtype = object))

        self.env = env
        self.buffer = buffer
        self.reward_function = reward_function

    def train_inner_iteration(self,i):
        """ """

        #for j in range(inner_iterations): # TODO
        pass

            # sample random minibatch of transitions
            # minibatch = self.buffer.sample_minibatch()

            # train model

            # if prioritized experience replay, then here


        # polyak averaging

        # logs

    def select_action_epsilon_greedy(self,epsilon, observations):
            """ selects an action using the model and an epsilon greedy policy """

            random_action = [rand.randint(0,100)<epsilon*100 for _ in range(len(observations))]
            if random_action:
                action = rand.randint(0,self.model.output_units)
            else:
                action = self.select_action(observations)
            return action

    def select_action(self,observations):
        """ selects the currently best action using the model """
        return tf.argmax(self.model(observations,training = False), axis = -1).numpy()