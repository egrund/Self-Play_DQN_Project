from model import MyMLP
import tensorflow as tf
import numpy as np
import random as rand

class DQNAgent:
    """ Implements a basic DQN Algorithm """

    def __init__(self, envs, buffer):

        # create an initialize model and target_model
        self.model = MyMLP(hidden_units = [128,64,32], output_units = len(envs[0].available_actions))
        self.target_model = MyMLP(hidden_units = [128,64,32], output_units = len(envs[0].available_actions))
        #self.model()
        #self.target_model()
        self.dqn_target.set_weights(np.array(self.dqn.get_weights(),dtype = object))

        self.envs = envs
        self.buffer = buffer

    def train(self,iterations : int, path_save_weights : str, path_save_logs : str):
        """ """

        for i in range(iterations):

            # epsilon decay

            # train the dqn + new sampels
            self.train_episode(i)

            # new sampling + add to buffer

            #buffer.extend()

            # write summary

            print("Iteration: ", i)

    def train_inner_iteration(self,i):
        """ """

        for j in range(inner_iterations): # TODO
            pass

            # sample random minibatch of transitions
            # minibatch = self.buffer.sample_minibatch()

            # train model

            # if prioritized experience replay, then here


        # polyak averaging

        # logs

def select_action_epsilon_greedy(self,epsilon, observations):
        """ selects an action using the model and an epsilon greedy policy """

        random_action = [rand.randint(0,100)<epsilon*100 for _ in range(len(self.envs))]
        if random_action:
            action = rand.randint(0,self.model.output_units)
        else:
            action = tf.argmax(self.model(observation,training = False), axis = -1).numpy()
        return action