from model import MyMLP
import tensorflow as tf
import numpy as np

class DQNAgent:
    """ Implements a basic DQN Algorithm """

    def __init__(self, env, buffer):

        # create an initialize model and target_model
        self.model = MyMLP(hidden_units = [128,64,32], output_units = len(env.available_actions))
        self.target_model = MyMLP(hidden_units = [128,64,32], output_units = len(env.available_actions))
        #self.model()
        #self.target_model()
        self.dqn_target.set_weights(np.array(self.dqn.get_weights(),dtype = object))

        self.buffer = buffer

    def train(self,iterations : int, path_save_weights : str, path_save_logs : str):

        for i in range(iterations):

            # epsilon decay

            # train the dqn + new sampels
            self.train_episode(i)

            # new sampling + add to buffer

            # write summary

            print("Iteration: ", i)

    def train_episode(self,i):

        for j in range(inner_iterations): # TODO
            pass

            # sample minibatch
            minibatch = self.buffer.sample_minibatch()

            # train model

            # if prioritized replay, then here


        # polyak averaging

        # logs

