from model import MyMLP
import tensorflow as tf
import numpy as np

class DQNAgent:
    """ Implements a basic DQN Algorithm """

    def __init__(self, env, buffer, reward_function = lambda d,r: r):

        # create an initialize model and target_model
        self.model = MyMLP(hidden_units = [128,64,32], output_units = len(env.available_actions))
        self.target_model = MyMLP(hidden_units = [128,64,32], output_units = len(env.available_actions))
        obs = tf.expand_dims(env.reset(),axis=0)
        self.model(obs)
        self.target_model(obs)
        self.target_model.set_weights(np.array(self.model.get_weights(),dtype = object))
        env.close()

        # self.env = env
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

    def select_action_epsilon_greedy(self,epsilon, observations, available_actions):
        """ selects an action using the model and an epsilon greedy policy """

        random_action_where = [np.random.randint(0,100)<epsilon*100 for _ in range(observations.shape[0])]
        print("where: ", random_action_where)
        random_actions = [np.random.Random_State.choice(a,size=(observations.shape[0])) for a in available_actions]
        print("random action: ", random_actions)
        best_actions = self.select_action(observations,available_actions)
        print("best actions: ", best_actions)
        return np.where(random_action_where,random_actions,best_actions)

    def select_action(self,observations, available_actions):
        """ selects the currently best action using the model """
        probs = self.model(observations,training = False)
        # remove all unavailable actions
        probs = tf.gather(probs,available_actions, axis=1, batch_dims = 1)
        # calculate best action
        inx = tf.argmax(probs, axis = -1).numpy()
        return tf.gather(available_actions,inx,axis = 1, batch_dims = 1).numpy()