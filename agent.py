from model import MyMLP
import tensorflow as tf
import numpy as np
import random as rnd

class DQNAgent:
    """ Implements a basic DQN Algorithm """

    def __init__(self, env, buffer, batch : int, reward_function = lambda d,r: r, polyak_update = 0.9, inner_iterations = 10):

        # create an initialize model and target_model
        self.model = MyMLP(hidden_units = [128,64,32], output_units = len(env.available_actions))
        self.target_model = MyMLP(hidden_units = [128,64,32], output_units = len(env.available_actions))
        obs = tf.expand_dims(env.reset(),axis=0)
        self.model(obs)
        self.target_model(obs)
        self.target_model.set_weights(np.array(self.model.get_weights(),dtype = object))
        self.model.model_target = self.target_model
        self.inner_iterations = inner_iterations
        self.polyak_update = polyak_update
        env.close()

        # self.env = env
        self.buffer = buffer
        self.reward_function = reward_function

    def train_inner_iteration(self, summary_writer, i):
        """ """
        for j in range(self.inner_iterations):

            # sample random minibatch of transitions
            minibatch = self.buffer.sample_minibatch(self.batch)  #Out:[state, action, reward, next_state, done]

            # train model  
            print(minibatch[0])
            #training_data = 
            
            #loss = self.dqn.step(s,a,r,s_new,done, self.optimizer, self.dqn_target)


            # if prioritized experience replay, then here


        # polyak averaging
        self.target_model.set_weights((1-self.polyak_update)*np.array(self.target_model.get_weights(),dtype = object) + 
                                      self.polyak_update*np.array(self.model.get_weights(),dtype = object))

        # logs
        if train_summary_writer:
            with train_summary_writer.as_default():
                tf.summary.scalar(m.name, loss, step=j+i*self.inner_iterations)
        if test_summary_writer:
            with test_summary_writer.as_default():
                tf.summary.scalar(m.name, loss, step=j+i*self.inner_iterations)
                
                
    def select_action_epsilon_greedy(self,epsilon, observations, available_actions):
        """ 
        selects an action using the model and an epsilon greedy policy 
        
        Parameters:
            epsilon (float):
            observations (array): (batch, 7, 7) using FourConnect
            available_actions (list): containing all the available actions for each batch observation
        
        returns: 
            the chosen action for each batch element
        """

        random_action_where = [np.random.randint(0,100)<epsilon*100 for _ in range(observations.shape[0])]
        random_actions = [np.random.choice(a) for a in available_actions]
        best_actions = self.select_action(observations,available_actions).numpy()
        return np.where(random_action_where,random_actions,best_actions)

    @tf.function
    def select_action(self,observations, available_actions):
        """ selects the currently best action using the model """
        probs = self.model(observations,training = False)
        # remove all unavailable actions
        probs = tf.gather(probs,available_actions, axis=1, batch_dims = 1)
        # calculate best action
        inx = tf.argmax(probs, axis = -1)
        # get best action for each batch element
        return tf.gather(available_actions,inx,axis = 1, batch_dims = 1)
