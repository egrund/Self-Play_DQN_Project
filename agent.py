from model import MyCNN_RL
import tensorflow as tf
import numpy as np
import random as rnd
import time

class Agent:
    
    def select_action(self, observations, available_actions, available_actions_bool):
        raise NotImplementedError("select_action has to be implemented to be used")

class DQNAgent(Agent):
    """ Implements a basic DQN Algorithm """

    def __init__(self, env, buffer, batch : int, model_path, polyak_update = 0.9, inner_iterations = 10, reward_function = lambda d,r: r, dropout_rate = 0.5, normalisation : bool = True):

        # create an initialize model and target_model
        self.model = MyCNN_RL(output_units = env.action_space.n, dropout_rate = dropout_rate, normalisation = normalisation)
        self.target_model = MyCNN_RL(output_units = env.action_space.n, dropout_rate = dropout_rate, normalisation = normalisation)
        obs = tf.expand_dims(env.reset(),axis=0)
        self.model(obs)
        self.target_model(obs)
        self.target_model.set_weights(np.array(self.model.get_weights(),dtype = object))
        self.inner_iterations = inner_iterations
        self.polyak_update = polyak_update
        env.close()

        # self.env = env
        self.model_path = model_path
        self.buffer = buffer
        self.batch = batch 
        self.reward_function = reward_function
      
    def train_inner_iteration(self, summary_writer, i):
        """ """
        #start = time.time()
        for j in range(self.inner_iterations):

            # sample random minibatch of transitions
            minibatch = self.buffer.sample_minibatch(self.batch)  #Out:[state, action, reward, next_state, done]
            
            state = tf.convert_to_tensor([sample[0] for sample in minibatch],dtype = tf.float32)
            actions = tf.convert_to_tensor([sample[1] for sample in minibatch],dtype = tf.float32)
            
            new_state = tf.convert_to_tensor([sample[3] for sample in minibatch],dtype = tf.float32)
            done = tf.convert_to_tensor([sample[4] for sample in minibatch],dtype = tf.float32)
            reward = self.reward_function(tf.cast(done, dtype = tf.bool), tf.convert_to_tensor([sample[2] for sample in minibatch],dtype = tf.float32))
            
            loss = self.model.train_step((state, actions, reward, new_state, done), self.target_model)

            # if prioritized experience replay, then here
            
        #print("inner_iteration_average per iteration: ", (time.time() - start)/self.inner_iterations)

        # polyak averaging
        self.target_model.set_weights(
            (1-self.polyak_update)*np.array(self.target_model.get_weights(),dtype = object) + 
                                      self.polyak_update*np.array(self.model.get_weights(),dtype = object))
        
        loss_value = loss.get('loss')
        # logs
        if summary_writer:
            with summary_writer.as_default():
                tf.summary.scalar('loss', loss_value, step=i)

        # reset all metrics
        self.model.reset_metrics()
        
        #print("Total_inner_iteration_time: ", time.time() - start, "\n")
        
        
        return loss_value
                
    def select_action_epsilon_greedy(self,epsilon, observations, available_actions, available_actions_bool):
        """ 
        selects an action using the model and an epsilon greedy policy 
        
        Parameters:
            epsilon (float):
            observations (array): (batch, 7, 7) using FourConnect
            available_actions (list): containing all the available actions for each batch observation
            available_actions_bool (list): containing for every index whether the action with this value is in available actions
        
        returns: 
            the chosen action for each batch element
        """

        random_action_where = [np.random.randint(0,100)<epsilon*100 for _ in range(observations.shape[0])]
        random_actions = [np.random.choice(a) for a in available_actions]
        best_actions = self.select_action(tf.convert_to_tensor(observations, dtype=tf.float32), available_actions, available_actions_bool).numpy()
        return np.where(random_action_where,random_actions,best_actions)

    #@tf.function
    def select_action(self, observations, available_actions, available_actions_bool):
        """ selects the currently best action using the model """
        probs = self.model(observations,training = False)
        # remove all unavailable actions
        probs = tf.where(available_actions_bool,probs,-1)
        # calculate best action
        return tf.argmax(probs, axis = -1)
    
    def save_models(self, i):
        """ saves the model and the target model using i as iteration count """

        self.model.save_weights(f"{self.model_path}/model/{i}")
        self.target_model.save_weights(f"{self.model_path}/target_model/{i}")

    def load_models(self, i):
        self.model.load_weights(f"{self.model_path}/model/{i}")
        self.target_model.load_weights(f"{self.model_path}/target_model/{i}")

    def copyAgent(self,env):
        """ 
        Creates an Agent which can be used to sample action, but which cannot be trained (only the models are copied)
        """

        copy = DQNAgent(env, buffer = None, batch =self.batch, model_path = "")
        copy.model.set_weights(np.array(self.model.get_weights(),dtype = object))
        copy.target_model.set_weights(np.array(self.target_model.get_weights(),dtype = object))
        return copy

class RandomAgent (Agent):

    def select_action(self,observations, available_actions, available_actions_bool):
        """ 
        selects an action using the model and an epsilon greedy policy 
        
        Parameters:
            observations (array): (batch, 7, 7) using FourConnect
            available_actions (list): containing all the available actions for each batch observation
            available_actions_bool (list): containing for every index whether the action with this value is in available actions
        
        returns: 
            the chosen action for each batch element
        """

        
        random_actions = [np.random.choice(a) for a in available_actions]
        return tf.convert_to_tensor(random_actions) 
    
    
class MinMax_Agent:
    """ 
        selects an action using the model and an min max policy 
        
        Parameters:
            observations (array): (batch, 7, 7) using FourConnect
            available_actions (list): containing all the available actions for each batch observation
            available_actions_bool (list): containing for every index whether the action with this value is in available actions
        
        returns: 
            the chosen action for each batch element
        """
    def select_action(self, observations, available_actions, available_actions_bool):
        raise NotImplementedError("select_action has to be implemented to be used")