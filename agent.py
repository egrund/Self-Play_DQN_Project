from model import MyCNN_RL
import tensorflow as tf
import numpy as np
import random as rnd
import time

class Agent:
    
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
        raise NotImplementedError("select_action has to be implemented to be used")

class DQNAgent(Agent):
    """ Implements a basic DQN Algorithm """

    def __init__(self, env, buffer, batch : int, model_path, polyak_update = 0.9, inner_iterations = 10, reward_function = lambda d,r: r, 
                 conv_kernel = [3], filters = 128, hidden_units = [64], dropout_rate = 0.5, normalisation : bool = True,prioritized_experience_replay : bool = True, 
                 gamma : tf.constant = tf.constant(0.99),loss_function = tf.keras.losses.MeanSquaredError(), output_activation = None):

        # create an initialize model and target_model
        self.model = MyCNN_RL(conv_kernel = conv_kernel, filters  = filters, hidden_units = hidden_units, output_units = env.action_space.n, 
                              output_activation = output_activation, loss = loss_function,
                              dropout_rate = dropout_rate, normalisation = normalisation, gamma = gamma)
        self.target_model = MyCNN_RL(conv_kernel = conv_kernel, filters  = filters, hidden_units = hidden_units, output_units = env.action_space.n, 
                                     output_activation = output_activation, loss = loss_function,
                              dropout_rate = dropout_rate, normalisation = normalisation, gamma = gamma)
        
        # build models
        obs = tf.expand_dims(env.reset(),axis=0)
        env.close()
        self.model(obs)
        self.target_model(obs)
        self.target_model.set_weights(np.array(self.model.get_weights(),dtype = object))
        self.inner_iterations = inner_iterations

        # save other variables as attributes
        self.polyak_update = polyak_update
        self.model_path = model_path
        self.buffer = buffer
        self.batch = batch 
        self.reward_function = reward_function
        self.prioritized_experience_replay = prioritized_experience_replay

        self.do_random = np.array([],dtype = np.int32)
      
    def train_inner_iteration(self, summary_writer, i, unavailable_actions_in):
        """ """
        u_p = 0
        
        for j in range(self.inner_iterations):

            # sample random minibatch of transitions
            minibatch = self.buffer.sample_minibatch(self.batch)  #Out:[state, action, reward, next_state, done]
            
            state = tf.convert_to_tensor([sample[0] for sample in minibatch],dtype = tf.float32)
            actions = tf.convert_to_tensor([sample[1] for sample in minibatch],dtype = tf.float32)
            
            new_state = tf.convert_to_tensor([sample[3] for sample in minibatch],dtype = tf.float32)
            done = tf.convert_to_tensor([sample[4] for sample in minibatch],dtype = tf.float32)
            reward = self.reward_function(tf.cast(done, dtype = tf.bool), tf.convert_to_tensor([sample[2] for sample in minibatch],dtype = tf.float32))
            a_action = [sample[5] for sample in minibatch]

            loss = self.model.train_step((state, actions, reward, new_state, done, a_action), self)

            
            # if prioritized experience replay, then here
            if self.prioritized_experience_replay:
                TD_error = self.calc_td_error(state, actions, reward, new_state, done, a_action, unavailable_actions_in)
                
                with tf.device("/CPU:0"):
                    #t = time.time()
                    self.buffer.update_priorities(TD_error)
                    #u_p += time.time() - t

        #print("update_priorities time: ", u_p)

        # polyak averaging
        self.target_model.set_weights(
            (1-self.polyak_update)*np.array(self.target_model.get_weights(),dtype = object) + 
                                      self.polyak_update*np.array(self.model.get_weights(),dtype = object))
        
        loss_value = loss.get('loss')#/batch
        # logs
        if summary_writer:
            with summary_writer.as_default():
                tf.summary.scalar('loss', loss_value, step=i)

        # reset all metrics
        self.model.reset_metrics()
        
        #print("Total_inner_iteration_time: ", time.time() - start, "\n")
        
        
        return loss_value
    
    def add_do_random(self, inputs):
        self.do_random = np.concatenate((self.do_random,np.array(inputs,dtype=np.int32)),axis=0,dtype=np.int32)
                
    def select_action_epsilon_greedy(self,epsilon, observations, available_actions, available_actions_bool, unavailable : bool = False):
        """ 
        selects an action using the model and an epsilon greedy policy 
        
        Parameters:
            epsilon (float):
            observations (array): (batch, 7, 7) using FourConnect (other env possible)
            available_actions (list): containing all the available actions for each batch observation
            available_actions_bool (list): containing for every index whether the action with this value is in available actions
        
        returns: 
            the chosen action for each batch element
        """

        random_action_where = np.array([np.random.randint(0,100)<epsilon*100 for _ in range(observations.shape[0])])

        # if we chose an unavailable action as the last action, and we are reminded, here add that now we choose a random action
        if np.any(self.do_random):
            random_action_where[self.do_random] = True
            #np.put(random_action_where,self.do_random,True)
            self.do_random = np.array([],dtype=np.int32)

        # if we also let the random action be unavailable sampling just takes very much longer. 
        random_actions = [np.random.choice(a) for a in available_actions]

        best_actions = self.select_action(tf.convert_to_tensor(observations, dtype=tf.float32), available_actions, available_actions_bool, unavailable).numpy()
        return np.where(random_action_where,random_actions,best_actions)

    #@tf.function
    def select_action(self, observations, available_actions, available_actions_bool, unavailable : bool = False):
        """ selects the currently best action using the model """
        probs = self.model(observations,training = False)

        # add the following print if playing against the agent to get information about it's decision
        #print("Model results: \n", probs.numpy().reshape((3,3)))
        
        # remove all unavailable actions
        if not unavailable:
            probs = tf.where(available_actions_bool,probs,tf.reduce_min(probs)-1)

        # calculate best action
        return tf.argmax(probs, axis = -1)
    
    #@tf.function
    def select_max_action_value(self, observations, available_actions_bool, unavailable : bool = False):
        """ selects the currently best action using the model """
        probs = self.target_model(observations,training = False)

        # add the following print if playing against the agent to get information about it's decision
        #print("Model results: \n", probs.numpy().reshape((3,3)))
        
        # remove all unavailable actions
        if not unavailable:
            probs = tf.where(available_actions_bool,probs,-1)

        # calculate best action
        return tf.reduce_max(probs, axis = -1)
    
    @tf.function(reduce_retracing=True)
    def calc_td_error(self, state, action, reward, new_state, done, available_action_bool, unavailable_actions_in):
        """ Calculates the TD error for prioritized experience replay 
        
        Parameters:
            state (tf.Tensor): current state
            action (tf.Tensor): action that was chosen
            reward (tf.Tensor): reward that was given
            new_state (tf.Tensor): the next state that occured    
            done (tf.Tensor): whether the game was finished after this action  
            available_action_bool (tf.Tensor): Mask with bool for which action is available    
        """

        old_Q = tf.gather(self.model(state,training=False),tf.cast(action,dtype=tf.int32),batch_dims=1)

        new_action = tf.argmax(self.model(new_state,training = False),axis = -1)
        new_action = self.select_action(new_state, None, available_action_bool, unavailable = unavailable_actions_in)
        new_Q = tf.gather(self.target_model(state,training = False), new_action, batch_dims = 1)

        # if the game is done, we cannot do another move
        # especially if we did the winning action the newly received state is in the perspective of the opponent. 
        return tf.abs(reward + tf.constant(0.99) * new_Q * (1-done) - old_Q)
    
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

        copy = DQNAgent(env, buffer = None, batch = self.batch, model_path = "")
        copy.model.set_weights(np.array(self.model.get_weights(),dtype = object))
        copy.target_model.set_weights(np.array(self.target_model.get_weights(),dtype = object))
        return copy

class RandomAgent (Agent):

    def select_action_epsilon_greedy(self,epsilon, observations, available_actions, available_actions_bool, unavailable : bool = False):
        """ 
        selects an action using the model and an epsilon greedy policy 
        but because it is a random agent, action is always random
        
        Parameters:
            epsilon (float):
            observations (array): (batch, 7, 7) using FourConnect
            available_actions (list): containing all the available actions for each batch observation
            available_actions_bool (list): containing for every index whether the action with this value is in available actions
        
        returns: 
            the chosen action for each batch element
        """
        
        #[np.random.choice(a) for a in available_actions]
        # [np.random.choice(np.arange(0,8,1)) if unavailable else np.random.choice(a) for a in available_actions] # TODO make this a variable
        return np.array([np.random.choice(a) for a in available_actions])
    
    
class MinMax_Agent (Agent):
    """ 
        selects an action using the model and an min max policy 
        
        Parameters:
            observations (array): (batch, 7, 7) using FourConnect
            available_actions (list): containing all the available actions for each batch observation
            available_actions_bool (list): containing for every index whether the action with this value is in available actions
        
        returns: 
            the chosen action for each batch element
        """
    
    #def select_action_epsilon_greedy(self,epsilon, observations, available_actions, available_actions_bool):
        #""" 
        #selects an action using the model and an epsilon greedy policy 
        
        #Parameters:
        #    epsilon (float):
        #    observations (array): (batch, 7, 7) using FourConnect
        #    available_actions (list): containing all the available actions for each batch observation
        #    available_actions_bool (list): containing for every index whether the action with this value is in available actions
        
        #returns: 
        #    the chosen action for each batch element
        #"""
        #random_action_where = [np.random.randint(0,100)<epsilon*100 for _ in range(observations.shape[0])]
        #random_actions = [np.random.choice(a) for a in available_actions]
        #best_actions = self.select_action(tf.convert_to_tensor(observations, dtype=tf.float32), available_actions, available_actions_bool).numpy()
        #return np.where(random_action_where,random_actions,best_actions)
    
    #def select_action(self, observations, available_actions, available_actions_bool):
        #""" selects the currently best action using the model """
        #raise NotImplementedError("select_action in MinMax_Agent has to be implemented to be used")