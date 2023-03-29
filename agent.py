from model import MyCNN_RL, MyMLP_RL
import tensorflow as tf
import numpy as np
import random as rnd
import time

class Agent:

    def __init__(self):
        self.do_random = np.array([],dtype = np.int32)

    def get_do_random(self):
        return self.do_random

    def action_choice(self, probs):
        """ returns the action that makes self.game_balance in the future closest to 0 """
        # choose action that makes future game_balance closest to zero
        return tf.argmax(probs,axis=-1)
    
    def add_do_random(self, inputs):
        """ 
        gives indices of samples in batch dim to choose a random action for in the next time select action is used. 
        It is used to make sure if there is just a penalty for unavailable actions, the agent does not sample several the same samples where it does the unavailable action when epsilon is low. 
        
        Parameters:
            inputs (numpy.array): 1D, containing indices (int) max value is the next coming batch_size
        """

        self.do_random = np.concatenate((self.do_random,np.array(inputs,dtype=np.int32)),axis=0,dtype=np.int32)
    
    def select_action_epsilon_greedy(self,epsilon, observations, available_actions, available_actions_bool, unavailable : bool = False):
        """ 
        selects an action using the model and an epsilon greedy policy 
        
        Parameters:
            epsilon (float):
            observations (array): (batch, 7, 7) using FourConnect (other env possible)
            available_actions (list): containing all the available actions for each batch observation
            available_actions_bool (list): containing for every index whether the action with this value is in available actions
            unavailable (bool): whether the agent is allowed to choose unavailable actions
        
        returns: 
            the chosen action for each batch element (numpy.array)
        """

        random_action_where = np.array([np.random.randint(0,100)<epsilon*100 for _ in range(observations.shape[0])])

        # if we chose an unavailable action as the last action, and we are reminded, here add that now we choose a random action
        if np.any(self.get_do_random()):
            random_action_where[self.get_do_random()] = True
            self.do_random = np.array([],dtype=np.int32)

        # if we also let the random action be unavailable sampling just takes very much longer. 
        random_actions = [np.random.choice(a) for a in available_actions]

        best_actions = self.select_action(tf.convert_to_tensor(observations, dtype=tf.float32), available_actions, available_actions_bool, unavailable).numpy()
        return np.where(random_action_where,random_actions,best_actions)
    
    def select_action(self, observations, available_actions, available_actions_bool = None, unavailable : bool = False, opponent_level = None, save_opponent_level = True, game_balance = None):
        """ 
        selects the currently best action using the model 
        
        Parameters:
            epsilon (float):
            observations (array): (batch, 7, 7) using FourConnect (other env possible)
            available_actions (list): containing all the available actions for each batch observation
            available_actions_bool (list): containing for every index whether the action with this value is in available actions
            unavailable (bool): whether the agent is allowed to choose unavailable actions
        
        returns: 
            the chosen best action for each batch element (tf.Tensor)
        """
        raise NotImplementedError("select action is not implemented in Agent")

# TODO best agent only other action decision formular
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
        
        super().__init__()
        
        # build models
        obs = tf.expand_dims(env.reset(),axis=0)
        env.close()
        self.model(obs)
        self.target_model(obs)
        self.target_model.set_weights(np.array(self.model.get_weights(),dtype = object))

        # save other variables as attributes
        self.inner_iterations = inner_iterations
        self.polyak_update = polyak_update
        self.model_path = model_path
        self.buffer = buffer
        self.batch = batch 
        self.reward_function = reward_function
        self.prioritized_experience_replay = prioritized_experience_replay
        self.conv_kernel = conv_kernel
        self.filters = filters
        self.hidden_units = hidden_units
        self.output_activation = output_activation
        self.gamma = gamma
      
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

    #@tf.function
    def select_action(self, observations, available_actions, available_actions_bool = None, unavailable : bool = False):
        """ 
        selects the currently best action using the model 
        
        Parameters:
            epsilon (float):
            observations (array): (batch, 7, 7) using FourConnect (other env possible)
            available_actions (list): containing all the available actions for each batch observation
            available_actions_bool (list): containing for every index whether the action with this value is in available actions
            unavailable (bool): whether the agent is allowed to choose unavailable actions
        
        returns: 
            the chosen best action for each batch element (tf.Tensor)
        """
        probs = self.model(observations,training = False)

        # add the following print if playing against the agent to get information about it's decision
        #print("Model results: \n", probs.numpy().reshape((3,3)))
        
        # remove all unavailable actions
        if not unavailable and available_actions_bool != None:
            probs = tf.where(available_actions_bool,probs,tf.reduce_min(probs)-1)

        # calculate best action
        # return tf.argmax(probs, axis = -1)
        return self.action_choice(probs)
    
    #@tf.function
    def select_best_action_value(self, observations, available_actions_bool, unavailable : bool = False):
        """ 
        selects the best next action Qsa given observations 
        
        Parameters:
            epsilon (float):
            observations (array): (batch, 7, 7) using FourConnect (other env possible)
            available_actions (list): containing all the available actions for each batch observation
            available_actions_bool (list): containing for every index whether the action with this value is in available actions
            unavailable (bool): whether the agent is allowed to choose unavailable actions
        
        returns: 
            max(Qsa) (tf.Tensor)
        """
        probs = self.target_model(observations,training = False)

        # add the following print if playing against the agent to get information about it's decision
        #print("Model results: \n", probs.numpy().reshape((3,3)))
        
        # remove all unavailable actions
        if not unavailable:
            probs = tf.where(available_actions_bool,probs,-1)

        # calculate best action
        return probs[self.action_choice(probs)]
    
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
            unavailalbe_actions_in (bool): Whether unavailable actions should be used when calculating the current Q value new_Q
        """

        old_Q = tf.gather(self.model(state,training=False),tf.cast(action,dtype=tf.int32),batch_dims=1)

        new_action = self.select_action(new_state, None, available_action_bool, unavailable = unavailable_actions_in)
        new_Q = tf.gather(self.target_model(state,training = False), new_action, batch_dims = 1)

        # if the game is done, we cannot do another move
        # especially if we did the winning action the newly received state is in the perspective of the opponent. 
        return tf.abs(reward + self.gamma * new_Q * (1-done) - old_Q)
    
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

        copy = DQNAgent(env,
                        buffer = None, 
                        batch = self.batch, 
                        model_path = "", 
                        conv_kernel = self.conv_kernel,
                        filters = self.filters,
                        hidden_units = self.hidden_units,
                        output_activation =self.output_activation)
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
            unavailable (bool): whether the agent is allowed to choose unavailable actions
        
        returns: 
            the chosen action for each batch element
        """
        
        return np.array([np.random.choice(a) for a in available_actions])
    
class AdaptingAgent(Agent):

    def __init__(self,best_agent : DQNAgent,game_balance_max : int = 500):

        super().__init__()

        self.best_agent = best_agent
        self.best_agent.model.trainable = False
        self.best_agent.target_model.trainable = False

        self.model = self.best_agent.model
        self.target_model = self.best_agent.target_model

        self.game_balance = np.array([],dtype = np.float32)
        self.max_balance_length = tf.constant(game_balance_max, dtype=tf.float32)
        self.opponent_level = tf.constant(-1.)

    def add_game_balance_information(self,inputs):
        """ 
        Adds information about the last rewards to the agent, so it can calculate the opponents level. (Should only use final state rewards, not in game)
        
        Parameters:
            inputs (numpy.array / list): 1D containing the last reward values
        """

        # add new samples to the beginning and remove too much at the end
        self.game_balance = np.concatenate((np.array(inputs),self.game_balance), axis=-1, dtype=np.float32)
        self.game_balance = self.game_balance[0:int(self.max_balance_length.numpy())]

    def get_game_balance(self, tensor = False):
        # if we do not have any opponent level information return 0
        if not self.game_balance.any():
            if tensor:
                return tf.constant(0.0)
            return 0.0
        if tensor:
            return tf.reduce_mean(tf.convert_to_tensor(self.game_balance))
        return np.mean(self.game_balance)
    
    def reset_game_balance(self):
        self.game_balance = np.array([],dtype = np.double)

    def reset_opponent_level(self):
        self.opponent_level = tf.constant(-1.)

    #@tf.function(reduce_retracing=True)
    def action_choice(self, probs):
        """ returns the action that makes self.game_balance in the future closest to 0 """
        # choose action that makes future game_balance closest to zero
        return tf.argmin(tf.math.abs(probs),axis=-1)
    
    #@tf.function(reduce_retracing=True)
    def select_action(self, observations, available_actions, available_actions_bool = None, unavailable : bool = False, opponent_level = None, save_opponent_level = True, game_balance = None):
        """ 
        selects the currently best action using the model 
        
        Parameters:
            epsilon (float):
            observations (array): (batch, 7, 7) using FourConnect (other env possible)
            available_actions (list): containing all the available actions for each batch observation
            available_actions_bool (list): containing for every index whether the action with this value is in available actions
            unavailable (bool): whether the agent is allowed to choose unavailable actions
        
        returns: 
            the chosen best action for each batch element (tf.Tensor)
        """
        
        probs = self.model(observations, training = False)

        # add the following print if playing against the agent to get information about it's decision
        #print("Model results: \n", probs.numpy().reshape((3,3)))
        
        # remove all unavailable actions
        if not unavailable and available_actions_bool != None:
            probs = tf.where(available_actions_bool, probs,-tf.reduce_max(tf.abs(probs))*200 +1)

        # calculate best action
        return self.action_choice(probs)

class AdaptingDQNAgent(AdaptingAgent):
    """ Implements a basic DQN Algorithm 
    
    Attributes:
        best_agent (DQNAgent): a fully initialized best agent
        buffer (Buffer): the buffer to save the with this agent sampled data in
    """

    def __init__(self, best_agent : DQNAgent, env, buffer, batch : int , model_path, polyak_update = 0.9, inner_iterations = 10, 
                 reward_function = lambda d,r:r,#lambda d,r: tf.where(r==0.0,tf.constant(1.0),tf.where(r==1.0,tf.constant(0.0), r)), 
                 hidden_units = [64], prioritized_experience_replay : bool = True, gamma : tf.constant = tf.constant(0.99), 
                 loss_function = tf.keras.losses.MeanSquaredError(), output_activation = None, game_balance_max : int = 500):
        
        super().__init__(best_agent,game_balance_max)

        # create an initialize model and target_model
        self.model = MyMLP_RL(hidden_units =hidden_units, output_units = env.action_space.n, output_activation = output_activation, 
                 loss = loss_function, gamma = gamma)
        self.target_model = MyMLP_RL(hidden_units =hidden_units, output_units = env.action_space.n, output_activation = output_activation, 
                 loss = loss_function, gamma = gamma)
        
        # build models
        obs = tf.expand_dims(env.reset(),axis=0)
        self.model(obs, agent = self)
        self.target_model(obs, agent = self)
        self.target_model.set_weights(np.array(self.model.get_weights(),dtype = object))

        # save other variables as attributes
        self.inner_iterations = inner_iterations
        self.polyak_update = polyak_update
        self.model_path = model_path
        self.buffer = buffer
        self.batch = batch 
        self.reward_function = reward_function
        self.prioritized_experience_replay = prioritized_experience_replay
        self.hidden_units = hidden_units
        self.output_activation = output_activation
        self.gamma = gamma
      
    def train_inner_iteration(self, summary_writer, i, unavailable_actions_in):
        """ """
        u_p = 0
        
        for j in range(self.inner_iterations):

            # sample random minibatch of transitions
            minibatch = self.buffer.sample_minibatch(self.batch)  #Out:[state, action, reward, next_state, done, available_actions, game_balance]
            
            state = tf.convert_to_tensor([sample[0] for sample in minibatch],dtype = tf.float32)
            actions = tf.convert_to_tensor([sample[1] for sample in minibatch],dtype = tf.float32)
            
            new_state = tf.convert_to_tensor([sample[3] for sample in minibatch],dtype = tf.float32)
            done = tf.convert_to_tensor([sample[4] for sample in minibatch],dtype = tf.float32)
            reward = self.reward_function(tf.cast(done, dtype = tf.bool), tf.convert_to_tensor([sample[2] for sample in minibatch],dtype = tf.float32))
            a_action = [sample[5] for sample in minibatch]
            game_balance = tf.expand_dims(tf.convert_to_tensor([sample[6] for sample in minibatch], dtype = tf.float32),axis=-1)
            opponent_level = tf.expand_dims(tf.convert_to_tensor([sample[7] for sample in minibatch], dtype = tf.float32),axis=-1)

            loss = self.model.train_step((state, actions, reward, new_state, done, a_action, game_balance, opponent_level), self, summary_writer, i*j)

            
            # if prioritized experience replay, then here
            if self.prioritized_experience_replay:
                TD_error = self.calc_td_error(state, actions, reward, new_state, done, a_action, unavailable_actions_in, game_balance)
                
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

    #@tf.function(reduce_retracing=True)
    def select_action(self, observations, available_actions, available_actions_bool = None, unavailable : bool = False, opponent_level = None, save_opponent_level = True, game_balance = None):
        """ 
        selects the currently best action using the model 
        
        Parameters:
            epsilon (float):
            observations (array): (batch, 7, 7) using FourConnect (other env possible)
            available_actions (list): containing all the available actions for each batch observation
            available_actions_bool (list): containing for every index whether the action with this value is in available actions
            unavailable (bool): whether the agent is allowed to choose unavailable actions
        
        returns: 
            the chosen best action for each batch element (tf.Tensor)
        """
        if opponent_level == None:
            opponent_level = tf.expand_dims(tf.repeat(self.opponent_level,observations.shape[0]),axis=-1)
        if game_balance == None:
            game_balance = tf.expand_dims(tf.repeat(self.get_game_balance(tensor = True),observations.shape[0]),axis=-1)
        
        probs, new_opponent_level = self.model(observations, training = False, agent = self, opponent_level = opponent_level, game_balance = game_balance)

        if save_opponent_level:
            self.opponent_level = tf.reduce_mean(new_opponent_level)

        # add the following print if playing against the agent to get information about it's decision
        #print("Model results: \n", probs.numpy().reshape((3,3)))
        
        # remove all unavailable actions
        if not unavailable and available_actions_bool != None:
            probs = tf.where(available_actions_bool, probs,tf.reduce_max(tf.abs(probs)) +1)

        # calculate best action
        return self.action_choice(probs)
    
    #@tf.function(reduce_retracing=True)
    def select_adapting_action_value(self, observations, available_actions_bool, opponent_level, game_balance, unavailable : tf.constant = tf.constant(False)):
        """ 
        selects the best next action Qsa given observations 
        
        Parameters:
            epsilon (float):
            observations (array): (batch, 7, 7) using FourConnect (other env possible)
            available_actions (list): containing all the available actions for each batch observation
            available_actions_bool (list): containing for every index whether the action with this value is in available actions
            unavailable (bool): whether the agent is allowed to choose unavailable actions
        
        returns: 
            max(Qsa) (tf.Tensor)
        """
        #if opponent_level == None: # this should not happen
            #opponent_level = tf.expand_dims(tf.repeat(self.opponent_level,observations.shape[0]),axis=-1)
        probs, _ = self.target_model(observations, training = False, agent = self, opponent_level = opponent_level, game_balance = game_balance)

        # add the following print if playing against the agent to get information about it's decision
        #print("Model results: \n", probs.numpy().reshgame_balance
        # remove all unavailable actions
        if not unavailable:
            probs = tf.where(available_actions_bool, probs,tf.reduce_max(tf.abs(probs)) +1)

        # calculate best action
        return tf.gather(probs,self.action_choice(probs),batch_dims = 1)
    
    #@tf.function(reduce_retracing=True)
    def calc_td_error(self, state, action, reward, new_state, done, available_action_bool, unavailable_actions_in, opponent_level = None, game_balance = None):
        """ Calculates the TD error for using prioritized experience replay. 
        
        Parameters:
            state (tf.Tensor): the current state
            action (tf.Tensor): the action that was chosen
            reward (tf.Tensor): the reward that was given
            new_state (tf.Tensor): the next state that occured after the action was taken 
            done (tf.Tensor): whether the game was finished after this action  
            available_action_bool (tf.Tensor): mask with bool for which action is available    
            unavailalbe_actions_in (bool): whether unavailable actions should be used when calculating the current Q value new_Q
        """
        if opponent_level == None:
            opponent_level = tf.expand_dims(tf.repeat(self.opponent_level,state.shape[0]),axis=-1)
        if game_balance == None:
            game_balance = tf.expand_dims(tf.repeat(self.get_game_balance(tensor = True),state.shape[0]),axis=-1)

        old_Q = tf.gather(self.model(state,training=False,agent = self, opponent_level = opponent_level, game_balance = game_balance)[0],tf.cast(action,dtype=tf.int32),batch_dims=1)

        new_action = self.select_action(new_state, None, available_action_bool, unavailable = unavailable_actions_in, opponent_level = opponent_level, save_opponent_level = False, game_balance = game_balance)
        new_Q = tf.gather(self.target_model(state, training = False, agent = self, opponent_level = opponent_level, game_balance = game_balance)[0], new_action, batch_dims = 1)

        # if the game is done, we cannot do another move
        # especially if we did the winning action the newly received state is in the perspective of the opponent, as there is no next state for us, because the opponent does not do a move. 
        new_game_balance = tf.add( game_balance, tf.divide(reward, self.max_balance_length))
        return tf.abs(new_game_balance + self.gamma * new_Q * (1-done) - old_Q)
    
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

        copy = AdaptingDQNAgent(self.best_agent, buffer = None, model_path = "",
                 reward_function = lambda d,r: tf.where(r==-0.1, tf.constant(0.1), tf.where(r==0.0,tf.constant(1.0),tf.where(r==1.0,tf.constant(-1.0), r))), 
                 hidden_units = self.hidden_units, gamma = self.gamma, output_activation = self.output_activation)
        copy.model.set_weights(np.array(self.model.get_weights(),dtype = object))
        copy.target_model.set_weights(np.array(self.target_model.get_weights(),dtype = object))
        return copy
    
class AdaptingAgent2(AdaptingAgent): # same as AdaptingAgent at the moment
    """ do not train this agent, only uses some functionality of DQN agent """

    def __init__(self, best_agent : DQNAgent, calculation_value : tf.constant = tf.constant(0.01), game_balance_max : int = 500):  
        super().__init__(best_agent,game_balance_max)
        self.calculation_value = calculation_value

    #@tf.function(reduce_retracing=True)
    def action_choice(self, probs):
        """ returns the action that makes self.game_balance in the future closest to 0 """
        # choose action that makes future game_balance closest to zero
        # scale positive values a little smaller so the propertion of loosing and winning is more balanced
        helping = tf.where(probs>0,probs*self.calculation_value,probs)
        return tf.argmin(tf.math.abs(helping),axis=-1)
    
class AdaptingAgent3(AdaptingAgent): # same as AdaptingAgent at the moment
    """ do not train this agent, only uses some functionality of DQN agent """

    def __init__(self, best_agent : DQNAgent, calculation_value : tf.constant = tf.constant(0.3), game_balance_max : int = 500):
        super().__init__(best_agent,game_balance_max)
        self.calculation_value = calculation_value

    #@tf.function(reduce_retracing=True)
    def action_choice(self, probs):
        """ returns the action that makes self.game_balance in the future closest to 0 """
        # choose action that makes future game_balance closest to zero
        # scale the positive action values between 0 and 1, then choose a certain percentage point

        scaled_around_value = tf.subtract(tf.divide(probs,tf.reduce_max(probs)), self.calculation_value)
        adapting_action =  tf.argmin(tf.math.abs(scaled_around_value),axis=-1)

        # return best action when we are loosing
        boolean_value = tf.where(tf.reduce_max(probs,axis=-1)<tf.constant(0.0),True,False)
        best_action = tf.argmax(probs,axis=-1)

        return tf.where(boolean_value, best_action, adapting_action)

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