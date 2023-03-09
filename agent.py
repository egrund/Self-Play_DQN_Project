from model import MyMLP
import tensorflow as tf
import numpy as np
import random as rnd

class DQNAgent:
    """ Implements a basic DQN Algorithm """

    def __init__(self, env, buffer, batch : int, model_path, reward_function = lambda d,r: r, polyak_update = 0.9, inner_iterations = 10):

        # create an initialize model and target_model
        self.model = MyMLP(hidden_units = [128,64,32], output_units = env.action_space.n)
        self.target_model = MyMLP(hidden_units = [128,64,32], output_units = env.action_space.n)
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

            # logs
            if summary_writer:
                with summary_writer.as_default():
                    tf.summary.scalar('loss', loss.get('loss'), step=j+i*self.inner_iterations)


        # polyak averaging
        self.target_model.set_weights(
            (1-self.polyak_update)*np.array(self.target_model.get_weights(),dtype = object) + 
                                      self.polyak_update*np.array(self.model.get_weights(),dtype = object))

        # reset all metrics
        self.model.reset_metrics()

        # save model
        self.model.save_weights(self.model_path + f"/{i}")
                
                
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
        best_actions = self.select_action(tf.convert_to_tensor(observations,dtype=tf.int32), available_actions_bool).numpy()
        return np.where(random_action_where,random_actions,best_actions)

    #@tf.function
    def select_action(self, observations, available_actions_bool):
        """ selects the currently best action using the model """
        probs = self.model(observations,training = False)
        # remove all unavailable actions
        probs = tf.where(available_actions_bool,probs,-1)
        # calculate best action
        return tf.argmax(probs, axis = -1)
