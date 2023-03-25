from agent import Agent
import tensorflow as tf
from gym import Env
import numpy as np

class SelfPLayWrapper(Env):

    """ A Wrapper for Env similar to keras-gym Connect Four (adapted to a newer python version)
    also works for similar other envs
    Important: loss_reward of env is not used by use - win_reward directly here
    
    Attributes: 
        env (gym.Env): the gym env to wrap arount, has to be a 2 player game
        opponent (Agent): has to be an agent
        epsilon (float): the epsilon value for the epsilon greedy policy
    """

    def __init__(self,env_class, opponent : Agent = None,epsilon : float = 0):
        super(Env, self).__init__()
        self.env = env_class()
        self.opponent = opponent
        self.epsilon = epsilon
        self.first_reward = None
        self.last_wrong = None

    def set_opponent(self, opponent : Agent):
        self.opponent = opponent
        
    def set_epsilon(self, epsilon : float):
        self.epsilon = epsilon
        
    def opponent_starts(self):
        """ let the opponent do the first action, works similar to reset"""
        s_0 = self.reset()
        # get the opponent's action
        o_action = self.opponent.select_action_epsilon_greedy(self.epsilon, tf.expand_dims(tf.cast(s_0, dtype = tf.float32), axis = 0), [self.env.available_actions], [self.env.available_actions_mask], False)[0]
        # do the opponent's action
        s_1,_,_ = self.env.step(o_action, return_wrong = False)

        return tf.cast(s_1, dtype=tf.float32)
    
    def step(self,a, unavailable_in : bool = False, agent = None): 
        # do my step
        if unavailable_in:
            s_0,r_0,d_0, w_0 = self.env.step(a,return_wrong =  True)
        else:
            s_0,r_0,d_0 = self.env.step(a,return_wrong =  False)
            w_0 = False
        
        if d_0 or w_0:
            return tf.cast(s_0, dtype= tf.float32),r_0,d_0
            
        # get the opponent's action
        o_action = self.opponent.select_action_epsilon_greedy(self.epsilon,tf.expand_dims(tf.cast(s_0, dtype = tf.float32), axis = 0),[self.env.available_actions], [self.env.available_actions_mask], False)[0]
        # do the opponent's action
        s_1,r_1,d_1 = self.env.step(o_action,return_wrong =  False)
        # calculate the returns
        if d_1:
            return tf.cast(s_1, dtype= tf.float32),- r_1,d_1
        
        return tf.cast(s_1, dtype= tf.float32),r_0,d_1
    
    def step_player(self,a, unavailable_in : bool = False):
        """ 
        does a step of the player, always has to be followed by a step of the opponent, which has to choose an action in between 
        used when playing in batches in sampler 
        """
        # do my step
        if unavailable_in:
            s_0,r_0,d_0, w_0 = self.env.step(a,return_wrong =  True)
        else:
            s_0,r_0,d_0 = self.env.step(a,return_wrong =  False)
            w_0 = False
        self.first_reward = r_0
        self.last_wrong = w_0

        if d_0 or w_0: # if done give empty state for opponent to calculate action, stop in part two of step
            return tf.zeros_like(s_0, dtype=tf.float32)
        
        # get opponents action by returning the input for it to step opponent
        return tf.cast(s_0, dtype = tf.float32)
    
    def step_opponent(self,o_action):
        """ 
        one step of the opponent, checking whether the env is done before 
        used when playing in batches in sampler 
        """

        if self.env.done or self.last_wrong:
            return tf.cast(self.env.state, dtype = tf.float32), self.first_reward, self.env.done

        # do the opponent's action
        s_1,r_1,d_1 = self.env.step(o_action,return_wrong =  False)
        # calculate the returns
        if d_1:
            return tf.cast(s_1, dtype= tf.float32), - r_1,d_1

        return tf.cast(s_1, dtype= tf.float32), self.first_reward,d_1
    
    def reset(self):
        return tf.cast(self.env.reset(), dtype= tf.float32)
    
    # all public things from env to access from the outside
    def render(self):
        self.env.render()

    @property
    def state(self):
        return self.env.state
    
    @property
    def available_actions(self):
        # the if statement is needed to use step_agent and step_opponent for more efficient sampling in batches. 
        if self.env.done:
            return np.array([-1])
        return self.env.available_actions
    
    @property
    def available_actions_mask(self):
        return self.env.available_actions_mask
    
    @property
    def num_rows(self):
        return self.env.num_rows
    
    @property
    def num_cols(self):
        return self.env.num_cols
    
    @property
    def num_players(self):
        return self.env.num_players
    
    @property
    def win_reward(self):
        return self.env.win_reward
    
    @property
    def loss_reward(self):
        return - self.win_reward
    
    @property
    def draw_reward(self):
        return self.env.draw_reward
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def max_time_steps(self):
        return self.env.max_time_steps 
    
    @property
    def filters(self):
        return self.env.filters
    