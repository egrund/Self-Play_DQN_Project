# Choose the env at the top
#**************************

from agent import Agent
import tensorflow as tf
from gym import Env

class SelfPLayWrapper(Env):

    """ A Wrapper for ConnectFourEnv of keras-gym (adapted to a newer python version)
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

    def set_opponent(self, opponent : Agent):
        self.opponent = opponent
        
    def set_epsilon(self, epsilon : float):
        self.epsilon = epsilon
        
    def opponent_starts(self):
        """ let the opponent do the first action, works similar to reset"""
        s_0 = self.reset()
        # get the opponent's action
        o_action = self.opponent.select_action_epsilon_greedy(self.epsilon, tf.expand_dims(tf.cast(s_0, dtype = tf.float32), axis = 0), [self.env.available_actions], [self.env.available_actions_mask])[0]
        # do the opponent's action
        s_1,_,_ = self.env.step(o_action)
        return tf.cast(s_1, dtype=tf.float32)
    
    def step(self,a): 
        # do my step
        s_0,r_0,d_0 = self.env.step(a)
        
        if d_0:
            return tf.cast(s_0, dtype= tf.float32),r_0,d_0
            
        # get the opponent's action
        o_action = self.opponent.select_action_epsilon_greedy(self.epsilon,tf.expand_dims(tf.cast(s_0, dtype = tf.float32), axis = 0),[self.env.available_actions], [self.env.available_actions_mask])[0]
        # do the opponent's action
        s_1,r_1,d_1 = self.env.step(o_action)
        # calculate the returns
        return tf.cast(s_1, dtype= tf.float32),r_0 - r_1,d_1
    
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
    