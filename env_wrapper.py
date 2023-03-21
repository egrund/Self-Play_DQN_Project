from keras_gym_env import ConnectFourEnv as GameEnv
from agent import Agent
import tensorflow as tf

class ConnectFourSelfPLay(GameEnv):

    """ A Wrapper for ConnectFourEnv of keras-gym (adapted to a newer python version)
    
    env (ConnectFourEnv)
    opponent (Agent): has to be an agent
    """

    def __init__(self,opponent : Agent = None,epsilon : float = 0):
        super(GameEnv, self).__init__()
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
        o_action = self.opponent.select_action_epsilon_greedy(self.epsilon, tf.expand_dims(s_0, axis = 0), [self.available_actions], [self.available_actions_mask])[0]
        # do the opponent's action
        s_1,_,_,_ = super().step(o_action)
        return tf.cast(s_1, dtype=tf.float32)
    
    def step(self,a): 
        # do my step
        s_0,r_0,d_0,state_id = super().step(a)
        
        if d_0:
            return tf.cast(s_0, dtype= tf.float32),r_0,d_0,state_id
            
        # get the opponent's action
        o_action = self.opponent.select_action_epsilon_greedy(self.epsilon,tf.expand_dims(tf.cast(s_0, dtype = tf.float32), axis = 0),[self.available_actions], [self.available_actions_mask])[0]
        # do the opponent's action
        s_1,r_1,d_1,state_id = super().step(o_action)
        # calculate the returns
        return tf.cast(s_1, dtype= tf.float32),r_0 - r_1,d_1,state_id
    
    def reset(self):
        return tf.cast(super().reset(), dtype= tf.float32)