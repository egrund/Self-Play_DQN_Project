from keras_gym_env import ConnectFourEnv
from agent import Agent

class ConnectFourSelfPLay(ConnectFourEnv):

    """ A Wrapper for ConnectFourEnv of keras-gym (adapted to a newer python version)
    
    env (ConnectFourEnv)
    opponent (Agent): has to be an agent
    """

    def __init__(self,opponent : Agent):
        super(ConnectFourEnv, self).__init__()
        self.opponent = opponent

    def set_opponent(self, opponent : Agent):
        self.opponent = opponent
    
    def opponent_starts(self):
        """ let the opponent do the first action, works similar to reset"""
        s_0 = self.reset()
        # get the opponent's action
        o_action = self.opponent.select_action(s_0, self.available_actions(), self.available_actions_mask())
        # do the opponent's action
        s_1,_,_,_ = super.step(o_action)
        return s_1
    
    def step(self,a): 
        # do my step
        s_0,r_0,d_0,_ = super.step(a)
        # get the opponent's action
        o_action = self.opponent.select_action(s_0, self.available_actions(), self.available_actions_mask())
        # do the opponent's action
        s_1,r_1,d_1,state_id = self.env.step(o_action)
        # calculate the returns
        return s_1,r_0-r_1,d_0 or d_1,state_id