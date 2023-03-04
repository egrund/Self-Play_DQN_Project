import random as rnd

class Buffer:
    """ 
    Implemens a replay buffer for our DQNAgent class. 
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.s = None
        self.a = None
        self.r = None
        self.s_n = None
        self.done = None
        self.sarsd_list = [[self.s, self.a, self.r, self.s_n, self.done] for i in range(capacity)]
        self.idx = 0

    def extend(self, state, action, reward, new_state, done):
        """ adds new data to the circular memory buffer """
        """
        Prameters:
        ______________
        state(array): current state
        action(int): which action will be performed
        reward(float): reward is calculated according to the win conditon and agent
        new_state(array): the next state
        done(bool): True if game is over 
        ______________
        
        
        Returns:
        ______________
        none
        ______________
        """
        #sarsd_list[idx][state, action, reward, next_state, done]]
        self.sarsd_list[self.idx][0] = state
        self.sarsd_list[self.idx][1] = action
        self.sarsd_list[self.idx][2] = reward
        self.sarsd_list[self.idx][3] = next_state     
        self.sarsd_list[self.idx][4] = done
        
        #for circular memory
        self.idx = (self.idx+1)%self.capacity

    def sample_minibatch(self, batch_size):
        """ samples a random minibatch from the buffer """
        """
        Prameters:
        ______________
        batch_size(int) = The sample size pulled from sarsd_list
        ______________
        
        
        Returns:
        ______________
        sample(list) = A sample pulled from sarsd_list containing [state, action, reward, next_state, done]
        ______________
        """
        sample = rnd.sample(sarsd_list, batch_size)
        
        return sample
