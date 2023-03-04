import random as rnd

class Buffer:
    """ 
    Implemens a replay buffer for our DQNAgent class. 
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.sarsd_list = []
        self.idx = 0

    def extend(self, sarsd):
        """ adds new data to memory buffer """
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
        #extend till the capacity is reached
        if(len(sarsd_list) < capacity):
            self.sarsd_list.extend(sarsd)
            
        else:
            self.idx = rnd.randint(0, capacity - 1) #random overwrite
            self.sarsd_list[self.idx][0] = sarsd[0] #state
            self.sarsd_list[self.idx][1] = sarsd[1] #action
            self.sarsd_list[self.idx][2] = sarsd[2] #reward
            self.sarsd_list[self.idx][3] = sarsd[3] #next_state
            self.sarsd_list[self.idx][4] = sarsd[4] #done
        
        

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
        sample = rnd.sample(self.sarsd_list, batch_size)
        
        return sample
