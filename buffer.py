import random as rnd

class Buffer:
    """ 
    Implemens a replay buffer for our DQNAgent class. 
    """

    def __init__(self, capacity, min_size):
        self.capacity = capacity
        self.min_size = min_size
        self.current_size = 0
        self.sarsd_list = []
        self.idx = 0

    def extend(self, sarsd):
        """ adds new data to memory buffer """
        """
        Prameters:
        ______________
        sarsd (list): list of new data samples to add to the buffer. each sample being in the order 
            state, action, reward, new_state, done
        ______________
        
        
        Returns:
        ______________
        none
        ______________
        """
        #extend till the capacity is reached
        for sample in sarsd:
            if(self.current_size < self.capacity):
                self.sarsd_list.append(sample)
                self.current_size += 1
                
            else:
                #randomly overwrite old data
                self.idx = rnd.randint(0, self.capacity - 1)
                self.sarsd_list[self.idx] = sample

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
