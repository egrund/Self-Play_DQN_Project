import random as rnd
import numpy as np

class Buffer:
    """ 
    Implemens a replay buffer for our DQNAgent class. Can use prioritized experience replay. 
    """

    def __init__(self, capacity, min_size):
        self.capacity = capacity
        self.min_size = min_size
        self.current_size = 0
        self.sarsd_list = []
        self.priorities = []
        self.last_indices = None
        self.empty = True # is False after adding data the first time

    def extend(self, sarsd):
        """ adds new data to memory buffer """
        """
        Prameters:
        ______________
        sarsd ( nested list): list of new data samples to add to the buffer. each sample being in the order 
            state, action, reward, new_state, done given as a list itself
        ______________
        
        
        Returns:
        ______________
        none
        ______________
        """

        # if there is data to add, buffer is not empty anymore
        if self.sarsd: 
            self.empty = False

        self.sort()

        # first element is supposed to have max priority
        max_priority = self.priorities[0] if self.priorities else 1

        #extend till the capacity is reached
        for sample in sarsd:

            # put priority at the beginning, add to front of list
            self.sarsd_list.insert(0,sample)
            self.priorities.insert(0,max_priority)
            
            if(self.current_size < self.capacity):
                self.current_size += 1          
            else:
                # remove old data at the end. 
                self.sarsd_list.pop(-1)
                self.priorities.pop(-1)

    def sample_minibatch(self, batch_size):
        """ samples a minibatch from the buffer """
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

        if self.empty:
            raise RuntimeError("The buffer has to be filled to sample.")

        indices = self.get_minibatch_indices(batch_size)
        self.last_indices = indices # save so we can later change the priorities
        output = []

        for i in indices:
            output.append(self.sarsd_list[i])
        
        return output
    
    def sort(self):
        if self.empty:
            raise RuntimeError("The buffer has to be filled to sort.")
        self.sarsd_list.sort(key = lambda x: self.priorities[self.sarsd_list.index(x)])

    def update_priorities(self,priorities):
        """ Updates the priorities for the last given minibatch in order. """

        if self.empty:
            raise RuntimeError("The buffer has to be filled to update.")
        if self.last_indices == None:
            raise RuntimeError("You need to sample from the buffer before you can update priorities.")

        for i,p in zip(self.last_indices,priorities): 
            self.priorities[i] = p

    def get_minibatch_indices(self,batch_size):
        """ returns random indices by priorities as weights for the next minibatch"""
        if self.empty:
            raise RuntimeError("The buffer has to be filled to sample.")
        self.normalize_priorities()
        return rnd.choices([i for i in range(0,self.current_size)],weights = self.priorities,k=batch_size)

    def normalize_priorities(self):
        """ normalize the priorities so they can be given as weights for random.choice """
        if self.empty:
            raise RuntimeError("The buffer has to be filled to normalize priorities.")
        
        priorities = np.power(np.array(self.priorities),0.3)
        self.priorities = priorities/(np.sum(priorities)+1)

