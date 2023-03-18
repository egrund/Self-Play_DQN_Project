import random as rnd
import numpy as np
from itertools import chain

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
        if sarsd: 
            self.empty = False

        priority_sorted = sorted(self.priorities, reverse=True)

        # first element is supposed to have max priority
        max_priority = priority_sorted[0] if priority_sorted else 1

        #extend till the capacity is reached
        for sample in sarsd:
            
            if(self.current_size < self.capacity):
                self.sarsd_list.insert(0,sample)
                self.priorities.insert(0,max_priority)
                priority_sorted.insert(0,max_priority)
                self.current_size += 1          
                # no sorting needed as max still in front
            else:
                # remove lowest data
                idx = self.priorities.index(priority_sorted[-1])
                self.sarsd_list[idx] = sample
                self.priorities[idx] = max_priority
                priority_sorted[-1] = max_priority
                # in case rather sort new that use index -2, as the two values could be the same and we could remove our new sample
                priority_sorted.sort(reverse=True)

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

    def update_priorities(self,priorities):
        """ Updates the priorities for the last given minibatch in order. """
        
        if self.empty:
            raise RuntimeError("The buffer has to be filled to update.")
        if self.last_indices == None:
            raise RuntimeError("You need to sample from the buffer before you can update priorities.")
        
        for i,p in zip(self.last_indices,priorities): 
            self.priorities[i] = p
            
            
        #
            #print zip(chain(*a),chain(*b))

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
        self.priorities = list(priorities/(np.sum(priorities)+1))

