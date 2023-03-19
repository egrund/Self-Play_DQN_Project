import random as rnd
import numpy as np
import time
class Buffer:
    """ 
    Implemens a replay buffer for our DQNAgent class. Can use prioritized experience replay. 
    """

    def __init__(self, capacity, min_size):
        self.capacity = capacity
        self.min_size = min_size
        self.current_size = 0
        self.sarsd_list = []
        self.priorities = np.array([])
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
        p = False
        a_time = 0
        b_time = 0
        c_time = 0
        d_time = 0
        e_time = 0
        
        if sarsd: 
            self.empty = False
 
        # first element is supposed to have max priority
        if(np.any(self.priorities)):
            max_priority = np.amax(self.priorities)
        else:
            max_priority = 1
            
        #extend till the capacity is reached
        for sample in sarsd:

            if(self.current_size < self.capacity):
                self.sarsd_list.insert(0,sample)
                self.priorities = np.insert(self.priorities,0,max_priority)
                self.current_size += 1          
                # no sorting needed as max still in front
            else:
                p = True
                # remove lowest data
                t = time.time()
                idx = np.argmin(self.priorities)               
                a_time += time.time() - t
                t = time.time()
                self.sarsd_list[idx] = sample
                b_time += time.time() - t
                t = time.time()
                self.priorities[idx] = max_priority
                c_time += time.time() - t                

                
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

