from keras_gym_env import ConnectFourEnv
import numpy as np
import tensorflow as tf

class Sampler:
    """ 
    Implements an algorithm to sample from a self-play environment using two different agents

    Attributes: 
        envs (list): List of all the ConnectFour environments to sample from
        batch (int): how many environments to sample from at the same time
        agents (list): list of two agents to use for the sampling procedure
    """

    def __init__(self,batch,agents):

        self.envs = [ConnectFourEnv() for _ in range(batch)]
        self.batch = batch
        self.agents = agents
  
    def sample_from_game(self,epsilon, save = True):
        """ 
        samples from every environment in self.envs until it is done or reaches 10000 steps. 

        Parameters:
            epsilon (float): epsilon for the epsilon greedy policy
        """

        sarsd = [[],[]]
        agent_turn = np.random.randint(0,2,(self.batch,))
        current_envs = [np.extract(agent_turn == i,self.envs) for i in range(2)]
        shapes = [current_envs[i].shape[0] for i in range(2)]

        observations = [np.array([env.reset() for env in which_agent ]) for which_agent in current_envs]

        for e in range(10000):
            #print("E: ", e)
            #print("Shapes: ", shapes)

            # get all the actions from the agents
            #print("first observation ", observations)
            available_actions = [[current_envs[i][j].available_actions for j in range(shapes[i])] for i in range(2)]
            available_actions_bool = [[current_envs[i][j].available_actions_mask for j in range(shapes[i])] for i in range(2)]

            actions = [self.agents[i].select_action_epsilon_greedy(epsilon, observations[i],available_actions[i], available_actions_bool[i])if shapes[i] != 0 else [] for i in range(2)]

            # do the step in every env
            results = [[current_envs[i][j].step(actions[i][j]) for j in range(shapes[i])] if shapes[i] != 0 else [] for i in range(2)]

            # add first observation to new results
            results = [[(observations[i][j],actions[i][j],results[i][j][1],results[i][j][0],results[i][j][2]) for j in range(shapes[i])] if shapes[i] != 0 else [] for i in range(2)] # state, action, reward, new state, done

            [sarsd[i].extend(results[i]) for i in range(2)] 

            # get observations for next turn, and remove everything that is done already
            observations = [np.array([results[i][j][3] for j in range(shapes[i]) if not results[i][j][4]]) for i in range(2)]
            current_envs = [np.array([current_envs[i][j] for j in range(shapes[i]) if not results[i][j][4]]) for i in range(2)]

            # check if all envs are done
            if all([observations[i].shape == (0,) for i in range(2)]):
                break

            # let them switch sides for the next turn
            current_envs.reverse()
            observations.reverse()
            shapes = [current_envs[i].shape[0] for i in range(2)]

        # save data in buffer
        if save:
            [self.agents[i].buffer.extend(sarsd[i]) for i in range(2)]

        # render for debugging
        # [e.render() for e in self.envs]

        # return averade reward for both agents
        #print(tuple([np.mean([sarsd[i][j][2] for j in range(len(sarsd[i]))], dtype=object) for i in range(2)]))
        return tuple([np.mean([sarsd[i][j][2] for j in range(len(sarsd[i]))], dtype=object) for i in range(2)])

    def fill_buffers(self,epsilon):
        """ 
        fills the empty buffer with min elements sampling several times from the environemnts
        
        Parameters: 
            epsilon (float): epsilon for the epsilon greedy policy
        """
        
        while(any([self.agents[i].buffer.current_size < self.agents[i].buffer.min_size for i in range(2)])):
            _ = self.sample_from_game(epsilon)
