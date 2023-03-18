from env_wrapper import ConnectFourSelfPLay
from agent import Agent
import numpy as np
import time

class Sampler:
    """ 
    Implements an algorithm to sample from a self-play environment using two different agents

    Attributes: 
        envs (list): List of all the ConnectFour environments to sample from
        batch (int): how many environments to sample from at the same time
        agents (list): list of two agents to use for the sampling procedure
    """

    def __init__(self,batch,agent, opponent : Agent ,opponent_epsilon : float = 0):

        self.envs = [ConnectFourSelfPLay(opponent,opponent_epsilon) for _ in range(batch)]
        self.batch = batch
        self.agent = agent

    def set_opponent(self, opponent):
        [env.set_opponent(opponent) for env in self.envs]

    def set_opponent_epsilon(self,epsilon):
        [env.set_epsilon(epsilon) for env in self.envs]

    def sample_from_game_wrapper(self,epsilon, save = True):
        """ samples from env wrappers"""

        sarsd = []
        current_envs = self.envs
        agent_turn = np.random.randint(0,2,(self.batch,))
        observations = np.array([env.opponent_starts() if whether else env.reset() for whether,env in zip(agent_turn,current_envs)])
        
        start = time.time()
        for e in range(10000):
            available_actions = [env.available_actions for env in current_envs]
            available_actions_bool = [env.available_actions_mask for env in current_envs]

            actions = self.agent.select_action_epsilon_greedy(epsilon, observations,available_actions, available_actions_bool)

            results = [env.step(actions[i]) for i,env in enumerate(current_envs)] # new state, reward, done, info

            # bring everything in the right order
            results = [[observations[i],actions[i],results[i][1],results[i][0],results[i][2]] for i in range(len(current_envs))] # state, action, reward, new state, done

            sarsd.extend(results)

            observations = np.array([results[i][3] for i in range(len(current_envs)) if not results[i][4]])
            current_envs = np.array([current_envs[i] for i in range(len(current_envs)) if not results[i][4]])

            # check if all envs are done
            if observations.shape == (0,):
                
                print(e," loops:", time.time()-start ,"\n","average time per loop: ", (time.time()-start)/e)
                break

        # save data in buffer
        if save:
            self.agent.buffer.extend(sarsd)
            
      

        # render for debugging or playing
        # [e.render() for e in self.envs]

        # return averade reward for the agent
        return np.mean([sample[2] for sample in sarsd],dtype=object)

  
    def sample_from_game(self,epsilon, save = True):
        """ 
        samples from every environment in self.envs until it is done or reaches 10000 steps. 

        Parameters:
            epsilon (float): epsilon for the epsilon greedy policy
            save (bool): whether you want to save the generated samples in the agent's buffer
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
        return tuple([np.mean([sarsd[i][j][2] for j in range(len(sarsd[i]))], dtype=object) for i in range(2)])

    def fill_buffers(self,epsilon):
        """ 
        fills the empty buffer with min elements sampling several times from the environemnts
        
        Parameters: 
            epsilon (float): epsilon for the epsilon greedy policy
        """
        
        while(any([self.agents[i].buffer.current_size < self.agents[i].buffer.min_size for i in range(2)])):
            _ = self.sample_from_game(epsilon)

    def fill_buffer(self,epsilon):
        """ fills the empty buffer of the agent 
        
        Parameters: 
            epsilon (float): epsilon for the epsilon greedy policy of the agent
        """

        while(self.agent.buffer.current_size < self.agent.buffer.min_size):
            _ = self.sample_from_game_wrapper(epsilon)
