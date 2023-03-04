from keras_gym_env import ConnectFourEnv
import numpy as np

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

    def sample_from_game(self,epsilon):
        """ """
        sarsd = [[],[]]
        agent_turn = np.random.randint(0,2,(self.batch,))
        action_space_size = self.envs[0].action_space.n
        current_envs = [np.extract(agent_turn == i,self.envs) for i in range(2)]

        observations = [np.array([env.reset() for env in which_agent ]) for which_agent in current_envs]

        for _ in range(10000):

            # get all the actions from the agents
            actions = [self.agents[i].select_action_epsilon_greedy(epsilon, observations[i]) for i in range(2) if observations[i].shape != 0]

            # do the step in every env
            print("CE ", current_envs)
            print("A ",actions)
            results = [[env.step(a) for env , a in which_agent]for which_agent in np.concatenate((np.expand_dims(current_envs,-1),np.expand_dims(actions,-1)),axis=-1)]

            # add first observation to new results
            results = [np.concatenate((observations[i],actions[i],results[i][:][1],results[i][:][0],results[i][:][0]),1) for i in range(2)] # state, action, reward, new state, done

            [sarsd[i].extend(results[i]) for i in range(2)] 

            # get observations for next turn, and remove everything that is done already
            observations = [np.extract( not results[i][:][2], results[i][:][0]) for i in range(2)]
            current_envs = [np.extract(not results[i][:][2], current_envs) for i in range(2)]

            # check if all envs are done
            if all([observations[i].shape == 0 for i in range(2)]):
                break

            # let them switch sides for the next turn
            current_envs.reverse()
            observations.reverse()

        # save data in buffer
        [self.agents[i].buffer.extend(sarsd[i]) for i in range(2)]

    def fill_buffers(self,epsilon):
        """ fills the empty buffer with min elements """
        
        while(any([self.agents[i].buffer.current_size < self.agents[i].buffer.min_size for i in range(2)])):
            self.sample_from_game(epsilon)
