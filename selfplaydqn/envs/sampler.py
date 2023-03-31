from env_wrapper2 import SelfPLayWrapper
import numpy as np
import time
import tensorflow as tf

class Sampler:
    """ 
    Implements an algorithm to sample from a self-play environment using two different agents

    Attributes: 
        envs (list): List of all the environments to sample from
        batch (int): how many environments to sample from at the same time
        agent (Agent): the agent to sample for
        opponent (Agent) or list of Agents: The opponent to use or use several
        single_opponent (bool): whether opponent is a single agent or a list
        opponent_epsilon (float): the epsilon value to use for the epsilon greedy policy of the opponent(s)
        unavailable_in (bool): if True, the sampling agent can decide for unavailable actions and gets a penalty as reward back and the env state does not change
        adapting_agent (bool): whether the sampling agent is an adapting agents that wants information about the rewards other than putting it in the buffer. 
            Also there is information about the average rewards of the past and the opponents level (decided by the agent) added to the buffer. 
    """

    def __init__(self,batch,agent, env_class, opponent,opponent_epsilon : float = 0, unavailable_in : bool = False, adapting_agent : bool = False):

        self.envs = [SelfPLayWrapper(env_class, opponent,opponent_epsilon) for _ in range(batch)]
        self.batch = batch
        self.agent = agent
        self.opponent = opponent
        self.single_opponent = True
        self.opponent_epsilon = opponent_epsilon
        self.unavailable_in = unavailable_in
        self.adapting_agent = adapting_agent

    def set_opponent(self, opponent):
        """ sets a new single opponent """
        [env.set_opponent(opponent) for env in self.envs]
        self.opponent = opponent 

    def set_multiple_opponents(self, opponents : list):
        """ sets multiple opponents for sampling. 
        Opponents has to be broadcastable to self.envs which are batch much."""
        if self.batch % len(opponents) != 0:
            raise ValueError("Opponents has to be broadcastable to self.envs which are batch much")
        # broadcast shapes together
        self.opponent = opponents * int((self.batch / len(opponents)))
        
        for env,o in zip(self.envs,self.opponent):
            env.set_opponent(o)
        self.single_opponent = False

    def set_opponent_epsilon(self,epsilon):
        """ sets a new opponent epsilon """
        [env.set_epsilon(epsilon) for env in self.envs]
        self.opponent_epsilon = epsilon

    def sample_from_game_wrapper(self,epsilon : float, save = True):
        """ 
        samples from env wrappers 
        
        Parameters:
            epsilon (float): the epsilon to use for the epsilon greedy policy of the sampling agent
            save (bool): whether the created samples should be saved in the sampling agents buffer
        """
        
        sarsd = []
        current_envs = self.envs
        agent_turn = np.random.randint(0,2,(self.batch,))
        observations = np.array([env.opponent_starts() if whether else env.reset() for whether,env in zip(agent_turn,current_envs)])    
        
        for e in range(10000):
            
            # agent turn
            available_actions = [env.available_actions for env in current_envs]
            available_actions_bool = [env.available_actions_mask for env in current_envs]
            actions = self.agent.select_action_epsilon_greedy(epsilon, observations,available_actions, available_actions_bool, unavailable = self.unavailable_in)

            if self.single_opponent:
                o_0 = np.array([env.step_player(actions[i],self.unavailable_in) for i,env in enumerate(current_envs)]) # only state for opponent imput

                # opponent turn
                available_actions = [env.available_actions for env in current_envs]
                available_actions_bool = [env.available_actions_mask for env in current_envs]
                o_actions = self.opponent.select_action_epsilon_greedy(self.opponent_epsilon,o_0,available_actions, available_actions_bool, False)
                results = [env.step_opponent(o_actions[i]) for i,env in enumerate(current_envs)] # new state, reward, done,
            else:
                # here the different opponents in the envs are used
                results = [env.step(actions[i],self.unavailable_in) for i,env in enumerate(current_envs)]

            # if adapting agent give the agent information about the player level (only reward from done envs)
            if self.adapting_agent:
                self.agent.add_game_balance_information([results[i][1] for i in range(len(current_envs)) if results[i][2]])

            # get next available actions, as we also save that in the buffer
            available_actions_bool = [env.available_actions_mask for env in current_envs]

            # bring everything in the right order
            if not self.adapting_agent:
                # state, action, reward, new state, done, next_available_actions_bool
                results = [[observations[i],actions[i],results[i][1],results[i][0],results[i][2], available_actions_bool[i]] for i in range(len(current_envs))]
            else:
                # state, action, reward, new state, done, next_available_actions_bool, game_balance, opponent_level
                results = [[observations[i],actions[i],results[i][1],results[i][0],results[i][2], available_actions_bool[i],self.agent.get_game_balance(), self.agent.opponent_level] for i in range(len(current_envs))] 

            sarsd.extend(results)
            
            # remove envs and last observations of envs that are done
            observations = np.array([results[i][3] for i in range(len(current_envs)) if not results[i][4]])
            current_envs = np.array([current_envs[i] for i in range(len(current_envs)) if not results[i][4]])

            # check if all envs are done
            if observations.shape == (0,):
                break

            # remind the agent whether his last action was an unavailable action (otherwise we would get a lot of the same samples out of it)
            # could only do it after removing done envs from current envs so the indices are right for the next round of sampling
            if self.unavailable_in:
                w_0 = [env.last_wrong for env in current_envs]
                self.agent.add_do_random(np.argwhere(w_0).ravel())

        # save data in buffer
        if save:
            self.agent.buffer.extend(sarsd)

        # return rewards for the agent
        return [sample[2] for sample in sarsd]
    
    def fill_buffer(self,epsilon):
        """ fills the empty buffer of the agent until its min fill value
        
        Parameters: 
            epsilon (float): epsilon for the epsilon greedy policy of the sampling agent
        """

        while(self.agent.buffer.current_size < self.agent.buffer.min_size):
            _ = self.sample_from_game_wrapper(epsilon)
