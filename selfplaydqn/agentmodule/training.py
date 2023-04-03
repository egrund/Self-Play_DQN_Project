from envs.sampler import Sampler
from agentmodule.testing import testing, testing_adapting
from agentmodule.agent import RandomAgent, Agent
from envs.envwrapper2 import SelfPLayWrapper
import tensorflow as tf
import numpy as np
import time
import tqdm
  
def train_self_play_best(agents : list, env_class, batch_size_sampling : int, iterations : int, writers : list, epsilon = 1, 
                         epsilon_decay : float = 0.9, epsilon_min : float = 0.01, sampling : int = 1, unavailable_in : bool = False, opponent_epsilon = lambda x: (x/2), 
                         d : int = 20, testing_size : int = 100):
    """ 
    A training algorithm to train a DQN agent 
    
    Parameters:
        agents (list): a list of agents to train (can be just one element inside)
        env_class (class): The class name of the environment to use
        batch_size_sampling (int): the batch size which determines how many games to play at the same time during each training iteration
        iterations (int): how many iterations to train for
        writers (list): a summary writer for each agent
        epsilon (float/int): The epsilon value to begin with
        epsilon_decay (float): how much to keep from epsilon each iteration
        epsilon_min (float): at which point not to decrease epsilon anymore
        sampling (int): how often to sample batch_size_sampling many games
        unavailable_in (bool): whether the agent(s) should be able to sample unavailable actions
        opponent_epsilon (func): a function to calculate the opponents epsilon from the agents epsilon
        d (int): every d iteration the agents models will be saved and tested
        testing_size (int): how many games to test for
    """

    sampler_time = 0
    inner_time = 0
    outer_time = 0

    # create Sampler and fill buffer
    old_agents = [agent.copyAgent(SelfPLayWrapper(env_class)) for agent in agents]
    dist_opponent = 0

    with tf.device("/CPU:0"):
        sampler = [Sampler(batch_size_sampling,
                           agent = agent, 
                           env_class= env_class, 
                           opponent = RandomAgent(), 
                           unavailable_in=unavailable_in) for agent in agents]
        [s.fill_buffer(epsilon) for s in sampler]

    for i in tqdm.tqdm(range(iterations)):
        
        start = time.time()
        
        # epsilon decay
        with tf.device("/CPU:0"):
            epsilon = epsilon * epsilon_decay if epsilon > epsilon_min else epsilon_min

        # train agent
        losses = [agent.train_inner_iteration(writers[j],i,unavailable_in) for j,agent in enumerate(agents)]
        inner_time += time.time() - start
        
        # save and test model
        if i % d == 0:
            [agent.save_models(i) for agent in agents]

            # testing and save test results in logs
            print()
            results = [testing(agent, env_class = env_class, size = testing_size, printing=True)[0] for agent in agents] # unique, percentage
            print()
            for ai in range(len(agents)):
                with writers[ai].as_default(): 
                    for j,value in enumerate(results[ai][0]):
                        tf.summary.scalar(f"reward {value}: ", results[ai][1][j], step=i)
                
                #prints to get times every 100 iterations
                print(f"Results Agent {ai}")
                print(f"Loss {i}: ", losses[ai].numpy())
            print(f"\ninner_iteration_average last {d} iterations: ", inner_time/d)             
            print(f"outer_iteration_average last {d} iterations: ", outer_time/d)
            print(f"Average_Sampling_Time last {d} iterations: ", sampler_time/d , "\n") 
            
            inner_time = 0
            sampler_time = 0
            outer_time = 0

        # new sampling + add to buffer
        with tf.device("/CPU:0"):

            [sampler[j].set_opponent(old_agents[int((j+dist_opponent)%len(agents))]) for j in range(len(agents))]
            dist_opponent = dist_opponent + 1 if dist_opponent < len(agents) -1 else 0

            sampler_start = time.time()
            # this automatically saves to the buffer of the agent
            _ = [[s.sample_from_game_wrapper(epsilon) for _ in range(sampling)] for s in sampler]
            sampler_time += time.time() - sampler_start

        # save agents weights as opponents of the next iteration
        old_agents = [agent.copyAgent(SelfPLayWrapper(env_class)) for agent in agents]

        outer_time += time.time()-start