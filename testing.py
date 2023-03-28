import numpy as np
import datetime
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from agent import RandomAgent
from sampler import Sampler
from env_wrapper2 import SelfPLayWrapper

def only_right_rewards(reward, right_reward_values, size):
    """ makes sure in the returns are only the rewards from right-reward_values with their proportion"""

    unique, counts = np.unique(reward, return_counts=True)

    # if not all rewards are in reward, add the rest with count 0 
    counts_list = []
    for a in right_reward_values:
        if a in unique:
            v = counts[np.argwhere(unique == a).ravel()[0]]
            counts_list.append(v)
        else:
            counts_list.append(0)
    unique = right_reward_values
    counts = np.array(counts_list)
    
    # count 0 from how many 1 and -1
    counts[1] = size - counts[0] - counts[2] # only needed if move reward is the same as draw reward
    proportion = counts / size * 100
    return unique, proportion

def testing(agent, env_class, size = 100, printing = True, load = None, plot = False):
    """ tests the given agent against a random agent
    
    Parameters: 
        agent (DQNAgent): the agent to test
        env_class: The env class to use
        size (int): over how many games to take the average reward
        printing (bool): if you want the results printed
        load (tuple): (start, stop, step) if you want to load and test several saved models of the given agent  
    """

    if load:
        start, stop, step = load
    else: 
        start, stop, step = 0,1,1

    random_agent = RandomAgent()
    sampler = Sampler(size,agent,env_class, random_agent)
    rewards = []
    env = SelfPLayWrapper(env_class)
    all_end_results = np.array([env.loss_reward,env.draw_reward,env.win_reward])

    for i in range(start,stop,step):
        if load:
            agent.load_models(i)
        reward = sampler.sample_from_game_wrapper(0.0,save = False)

        unique, counts = only_right_rewards(reward, all_end_results, size)
        rewards.append((unique, counts))
        
        if printing:
            if load:
                print(f"Best Agent {i} testing:")
            else:
                print("Best Agent testing:")
            for i,value in enumerate(unique):
                print(f" reward {value}: {counts[i]} percent")

    if plot:
        rewards_dict = [{r[0][j]:r[1][j] for j in range(3)} for r in rewards]
        # next two lines: https://seaborn.pydata.org/examples/wide_data_lineplot.html
        rewards_df = pd.DataFrame(rewards_dict,index=range(start,stop,step))
        sns.lineplot(rewards_df,palette="tab10", linewidth=2.5)
        plt.show()

    return rewards

def testing_adapting(agent, env_class, batch_size = 100, sampling = 10, printing = True, load = None, plot = False):
    """ 
    tests the given agent against all opponents
    get difference between several agents
    
    Parameters: 
        agent (AdaptingDQNAgent): the agent to test
        env_class: The env class to use
        opponents (list): elements of type Agent to use as opponents
        size (int): over how many games to take the average reward
        printing (bool): if you want the results printed
        load (tuple): (start, stop, step) if you want to load and test several saved models of the given agent  
    """

    if load:
        start, stop, step = load
    else: 
        start, stop, step = 0,1,1

    random_agent = RandomAgent()
    sampler = Sampler(batch_size, agent, env_class, random_agent, adapting_agent=True)
    rewards = []
    env = SelfPLayWrapper(env_class)
    all_end_results = np.array([env.loss_reward,env.draw_reward,env.win_reward])

    for i in range(start,stop,step):
        if load:
            agent.load_models(i)
        agent.reset_opponent_level()
        if load:
            print(f"Adapting Agent {i} testing:")
        else:
            print("Adapting Agent testing:")

        rewards_list = []
        for j in range(sampling):
            reward = sampler.sample_from_game_wrapper(0.0,save = False)
            rewards_list.append(reward)

            unique, counts = only_right_rewards(reward, all_end_results, batch_size)
            
            if printing:
                for i,value in enumerate(unique):
                    print(f"{j}*{batch_size} reward {value}: {counts[i]} percent")
                print("Opponent level: ", agent.get_opponent_level())
                print()

        unique, counts = only_right_rewards(rewards_list, all_end_results, batch_size * sampling)
        rewards.append((unique,counts))

    if plot:
        rewards_dict = [{r[0][j]:r[1][j] for j in range(3)} for r in rewards]
        # next two lines: https://seaborn.pydata.org/examples/wide_data_lineplot.html
        rewards_df = pd.DataFrame(rewards_dict,index=range(start,stop,step))
        sns.lineplot(rewards_df,palette="tab10", linewidth=2.5)
        plt.show()

    return rewards

def testing_adapting_different(agent, env_class, opponents, size = 100, printing = True, plot = False):
    """ tests the given agent against opponents
    get differences betwee the several opponents
    
    Parameters: 
        agent (AdaptingDQNAgent): the agent to test
        env_class: The env class to use
        opponents (list): elements of type Agent to use as opponents
        size (int): over how many games to take the average reward
        printing (bool): if you want the results printed
    """

    random_agent = RandomAgent()
    sampler = Sampler(size, agent, env_class, random_agent,adapting_agent=True)
    rewards = []
    env = SelfPLayWrapper(env_class)
    all_end_results = np.array([env.loss_reward,env.draw_reward,env.win_reward])

    for j,o in enumerate([random_agent].extend(opponents)):
        sampler.set_opponent(o)
        reward = sampler.sample_from_game_wrapper(0.0,save = False)

        unique, counts = only_right_rewards(reward, all_end_results, size)
        rewards.append((unique, counts))
        
        if printing:
            print(f"Adapting Agent testing against opponent {j}:")
            for i,value in enumerate(unique):
                print(f" reward {value}: {counts[i]} percent")

    if plot:
        rewards_dict = [{r[0][j]:r[1][j] for j in range(3)} for r in rewards]
        # next two lines: https://seaborn.pydata.org/examples/wide_data_lineplot.html
        rewards_df = pd.DataFrame(rewards_dict,index=range(len(opponents)+1))
        sns.lineplot(rewards_df,palette="tab10", linewidth=2.5)
        plt.show()

    return rewards


if __name__ == "__main__":
    from keras_gym_env import ConnectFourEnv
    from tiktaktoe_env import TikTakToeEnv

    # hyperparameter for testing
    AV = 10000 # how many games to play for each model to test

    # create agent
    best_agent = RandomAgent() # DQNAgent(env,best_buffer, batch = BATCH_SIZE, model_path = model_path_best, polyak_update = POLYAK, inner_iterations = INNER_ITS)

    rewards = testing(best_agent, TikTakToeEnv, AV, printing = True)

    print("done")