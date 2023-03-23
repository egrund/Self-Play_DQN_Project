import numpy as np
import datetime
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from agent import RandomAgent
from sampler import Sampler

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

    for i in range(start,stop,step):
        if load:
            agent.load_models(i)
        reward = sampler.sample_from_game_wrapper(0.0,save = False)

        unique, counts = np.unique(reward, return_counts=True)

        # if not all rewards are in reward, add the rest with count 0
        if unique.shape[0] != 3: 
            all = [-1,0,1]
            for a in all:
                if not a in unique:
                    counts = np.insert(counts,a+1,0)
                    unique = np.insert(unique,a+1,a)
        
        # count 0 from how many 1 and -1
        counts[1] = size - counts[0] - counts[2]
        counts = counts / size * 100
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

if __name__ == "__main__":
    from keras_gym_env import ConnectFourEnv
    from tiktaktoe_env import TikTakToeEnv

    # hyperparameter for testing
    AV = 10000 # how many games to play for each model to test

    # create agent
    best_agent = RandomAgent() # DQNAgent(env,best_buffer, batch = BATCH_SIZE, model_path = model_path_best, polyak_update = POLYAK, inner_iterations = INNER_ITS)

    rewards = testing(best_agent, TikTakToeEnv, AV, printing = True)

    print("done")