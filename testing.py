import numpy as np
import datetime
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from agent import RandomAgent, Agent
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
        rewards_dict = [[{"reward":r[0][j], "percentage":r[1][j], "index":idx} for j in range(3)] for idx,r in zip(range(start,stop,step),rewards)]
        rewards_df = pd.DataFrame([rd for subrd in rewards_dict for rd in subrd])
        axes = sns.lineplot(rewards_df, linewidth=2, palette= "tab10",x="index", y="percentage", hue="reward")
        axes.grid(True,color = 'black', linestyle="--",linewidth=0.5)
        plt.show()

    return rewards

def testing_adapting(agent, env_class, batch_size = 100, sampling = 10, printing = True, load = None, plot = False):
    """ 
    tests the given agent against all opponents
    get difference between several agents
    
    Parameters: 
        agent (AdaptingDQNAgent): the agent to test
        env_class: The env class to use
        batch_size (int): over how many games to take the average reward
        sampling (int): how often to sample batch_size many games after each other
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
        agent.reset_game_balance()
        agent.reset_opponent_level()
        if load and printing:
            print(f"Adapting Agent {i} testing:")
        elif printing:
            print("Adapting Agent testing:")

        rewards_list = []
        for j in range(sampling):
            reward = sampler.sample_from_game_wrapper(0.0,save = False)
            rewards_list.extend(reward)

            unique, counts = only_right_rewards(reward, all_end_results, batch_size)
            
            if printing:
                for i,value in enumerate(unique):
                    print(f"{j}*{batch_size} reward {value}: {counts[i]} percent")
                print("Game balance: ", agent.get_game_balance())
                print("Opponent level: ", agent.opponent_level.numpy())
                print()

        unique, counts = only_right_rewards(rewards_list, all_end_results, batch_size * sampling)
        rewards.append((unique,counts))

    if plot:
        rewards_dict = [[{"reward":r[0][j], "percentage":r[1][j], "index":idx} for j in range(3)] for idx,r in zip(range(start,stop,step),rewards)]
        rewards_df = pd.DataFrame([rd for subrd in rewards_dict for rd in subrd])
        axes = sns.lineplot(rewards_df, linewidth=2, palette= "tab10",x="index", y="percentage", hue="reward")
        axes.grid(True,color = 'black', linestyle="--",linewidth=0.5)
        plt.show()

    return rewards

def testing_adapting_dif_epsilon_opponents(agent, env_class, opponent : Agent, opponent_size = 10, batch_size = 100, sampling = 10, printing = True, plot = False):
    """ tests the given agent against opponents
    get differences betwee the opponents with different epsilon values
    
    Parameters: 
        agent (AdaptingDQNAgent): the agent to test
        env_class: The env class to use
        opponent (Agent): besta agent to use as opponents with different epsilon values
        batch_size (int): over how many games to take the average reward
        sampling (int): how often to sample batch_size many games after each other
        printing (bool): if you want the results printed
    """

    sampler = Sampler(batch_size, agent, env_class, opponent,adapting_agent=False) #True)
    rewards = []
    env = SelfPLayWrapper(env_class)
    all_end_results = np.array([env.loss_reward,env.draw_reward,env.win_reward])

    for e in np.linspace(1,0,opponent_size):
        #agent.reset_game_balance()
        #agent.reset_opponent_level()
        sampler.set_opponent_epsilon(e)

        if printing:
            print("Test Agent against opponent with epsilon ", e)

        rewards_list = []
        for j in range(sampling):
            reward = sampler.sample_from_game_wrapper(0.0,save = False)
            rewards_list.extend(reward)

            unique, counts = only_right_rewards(reward, all_end_results, batch_size)
            
            if printing:
                for i,value in enumerate(unique):
                    print(f"{j}*{batch_size} reward {value}: {counts[i]} percent")
                #print("Game balance: ", agent.get_game_balance())
                #print("Opponent level: ", agent.opponent_level.numpy())
                print()

        unique, counts = only_right_rewards(rewards_list, all_end_results, batch_size * sampling)
        rewards.append((unique,counts))

    if plot:
        rewards_dict = [[{"reward":r[0][j], "percentage":r[1][j], "epsilon":idx} for j in range(3)] for idx,r in zip(np.linspace(1,0,opponent_size),rewards)]
        rewards_df = pd.DataFrame([rd for subrd in rewards_dict for rd in subrd])
        axes = sns.lineplot(rewards_df, linewidth=2, palette= "tab10",x="epsilon", y="percentage", hue="reward")
        axes.grid(True,color = 'black', linestyle="--",linewidth=0.5)
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