import numpy as np
import datetime
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from agentmodule.agent import RandomAgent, Agent
from envs.sampler import Sampler
from envs.env_wrapper2 import SelfPLayWrapper

def only_right_rewards(reward, right_reward_values, size, index_calculate_new = 1):
    """ makes sure in the returns are only the rewards from right-reward_values with their proportion 
    
    Parameters:
        reward (np.array): an array containing all the rewards gotten
        right_reward_values (np.array): contains all the rewards we want percentage values for in the output
        size (int): how many games were played to get this many rewards
        index_calculate_new (i): the index which should be calculated from the percentage of the other rewards. e.g. draw reward as it is the same as move reward
    """

    unique, counts = np.unique(reward, return_counts=True)

    # add all rewards, the ones not in reward with count 0 
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
    counts[index_calculate_new] = size - np.sum(counts) + counts[index_calculate_new]  # only needed if move reward is the same as draw reward

    # make a percentage out of the values
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
        plot(bool): Whether the results should be plotted
    """

    if load != None:
        start, stop, step = load
    else: 
        # if load == None we just do one step
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
            for j,value in enumerate(unique):
                print(f" reward {value}: {counts[j]} percent")

    if plot:
        rewards_dict = [[{"Reward":r[0][j], "Percentage":r[1][j], "Index":idx} for j in range(3)] for idx,r in zip(range(start,stop,step),rewards)]
        rewards_df = pd.DataFrame([rd for subrd in rewards_dict for rd in subrd])
        axes = sns.lineplot(rewards_df, linewidth=2, palette= "tab10",x="Index", y="Percentage", hue="Reward")
        axes.set_xlabel("Index",size="xx-large")
        axes.set_ylabel("Percentage",size="xx-large")
        axes.legend(prop={'size':15}, title = "Reward", title_fontsize = "xx-large")
        axes.set_yticks(range(0,101,10))
        axes.grid(True,color = 'black', linestyle="--",linewidth=0.5)
        plt.show()

    return rewards

def testing_dif_agents(agent, env_class, size = 100, load = None, printing = True, plot = False):
    """ tests the given agent against a random agent
    
    Parameters: 
        agent (DQNAgent): the agent to test
        env_class: The env class to use
        size (int): over how many games to take the average reward
        load (tuple): (config_list, indices_list) if you want to load and test several saved models of the given agent  
        printing (bool): if you want the results printed
        plot(bool): Whether the results should be plotted
    """

    config_list, indices_list = load

    random_agent = RandomAgent()
    sampler = Sampler(size,agent,env_class, random_agent)
    rewards = []
    env = SelfPLayWrapper(env_class)
    all_end_results = np.array([env.loss_reward,env.draw_reward,env.win_reward])

    for i in range(len(config_list)):
        if load:
            agent.model_path = f"model/" + config_list[i]

        for idx in indices_list[i]:
            agent.load_models(idx)
            reward = sampler.sample_from_game_wrapper(0.0,save = False)

            unique, counts = only_right_rewards(reward, all_end_results, size)
            rewards.append((unique, counts, config_list[i], idx))
            
            if printing:
                print(f"Best Agent {config_list[i]};{idx}testing:")
                for j,value in enumerate(unique):
                    print(f" reward {value}: {counts[j]} percent")

    if plot:
        rewards_dict = [[{"Reward":r[0][j], "Percentage":r[1][j], "config_index": r[2] + str(r[3])} for j in range(3)] for r in rewards]
        rewards_df = pd.DataFrame([rd for subrd in rewards_dict for rd in subrd])
        axes = sns.barplot(rewards_df, linewidth=2, palette= "tab10",y="config_index", x="Percentage", hue="Reward", orient="h")
        axes.set_xlabel("Percentage",size="xx-large")
        #axes.set_ylabel("config_index",size="xx-large")
        axes.legend(prop={'size':15}, title = "Reward", title_fontsize = "xx-large", loc="center")
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
        plot(bool): Whether the results should be plotted
    """

    if load != None:
        start, stop, step = load
    else: 
        # if load == None we just do one step
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
                for h,value in enumerate(unique):
                    print(f"{j}*{batch_size} reward {value}: {counts[h]} percent")
                print("Game balance: ", agent.get_game_balance())
                print("Opponent level: ", agent.opponent_level.numpy())
                print()

        unique, counts = only_right_rewards(rewards_list, all_end_results, batch_size * sampling)
        rewards.append((unique,counts))

    if plot:
        rewards_dict = [[{"Reward":r[0][j], "Percentage":r[1][j], "Index":idx} for j in range(3)] for idx,r in zip(range(start,stop,step),rewards)]
        rewards_df = pd.DataFrame([rd for subrd in rewards_dict for rd in subrd])
        axes = sns.lineplot(rewards_df, linewidth=2, palette= "tab10",x="Index", y="Percentage", hue="Reward")
        axes.set_xlabel("Index",size="xx-large")
        axes.set_ylabel("Percentage",size="xx-large")
        axes.legend(prop={'size':15}, title = "Reward", title_fontsize = "xx-large")
        axes.set_yticks(range(0,101,10))
        axes.grid(True,color = 'black', linestyle="--",linewidth=0.5)
        plt.show()

    return rewards

def testing_adapting_dif_epsilon_opponents(agent, env_class, opponent : Agent, opponent_size = 10, batch_size = 100, 
                                           sampling = 10, printing = True, plot = False, adapting = False):
    """ tests the given agent against opponents
    get differences betwee the opponents with different epsilon values
    
    Parameters: 
        agent (AdaptingDQNAgent): the agent to test
        env_class: The env class to use
        opponent (Agent): besta agent to use as opponents with different epsilon values
        batch_size (int): over how many games to play the average reward from at the same time
        sampling (int): how often to sample batch_size many games after each other
        printing (bool): if you want the results printed
        plot(bool): Whether the results should be plotted
        adapting (bool): Whether the agent should get in game information about the average reward
    """

    sampler = Sampler(batch_size, agent, env_class, opponent,adapting_agent=adapting) 
    rewards = []
    env = SelfPLayWrapper(env_class)
    all_end_results = np.array([env.loss_reward,env.draw_reward,env.win_reward])

    for e in np.linspace(1,0,opponent_size):
        if adapting:
            agent.reset_game_balance()
            agent.reset_opponent_level()
        sampler.set_opponent_epsilon(e)

        if printing:
            print("Test Agent against opponent with epsilon ", e)

        rewards_list = []
        for j in range(sampling):
            reward = sampler.sample_from_game_wrapper(0.0,save = False)
            rewards_list.extend(reward)

            unique, counts = only_right_rewards(reward, all_end_results, batch_size)
            
            if printing:
                for h,value in enumerate(unique):
                    print(f"{j}*{batch_size} reward {value}: {counts[h]} percent")
                if adapting:
                    print("Game balance: ", agent.get_game_balance())
                    print("Opponent level: ", agent.opponent_level.numpy())
                print()

        unique, counts = only_right_rewards(rewards_list, all_end_results, batch_size * sampling)
        rewards.append((unique,counts))

    if plot:
        rewards_dict = [[{"Reward":r[0][j], "Percentage":r[1][j], "Epsilon":idx} for j in range(3)] for idx,r in zip(np.linspace(1,0,opponent_size),rewards)]
        rewards_df = pd.DataFrame([rd for subrd in rewards_dict for rd in subrd])
        axes = sns.lineplot(rewards_df, linewidth=2, palette= "tab10",x="Epsilon", y="Percentage", hue="Reward")
        axes.set_xlabel("Epsilon",size="xx-large")
        axes.set_ylabel("Percentage",size="xx-large")
        axes.legend(prop={'size':15}, title = "Reward", title_fontsize = "xx-large")
        axes.set_yticks(range(0,101,10))
        axes.grid(True,color = 'black', linestyle="--",linewidth=0.5)
        plt.show()

    return rewards


if __name__ == "__main__":
    from envs.keras_gym_env import ConnectFourEnv
    from envs.tiktaktoe_env import TikTakToeEnv

    # hyperparameter for testing
    AV = 10000 # how many games to play for each model to test

    # create agent
    best_agent = RandomAgent() # DQNAgent(env,best_buffer, batch = BATCH_SIZE, model_path = model_path_best, polyak_update = POLYAK, inner_iterations = INNER_ITS)

    rewards = testing(best_agent, TikTakToeEnv, AV, printing = True)

    print("done")