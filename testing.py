from keras_gym_env import ConnectFourEnv
import numpy as np
import datetime
import tensorflow as tf

from agent import DQNAgent, RandomAgent
from buffer import Buffer
from sampler import Sampler

def testing(agent, size = 100, printing = True, load = None):
    """ tests the given agent against a random agent
    
    Parameters: 
        agent (DQNAgent): the agent to test
        size (int): over how many games to take the average reward
        printing (bool): if you want the results printed
        load (tuple): (start, stop, step) if you want to load and test several saved models of the given agent  
    """

    if load:
        start, stop, step = load
    else: 
        start, stop, step = 0,1,1

    random_agent = RandomAgent()
    sampler = Sampler(size,agent,random_agent)
    rewards = []

    for i in range(start,stop,step):
        if load:
            agent.load_models(i)
        reward = sampler.sample_from_game_wrapper(0.0,save = False)
        rewards.append(reward)
        if printing:
            if load:
                print(f"Best Agent {i} average reward: {reward}")
            else:
                print(f"Best Agent average reward: {reward}")

    return rewards

if __name__ == "__main__":
    #Subfolder for Logs
    config_name = "best_agent"
    #createsummary writer for vusalization in tensorboard    
    time_string = ""

    best_test_path = f"logs/{config_name}/{time_string}/best_train"
    best_test_writer = tf.summary.create_file_writer(best_test_path)

    model_path_best = f"model/{config_name}/{time_string}/best"

    # Hyperparameter for agent
    iterations = 1000
    INNER_ITS = 500
    BATCH_SIZE = 6
    #reward_function_adapting_agent = lambda d,r: tf.where(d, tf.where(r==0.0,tf.constant(1.0),tf.constant(0.0)), r)
    epsilon = 1 #TODO
    EPSILON_DECAY = 0.99
    POLYAK = 0.9

    # hyperparameter for testing
    AV = 1000 # how many games to play for each model to test

    # create agent
    env = ConnectFourEnv()
    best_buffer = Buffer(100000,1000)
    best_agent = DQNAgent(env,best_buffer, batch = BATCH_SIZE, model_path = model_path_best, polyak_update = POLYAK, inner_iterations = INNER_ITS)

    rewards = testing(49,1000,50,best_agent,size=AV)

    print("done")