from sampler import Sampler
from testing import testing
from agent import RandomAgent
from env_wrapper import SelfPLayWrapper
from keras_gym_env import ConnectFourEnv
import tensorflow as tf
import numpy as np
import time
import tqdm
  
def train_self_play_best(agent, BATCH_SIZE, iterations : int, train_writer, epsilon = 1, epsilon_decay = 0.9, epsilon_min = 0.01,env = SelfPLayWrapper(ConnectFourEnv())):
    """ """
    sampler_time_100 = 0
    inner_time_100 = 0
    outer_time_100 = 0
    # create Sampler 
    old_agent = agent.copyAgent(env)
    with tf.device("/CPU:0"):
        sampler = Sampler(BATCH_SIZE,agent = agent, opponent = RandomAgent())
        sampler.fill_buffer(epsilon)

    for i in tqdm.tqdm(range(iterations)):
        
        start = time.time()
        
        # epsilon decay
        with tf.device("/CPU:0"):
            epsilon = epsilon * epsilon_decay if epsilon > epsilon_min else epsilon_min

        # train agent
        loss = agent.train_inner_iteration(train_writer,i)
        inner_time_100 += time.time() - start
        
        # save model
        if i % 100 == 0:
            agent.save_models(i)

            # testing and save test results in logs
            unique, percentage = testing(agent, size = 100, printing=True)[0]
            with train_writer.as_default(): # TODO loss writer
                for j,value in enumerate(unique):
                    tf.summary.scalar(f"reward {value}: ", percentage[j], step=i)
                
            #prints to get times every 100 iterations
            print("Loss ",i,": ", loss.numpy(), "\n")
            print("inner_iteration_average last 100 iterations: ", inner_time_100/100)             
            print("outer_iteration_average last 100 iterations: ", outer_time_100/100)
            print("Average_Sampling_Time last 100 iterations: ", sampler_time_100/100 , "\n") 
            
            inner_time_100 = 0
            sampler_time_100 = 0
            outer_time_100 = 0

        # new sampling + add to buffer
        with tf.device("/CPU:0"):
            #sampler_time = time.time()
            sampler.set_opponent(old_agent)
            sampler.set_opponent_epsilon(epsilon)
            #print("set_opponents",time.time() - sampler_time)
            #sampler_time = time.time()
            _ = sampler.sample_from_game_wrapper(epsilon)
            #print("sampler_time",time.time() - sampler_time)
            #sampler_time_100 += time.time() - sampler_time
        #print("h")
        old_agent = agent.copyAgent(env)

        end = time.time()

        # write summary
            
        # logging the metrics to the log file which is used by tensorboard
        with train_writer.as_default():
            #tf.summary.scalar(f"average_reward", average_reward , step=i) # does not help in self-play
            tf.summary.scalar(f"time per iteration", end-start, step=i)
        outer_time_100 += time.time()-start