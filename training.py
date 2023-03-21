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
            unique, percentage = testing(agent, size = 100, printing=True)[0]
            with train_writer.as_default(): # TODO loss writer
                for i,value in enumerate(unique):
                    tf.summary.scalar(f"reward {value}: ", percentage[i], step=i)
                
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
            sampler_time = time.time()
            sampler.set_opponent(old_agent)
            sampler.set_opponent_epsilon(epsilon)
            print("set_opponents",time.time() - sampler_time)
            sampler_time = time.time()
            _ = sampler.sample_from_game_wrapper(epsilon)
            print("sampler_time",time.time() - sampler_time)
            sampler_time_100 += time.time() - sampler_time
        #print("h")
        old_agent = agent.copyAgent(env)

        end = time.time()

        # write summary
            
        # logging the metrics to the log file which is used by tensorboard
        with train_writer.as_default():
            #tf.summary.scalar(f"average_reward", average_reward , step=i) # does not help in self-play
            tf.summary.scalar(f"time per iteration", end-start, step=i)
        outer_time_100 += time.time()-start

def train_adapting(agents, BATCH_SIZE, iterations : int, train_writer, epsilon = 1, epsilon_decay = 0.9): # 
    # create Sampler 
    sampler = Sampler(BATCH_SIZE,agents)
    sampler.fill_buffers(epsilon)
    for i in range(iterations):
        
        # epsilon decay
        epsilon = epsilon * epsilon_decay

        # train the dqn + new samples
        if len(agents)==len(train_writer):
            [agent.train_inner_iteration(train_writer[j], i) for j, agent in enumerate(agents)]
        else:
            raise IndexError("You need the same amount of summary writers and agents.")

        # new sampling + add to buffer
        av_rewards = sampler.sample_from_game(epsilon) # TODO log average rewards

        # write summary
            
        # logging the validation metrics to the log file which is used by tensorboard
        #with train_summary_writer.as_default():
        #    for metric in model.metrics:
        #        tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)

        #print("\n")