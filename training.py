from sampler import Sampler
from testing import testing
from agent import RandomAgent
from env_wrapper import ConnectFourSelfPLay
import tensorflow as tf
import numpy as np
import time
import tqdm
  
def train_self_play_best(agent, BATCH_SIZE, iterations : int, train_writer, epsilon = 1, epsilon_decay = 0.9, epsilon_min = 0.01,env = ConnectFourSelfPLay()): # 
    """ """
    # create Sampler 
    old_agent = agent.copy(env)
    sampler = Sampler(BATCH_SIZE,agent = agent, opponent = old_agent)
    sampler.fill_buffer(epsilon)
    for i in tqdm.tqdm(range(iterations)):
        
        start = time.time()
        # epsilon decay
        epsilon = epsilon * epsilon_decay if epsilon > epsilon_min else epsilon_min

        # train agent
        agent.train_inner_iteration(train_writer,i)

        # save model
        if i % 100 == 0:
            agent.save_models(i)
            rewards = testing(agent, size = 100, printing=True)

            #with train_writer.as_default():
                #tf.summary.scalar(f"average_reward", rewards[0][0], step=i)

        # new sampling + add to buffer
        sampler.set_opponent(old_agent)
        _ = sampler.sample_from_game_wrapper(epsilon)
        old_agent = agent.copy(env)

        end = time.time()

        # write summary
            
        # logging the metrics to the log file which is used by tensorboard
        with train_writer.as_default():
            #tf.summary.scalar(f"average_reward", average_reward , step=i) # does not help in self-play
            tf.summary.scalar(f"time per iteration", end-start, step=i)

        #print("\n")

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