from sampler import Sampler
from testing import testing
from agent import RandomAgent
from env_wrapper2 import SelfPLayWrapper
import tensorflow as tf
import numpy as np
import time
import tqdm
  
def train_self_play_best(agents : list, env_class, batch_size_sampling, iterations : int, train_writer : list, test_writer : list, epsilon = 1, epsilon_decay = 0.9, epsilon_min = 0.01, sampling = 1, unavailable_in : bool = False):
    """ """
    sampler_time = 0
    inner_time = 0
    outer_time = 0
    # create Sampler 
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
        losses = [agent.train_inner_iteration(train_writer[j],i) for j,agent in enumerate(agents)]
        inner_time += time.time() - start
        
        # save model
        d = 50
        if i % d == 0:
            [agent.save_models(i) for agent in agents]

            # testing and save test results in logs
            print()
            results = [testing(agent, env_class = env_class, size = 1000, printing=True)[0] for agent in agents] # unique, percentage
            print()
            for ai in range(len(agents)):
                with test_writer[ai].as_default(): 
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
            #sampler_time = time.time()
            [sampler[j].set_opponent(old_agents[int((j+dist_opponent)%len(agents))]) for j in range(len(agents))]
            dist_opponent = dist_opponent + 1 if dist_opponent < len(agents) -1 else 0
            [s.set_opponent_epsilon(epsilon/2) for s in sampler] #TODO make a variable
            #print("set_opponents",time.time() - sampler_time)
            sampler_start = time.time()
            _ = [[s.sample_from_game_wrapper(epsilon) for _ in range(sampling)] for s in sampler]
            #print("sampler_time",time.time() - sampler_time)
            sampler_time += time.time() - sampler_start
        #print("h")
        old_agents = [agent.copyAgent(SelfPLayWrapper(env_class)) for agent in agents]

        end = time.time()

        # write summary
            
        # logging the metrics to the log file which is used by tensorboard
        #with train_writer[0].as_default():
            #tf.summary.scalar(f"average_reward", average_reward , step=i) # does not help in self-play
            #tf.summary.scalar(f"time per iteration", end-start, step=i)
        outer_time += time.time()-start