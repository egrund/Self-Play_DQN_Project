from sampler import Sampler
from testing import testing, testing_adapting
from agent import RandomAgent, Agent
from env_wrapper2 import SelfPLayWrapper
import tensorflow as tf
import numpy as np
import time
import tqdm
  
def train_self_play_best(agents : list, env_class, batch_size_sampling, iterations : int, writers : list, epsilon = 1, 
                         epsilon_decay = 0.9, epsilon_min = 0.01, sampling = 1, unavailable_in : bool = False, opponent_epsilon = lambda x: (x/2), d : int = 20, testing_size : int = 100):
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
        losses = [agent.train_inner_iteration(writers[j],i,unavailable_in) for j,agent in enumerate(agents)]
        inner_time += time.time() - start
        
        # save model
        d = 50
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
            #sampler_time = time.time()
            [sampler[j].set_opponent(old_agents[int((j+dist_opponent)%len(agents))]) for j in range(len(agents))]
            dist_opponent = dist_opponent + 1 if dist_opponent < len(agents) -1 else 0
            [s.set_opponent_epsilon( opponent_epsilon(epsilon) ) for s in sampler]
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

def train_adapting(agent : Agent, opponents : list, env_class, batch_size_sampling, iterations : int, writer, epsilon = 1, 
                         epsilon_decay = 0.9, epsilon_min = 0.01, sampling = 1, unavailable_in : bool = False, opponent_epsilon = lambda x: (x/2), d : int = 20,
                         testing_size : int = 100):
    """ """
    sampler_time = 0
    inner_time = 0
    outer_time = 0
    # create Sampler 
    dist_opponent = 0

    with tf.device("/CPU:0"):
        sampler = Sampler(batch_size_sampling,
                           agent = agent, 
                           env_class= env_class, 
                           opponent = RandomAgent(), 
                           unavailable_in=unavailable_in,
                           adapting_agent = True)
        sampler.fill_buffer(epsilon)

    for i in tqdm.tqdm(range(iterations)):
        
        start = time.time()
        
        # epsilon decay
        with tf.device("/CPU:0"):
            epsilon = epsilon * epsilon_decay if epsilon > epsilon_min else epsilon_min

        # train agent
        losses = agent.train_inner_iteration(writer,i,unavailable_in)
        inner_time += time.time() - start
        
        # save model
        # d = 50
        if i % d == 0:
            agent.save_models(i)

            # testing and save test results in logs
            print()
            results = testing_adapting(agent, env_class = env_class, size = testing_size, printing=True)[0]
            print()
            with writer.as_default(): 
                for j,value in enumerate(results[0]):
                    tf.summary.scalar(f"reward {value}: ", results[1][j], step=i)
                
            #prints to get times every 100 iterations
            print(f"Results adapting Agent")
            print(f"Loss {i}: ", losses.numpy())
            print(f"\ninner_iteration_average last {d} iterations: ", inner_time/d)             
            print(f"outer_iteration_average last {d} iterations: ", outer_time/d)
            print(f"Average_Sampling_Time last {d} iterations: ", sampler_time/d , "\n") 
            
            inner_time = 0
            sampler_time = 0
            outer_time = 0

        # new sampling + add to buffer
        with tf.device("/CPU:0"):
            #sampler_time = time.time()
            sampler.set_opponent(opponents[int(i%len(opponents))])
            agent.reset_opponent_level()
            sampler.set_opponent_epsilon( opponent_epsilon(epsilon) )
            #print("set_opponents",time.time() - sampler_time)
            sampler_start = time.time()
            _ = [sampler.sample_from_game_wrapper(epsilon) for _ in range(sampling)]
            #print("sampler_time",time.time() - sampler_time)
            sampler_time += time.time() - sampler_start
        #print("h")

        end = time.time()

        # write summary
            
        # logging the metrics to the log file which is used by tensorboard
        #with train_writer[0].as_default():
            #tf.summary.scalar(f"average_reward", average_reward , step=i) # does not help in self-play
            #tf.summary.scalar(f"time per iteration", end-start, step=i)
        outer_time += time.time()-start