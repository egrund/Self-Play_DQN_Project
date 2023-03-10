from keras_gym_env import ConnectFourEnv
import numpy as np
import datetime
import tensorflow as tf

from agent import DQNAgent, RandomAgent
from buffer import Buffer
from sampler import Sampler

#Subfolder for Logs
config_name = "best_agent"
#createsummary writer for vusalization in tensorboard    
time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# time_string = ""

best_test_path = f"logs/{config_name}/{time_string}/best_train"
#adapting_train_path = f"logs/{config_name}/{time_string}/adapting_train"
best_test_writer = tf.summary.create_file_writer(best_test_path)
#dapting_train_writer = tf.summary.create_file_writer(adapting_train_path)

#best_test_path = f"logs/{config_name}/{time_string}/best_test"
#adapting_test_path = f"logs/{config_name}/{time_string}/adapting_test"
#best_test_writer = tf.summary.create_file_writer(best_test_path)
#adapting_test_writer = tf.summary.create_file_writer(adapting_test_path)

#train_writer = [best_train_writer, adapting_train_writer]
#test_writer = [best_test_writer, adapting_test_writer]

model_path_best = f"model/{config_name}/{time_string}/best"
#model_path_adapting = f"model/{config_name}/{time_string}/adapting"

# Hyperparameter
iterations = 1000
INNER_ITS = 500
BATCH_SIZE = 6
#reward_function_adapting_agent = lambda d,r: tf.where(d, tf.where(r==0.0,tf.constant(1.0),tf.constant(0.0)), r)
epsilon = 1 #TODO
EPSILON_DECAY = 0.99
POLYAK = 0.9

I = 249 # which model to load
AV = 100 # how many games to play for each model to test

# create buffer
best_buffer = Buffer(100000,1000)
#adapting_buffer = Buffer(100000,1000)

# create agent
env = ConnectFourEnv()
best_agent = DQNAgent(env,best_buffer, batch = BATCH_SIZE, model_path = model_path_best, polyak_update = POLYAK, inner_iterations = INNER_ITS)

best_agent.load_models(I)

random_agent = RandomAgent()
sampler = Sampler(AV,[best_agent,random_agent])

rewards, _ = sampler.sample_from_game(0.0,save = False)
print(f"Best Agent {I} average reward: {rewards}")



    



print("done")