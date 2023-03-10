from keras_gym_env import ConnectFourEnv
import numpy as np
import datetime
import tensorflow as tf
import random as rnd

from agent import DQNAgent
from buffer import Buffer
from training import train_self_play_best

# seeds
seed = 42
np.random.seed(42)
rnd.seed(42)


#Subfolder for Logs
config_name = "best_agent"
#createsummary writer for vusalization in tensorboard    
time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# time_string = ""

best_train_path = f"logs/{config_name}/{time_string}/best_train"
#adapting_train_path = f"logs/{config_name}/{time_string}/adapting_train"
best_train_writer = tf.summary.create_file_writer(best_train_path)
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
iterations = 5000
INNER_ITS = 100
BATCH_SIZE = 512
#reward_function_adapting_agent = lambda d,r: tf.where(d, tf.where(r==0.0,tf.constant(1.0),tf.constant(0.0)), r)
epsilon = 1
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
POLYAK = 0.9

# create buffer
best_buffer = Buffer(capacity = 100000,min_size = 30000)
#adapting_buffer = Buffer(100000,1000)

# create agent
env = ConnectFourEnv()
best_agent = DQNAgent(env,best_buffer, batch = BATCH_SIZE, model_path = model_path_best, polyak_update = POLYAK, inner_iterations = INNER_ITS)
# adapting_agent = DQNAgent(env, adapting_buffer, batch = BATCH_SIZE, model_path = model_path_adapting, reward_function = reward_function_adapting_agent)
# agents = [best_agent, adapting_agent]

train_self_play_best(best_agent, BATCH_SIZE, iterations, best_train_writer, epsilon= epsilon, epsilon_decay = EPSILON_DECAY,epsilon_min = EPSILON_MIN)


print("done")
