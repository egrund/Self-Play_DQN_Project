from env_wrapper import SelfPLayWrapper

import numpy as np
import datetime
import tensorflow as tf
import random as rnd


from agent import DQNAgent
from buffer import Buffer
from training import train_self_play_best

# seeds
seed = 42
np.random.seed(seed)
rnd.seed(seed)

#Subfolder for Logs
config_name = "best_agent"
#createsummary writer for vusalization in tensorboard    
time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# time_string = ""

best_train_path = f"logs/{config_name}/{time_string}/best_train"
best_train_writer = tf.summary.create_file_writer(best_train_path)
model_path_best = f"model/{config_name}/{time_string}/best"

# Hyperparameter
iterations = 5000
INNER_ITS = 50
BATCH_SIZE = 512
#reward_function_adapting_agent = lambda d,r: tf.where(d, tf.where(r==0.0,tf.constant(1.0),tf.constant(0.0)), r)
epsilon = 1
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
POLYAK = 0.9
dropout_rate = 0.2, 
normalisation = True

# create buffer
best_buffer = Buffer(capacity = 100000,min_size = 5000)

# create agent
#env = ConnectFourEnv()
env = SelfPLayWrapper()
best_agent = DQNAgent(env,best_buffer, batch = BATCH_SIZE, model_path = model_path_best, polyak_update = POLYAK, inner_iterations = INNER_ITS, dropout_rate = dropout_rate, normalisation = normalisation)

train_self_play_best(best_agent, BATCH_SIZE, iterations, best_train_writer, epsilon= epsilon, epsilon_decay = EPSILON_DECAY,epsilon_min = EPSILON_MIN,env = env)


print("done")
