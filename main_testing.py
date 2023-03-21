from keras_gym_env import ConnectFourEnv
import numpy as np
import datetime
import tensorflow as tf
import random as rnd

from env_wrapper import SelfPLayWrapper
from agent import DQNAgent
from buffer import Buffer
from training import train_self_play_best
from testing import testing

### from env_wrapper import ConnectFourSelfPLay
# seeds
seed = 42
np.random.seed(seed)
rnd.seed(seed)

#Subfolder for Logs
config_name = "best_agent_3"
#createsummary writer for vusalization in tensorboard    
time_string = "20230320-040754"
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

AV = 1000 # how many games to play for each model to test

rewards = testing(best_agent,size=AV,load = (100,2700+1,100))# start,stop,step(100,2700+1,100)

print("done")