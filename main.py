from keras_gym_env import ConnectFourEnv
import numpy as np

from agent import DQNAgent
from buffer import Buffer
from sampler import Sampler
from training import train

#Subfolder for Logs
config_name = "default"

#createsummary writer for vusalization in tensorboard    
time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_file_path = f"logs/{config_name}/{time_string}/train"
test_file_path = f"logs/{config_name}/{time_string}/test"
train_summary_writer = tf.summary.create_file_writer(train_file_path)
test_summary_writer = tf.summary.create_file_writer(test_file_path)

# Hyperparameter
BATCH_SIZE = 4
reward_function_adapting_agent = lambda d,r: np.where(r==0,[1,0]) if d else r
EPSILON = 0.01 #TODO

# create buffer
best_buffer = Buffer(100000,1000)
adapting_buffer = Buffer(100000,1000)

# create agent
env = ConnectFourEnv()
best_agent = DQNAgent(env,best_buffer)
adapting_agent = DQNAgent(env, adapting_buffer, reward_function = reward_function_adapting_agent)
agents = [best_agent, adapting_agent]



#states = sampler.sample_from_game(EPSILON)
