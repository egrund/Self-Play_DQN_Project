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

best_train_path = f"logs/{config_name}/{time_string}/best_train"
adapting_train_patch = = f"logs/{config_name}/{time_string}/adapting_train"
best_train_writer = tf.summary.create_file_writer(best_train_path)
adapting_train_writer = tf.summary.create_file_writer(adapting_train_path)

best_test_path = f"logs/{config_name}/{time_string}/best_train"
adapting_test_patch = = f"logs/{config_name}/{time_string}/adapting_train"
best_test_writer = tf.summary.create_file_writer(best_test_path)
adapting_test_writer = tf.summary.create_file_writer(adapting_test_path)

train_writer = [best_train_writer, adapting_train_writer]
test_writer = [best_test_writer, adapting_test_writer]

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

training(agents, iterations, train_writer, test_writer)


#states = sampler.sample_from_game(EPSILON)
