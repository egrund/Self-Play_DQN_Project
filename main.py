from keras_gym_env import ConnectFourEnv
import numpy as np
import datetime
import tensorflow as tf

from agent import DQNAgent
from buffer import Buffer
from training import train

#Subfolder for Logs
config_name = "default"
#createsummary writer for vusalization in tensorboard    
time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# time_string = ""

best_train_path = f"logs/{config_name}/{time_string}/best_train"
adapting_train_path = f"logs/{config_name}/{time_string}/adapting_train"
best_train_writer = tf.summary.create_file_writer(best_train_path)
adapting_train_writer = tf.summary.create_file_writer(adapting_train_path)

best_test_path = f"logs/{config_name}/{time_string}/best_test"
adapting_test_path = f"logs/{config_name}/{time_string}/adapting_test"
best_test_writer = tf.summary.create_file_writer(best_test_path)
adapting_test_writer = tf.summary.create_file_writer(adapting_test_path)

train_writer = [best_train_writer, adapting_train_writer]
test_writer = [best_test_writer, adapting_test_writer]

model_path_best = f"model/{config_name}/{time_string}/best"
model_path_adapting = f"model/{config_name}/{time_string}/adapting"

# Hyperparameter
iterations = 12
BATCH_SIZE = 6
reward_function_adapting_agent = lambda d,r: np.where(r==0,[1,0]) if d else r
EPSILON = 1 #TODO

# create buffer
best_buffer = Buffer(100000,1000)
adapting_buffer = Buffer(100000,1000)

# create agent
env = ConnectFourEnv()
best_agent = DQNAgent(env,best_buffer, batch = BATCH_SIZE, model_path = model_path_best)
adapting_agent = DQNAgent(env, adapting_buffer, batch = BATCH_SIZE, model_path = model_path_adapting, reward_function = reward_function_adapting_agent)
agents = [best_agent, adapting_agent]

train(agents, BATCH_SIZE, iterations, train_writer, test_writer, epsilon= EPSILON, epsilon_decay = 0.9)


print("done")
