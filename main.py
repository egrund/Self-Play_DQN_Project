from keras_gym_env import ConnectFourEnv
import numpy as np

from agent import DQNAgent
from buffer import Buffer
from sampler import Sampler
from training import train

# Hyperparameter
BATCH_SIZE = 4
reward_function_adapting_agent = lambda d,r: np.where(r==0,[1,0]) if d else r
EPSILON = 0.01 #TODO

# create environment
env = ConnectFourEnv()
actions = env.available_actions

# create buffer
best_buffer = Buffer(100000,1000)
adapting_buffer = Buffer(100000,1000)

# create agent
best_agent = DQNAgent(env,best_buffer)
adapting_agent = DQNAgent(env, adapting_buffer, reward_function = reward_function_adapting_agent)

# create Sampler
sampler = Sampler(BATCH_SIZE,[best_agent,adapting_agent])
sampler.fill_buffers(EPSILON)