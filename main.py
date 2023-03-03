from keras_gym_env import ConnectFourEnv

from agent import DQNAgent
from buffer import Buffer
from sampler import Sampler

# Hyperparameter
BATCH_SIZE = 16

# create environment
env = ConnectFourEnv()
actions = env.available_actions

# create buffer
buffer = Buffer()

# create agent
agent = DQNAgent(env,buffer)

# create Sampler
sampler = Sampler(BATCH_SIZE,agent)
buffer.fill()