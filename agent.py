from model import MyMLP
import tensorflow as tf

class DQNAgent:

    def __init__(self, env):

        # create an initialize model and target_model
        self.model = MyMLP(hidden_units = [128,64,32], output_units = len(env.available_actions))
        self.target_model = MyMLP(hidden_units = [128,64,32], output_units = len(env.available_actions))
        self.model(tf.random.uniform(shape=()))
