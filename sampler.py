from keras_gym_env import ConnectFourEnv

class Sampler:
    """ 
    
    """

    def __init__(self,batch,agent):

        self.envs = [ConnectFourEnv() for _ in range(batch)]
        self.agent = agent

    def sample_from_game(self):
        """ """
        sarsa = list()

        action_space_size = self.envs[0].action_space.n
        observations = [env.reset() for env in self.envs]

        for _ in range(10000):

            actions = self.agent