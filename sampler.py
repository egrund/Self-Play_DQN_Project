

class Sampler:
    """ 
    
    """

    def __init__(self,env,model):

        self.env = env
        self.model = model

    def sample_from_game(self):
        """ """
        sarsa = list()

        action_space_size = self.env.action_space.n
        observations = [self.env.reset()]

        for _ in range(10000):

            