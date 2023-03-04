def train(self,iterations : int, path_save_weights : str, path_save_logs : str):
        """ """

        for i in range(iterations):

            # epsilon decay

            # train the dqn + new sampels
            self.train_episode(i)

            # new sampling + add to buffer

            #buffer.extend()

            # write summary

            print("Iteration: ", i)