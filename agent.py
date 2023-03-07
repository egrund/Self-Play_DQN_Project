def train_inner_iteration(self, summary_writer, i):
        """ """
        for j in range(self.inner_iterations):

            # sample random minibatch of transitions
            minibatch = self.buffer.sample_minibatch()

            # train model
                
            training_data = 
            
            #loss = self.dqn.step(s,a,r,s_new,done, self.optimizer, self.dqn_target)


            # if prioritized experience replay, then here


        # polyak averaging
        self.target_model.set_weights((1-self.polyak_update)*np.array(self.target_model.get_weights(),dtype = object) + 
                                      self.polyak_update*np.array(self.model.get_weights(),dtype = object))

        # logs
        if train_summary_writer:
            with train_summary_writer.as_default():
                tf.summary.scalar(m.name, loss, step=j+i*self.inner_iterations)
        if test_summary_writer:
            with test_summary_writer.as_default():
                tf.summary.scalar(m.name, loss, step=j+i*self.inner_iterations)
                
    
