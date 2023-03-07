def train(self, agents, iterations : int, train_writer, test_writer, EPSILON = 1, EPSILON_DECAY = 0.9):
        """ """

    
    
    # create Sampler
    sampler = Sampler(BATCH_SIZE, agents)
    sampler.fill_buffers(EPSILON)
        
    for i in range(iterations):
        
        # epsilon decay
        EPSILON = EPSILON * EPSILON_DECAY

        # train the dqn + new samples
        if len(agents)==len(train_summary_writer):
            [agent.train_inner_iteration(self, train_summary_writer[j], i) for j, agent in enumerate(agents)]
        else:
            raise IndexError("You need the same amount of summary writers and agents.")

        # new sampling + add to buffer
        sampler.sample_from_game(epsilon)

        # write summary
        # create directory for logs
            
        # logging the validation metrics to the log file which is used by tensorboard
        #with train_summary_writer.as_default():
        #    for metric in model.metrics:
        #        tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)

        # reset all metrics
        model.reset_metrics()
        print("\n")

        # save model
        if iterations%10 == 0:
            model.save_weights(f"model/{config_name}/{epoch}")
