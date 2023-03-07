from sampler import Sampler

def train(agents, BATCH_SIZE, iterations : int, train_writer, test_writer, epsilon = 1, epsilon_decay = 0.9): # 
    # create Sampler 
    sampler = Sampler(BATCH_SIZE,agents)
    sampler.fill_buffers(epsilon)
    for i in range(iterations):
        
        # epsilon decay
        epsilon = epsilon * epsilon_decay

        # train the dqn + new samples
        if len(agents)==len(train_writer):
            [agent.train_inner_iteration(train_writer[j], i) for j, agent in enumerate(agents)]
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

        print("\n")
