import tensorflow as tf

class MyCNNNormalizationLayer(tf.keras.layers.Layer):
    """ a layer for a CNN with kernel size 3 and ReLu as the activation function """

    def __init__(self,filters,normalization=False, reg = None, kernel_size = 3):
        """ Constructor
        
        Parameters: 
            filters (int) = how many filters the Conv2D layer will have
            normalization (boolean) = whether the output of the layer should be normalized 
        """
        super(MyCNNNormalizationLayer, self).__init__()
        self.conv_layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', kernel_regularizer = reg)
        self.norm_layer = tf.keras.layers.BatchNormalization() if normalization else None
        self.activation = tf.keras.layers.Activation("relu")

    #@tf.function(reduce_retracing=True)
    def call(self,x,training=None):
        """ forward propagation """

        x = self.conv_layer(x)
        if self.norm_layer:
            x = self.norm_layer(x,training)
        x = self.activation(x)

        return x
    
class MyCNNBlock(tf.keras.layers.Layer):
    """ a block for a CNN having several convoluted layers with filters and kernel size 3 and ReLu as the activation function """

    def __init__(self,layers,filters,global_pool = False,mode = None,normalization = False, reg = None, dropout_layer = None):
        """ Constructor 
        
        Parameters: 
            layers (list) = how many Conv2D you want, and for each what kernel_size
            filters (int) = how many filters the Conv2D layers should have
            global_pool (boolean) = global average pooling at the end if True else MaxPooling2D
            mode (str) = whether we want to implement a denseNet or a resNet
            normalization (bool) = whether we want to use BatchNormalization
            reg (tensorflow Regularizer) = the Regularizer to use
            dropout_layer (tf.keras.layers.Dropout): The dropout layer to use
        """

        super(MyCNNBlock, self).__init__()
        self.dropout_layer = dropout_layer
        self.conv_layers =  [MyCNNNormalizationLayer(filters,normalization, reg, k) for k in layers]
        self.mode = mode
        switch_mode = {"dense":tf.keras.layers.Concatenate(axis=-1), "res": tf.keras.layers.Add(),}
        self.extra_layer = None if mode == None else switch_mode.get(mode,f"{mode} is not a valid mode for MyCNN. Choose from 'dense' or 'res'.")
        self.pool = tf.keras.layers.GlobalAvgPool2D() if global_pool else (tf.keras.layers.MaxPooling2D(pool_size=2, strides=2) if global_pool is not None else None)

    #@tf.function(reduce_retracing=True)
    def call(self,inputs,training=None):
        """ forward propagation of this block """
        x = inputs
        for i, layer in enumerate(self.conv_layers):
            x = layer(x, training)
            if(i==0 and self.mode == "res"): # for resnet add output of first layer to final output, not input of first layer
                inputs = x
            if self.dropout_layer:
                x = self.dropout_layer(x, training)
        if(self.extra_layer is not None):
            x = self.extra_layer([inputs, x])

        if self.pool is not None:
            x = self.pool(x)
        return x

class MyCNN_RL(tf.keras.Model):
    """ an ANN created to train on the mnist dataset """
    
    def __init__(self,conv_kernel : list = [3], filters : int = 128, hidden_units : list = [64],output_units : int = 10, 
                 hidden_activation = tf.nn.relu, output_activation = tf.nn.softmax, optimizer = tf.keras.optimizers.Adam(), 
                 loss = tf.keras.losses.CategoricalCrossentropy(), dropout_rate = 0.5, normalisation : bool = True, gamma : tf.constant = tf.constant(0.99)):
        """ Constructor 
        
        Parameters: 
            conv_kernel (list) = list containing one element for each conv layer in one block, values are the kernel size (each can be a 2D tuple or an int)
            filters (int) = the amount of filters in each layer of the conv block
            hidden_units (list) = list containing one element for each hidden layer and the values are the units of each layer 
            output_units (int) = the number of wanted output units
            hidden_activation (function)= the activation function for the hidden layers
            output_activation (function)= the activation fuction for the output layer
            optimizer (tf.keras.optimizers.Optimizer): The optimizer to use
            loss (tf.keras.losses.Loss): The loss to use
            dropout_rate (0<= int <1) = rate of dropout for after input and after dense
            normalisation(boolean) = if True use Batchnorm
            gamma (tf.constant) float = The discount factor for future rewards. 
        """

        super(MyCNN_RL, self).__init__()
        
        self.dropout_rate = dropout_rate
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate) if self.dropout_rate else None
        self.block = MyCNNBlock(layers = conv_kernel, filters = filters, global_pool=True, normalization = normalisation, dropout_layer = self.dropout_layer)
        self.dense_list = [tf.keras.layers.Dense(units, activation=hidden_activation) for units in hidden_units ]
        self.out = tf.keras.layers.Dense(output_units, activation=output_activation)

        self.optimizer = optimizer
        self.loss = loss
        self.output_units = output_units
        self.gamma = gamma

        self.metric = [tf.keras.metrics.Mean(name="loss")]

    def reset_metrics(self):
        """ resets all the metrices that are observed during training and testing """
        for m in self.metric:
            m.reset_states()

    #@tf.function(reduce_retracing=True)
    def call(self, states, training = False, intermediate = False):
        """ forward propagation of the ANN """
        x= self.block(states, training = training)
        inter = x # save intermediate result if we want to output it
        for layer in self.dense_list:
            x = layer(x)
        x = self.out(x)
        if intermediate:
            return x, inter
        return x

    #@tf.function(reduce_retracing=True)
    def train_step(self, inputs, agent):
        """ 
        one train step of the model

        inputs (list): state, action, reward, new_state, done (axis 0 is the batch dim)
        """

        # s,a,r,s_new, done = inputs
        s,a,r,s_new, done, a_action = inputs

        with tf.GradientTape() as tape: 
            
            # calculate the target Q value, only if not done
            # Qmax = tf.math.reduce_max(target_model(s_new),axis=1)
            # we do not want unavailable actions to be the best next action
            Qmax = agent.select_best_action_value(observations = s_new,available_actions_bool = a_action, unavailable = False)

            # calculate q value of this state action pair
            Qsa_estimated = tf.gather(self(s),indices = tf.cast(a,tf.int32),axis=1,batch_dims=1)
            # calculate the best Qsa which is the target
            target = r + self.gamma*(Qmax)*(1-done)

            losses = self.loss(Qsa_estimated, target)

            self.metric[0].update_state(losses)

        # get the gradients
        gradients = tape.gradient(losses,self.trainable_variables)

        # apply the gradient
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return{m.name : m.result() for m in self.metric}

        
class MyMLP_RL(tf.keras.Model):

    def __init__(self,hidden_units : list = [64],output_units : int = 10, 
                 hidden_activation = tf.nn.relu, output_activation = tf.nn.softmax, optimizer = tf.keras.optimizers.Adam(), 
                 loss = tf.keras.losses.MeanSquaredError(), gamma : tf.constant = tf.constant(0.99)):
        """ Constructor 
        
        Parameters: 
            hidden_units (list) = list containing one element for each hidden layer and the values are the units of each layer 
            output_units (int) = the number of wanted output units
            hidden_activation (function)= the activation function for the hidden layers
            output_activation (function)= the activation fuction for the output layer
            optimizer (tf.keras.optimizers.Optimizer)= the optimizer to use for training
            loss (tf.keras.losses.Loss)= the loss function to use for training
            gamma (tf.constant) = the discount factor to use for future rewards
        """

        super(MyMLP_RL, self).__init__()
        
        self.concat_layer = tf.keras.layers.Concatenate(axis=-1)
        self.dense_list1 = [tf.keras.layers.Dense(units, activation=hidden_activation) for units in hidden_units ]
        self.out1 = tf.keras.layers.Dense(1, activation=hidden_activation)

        self.dense_list = [tf.keras.layers.Dense(units, activation=hidden_activation) for units in hidden_units ]
        self.out = tf.keras.layers.Dense(output_units, activation=output_activation)

        self.optimizer = optimizer
        self.loss = loss
        self.output_units = output_units
        self.gamma = gamma

        self.metric = [tf.keras.metrics.Mean(name="loss")]

    def reset_metrics(self):
        """ resets all the metrices that are observed during training and testing """
        for m in self.metric:
            m.reset_states()

    #@tf.function(reduce_retracing=True)
    def call(self, states, available_actions_bool = None, training = False, agent = None, opponent_level = None, game_balance = None):
        """ forward propagation of the ANN """

        if agent == None:
            raise ValueError("agent cannot be None")
        if opponent_level == None:
            opponent_level = tf.expand_dims(tf.repeat(tf.constant(-1.,dtype=tf.float32),states.shape[0]), axis=-1) # -1 is basically no information, as the value given from the agent is limited by relu
        if game_balance == None:
            game_balance = tf.expand_dims(tf.repeat(tf.constant(0.,dtype=tf.float32),states.shape[0]), axis=-1)

        # get values from best agent about the game
        best_probs, inter = agent.best_agent.target_model(states,training = training, intermediate = True)
        if available_actions_bool != None:
            action_choice_best = agent.best_agent.select_action(states,None,None,True)# basically max value, includes unavailable actions
        else:
            action_choice_best = tf.expand_dims(tf.repeat(tf.constant(0.,dtype=tf.float32),states.shape[0]),axis=-1)

        # calculate the opponent level
        x = self.concat_layer((inter, opponent_level, game_balance))
        for layer in self.dense_list1:
            x = layer(x)
        new_opponent_level = self.out1(x)

        x = self.concat_layer((best_probs, inter, new_opponent_level, action_choice_best))

        for layer in self.dense_list:
            x = layer(x)
        x = self.out(x)
        return x, new_opponent_level

    #@tf.function(reduce_retracing=True)
    def train_step(self, inputs, agent, summary_writer = None, step = None):
        """ 
        one train step of the model

        inputs (list): state, action, reward, new_state, done, available_actions (after the action), game_balance, opponent_level 
        """

        # s,a,r,s_new, done = inputs
        s,a,r,s_new, done, a_action, game_balance, opponent_level = inputs

        with tf.GradientTape() as tape: 
            
            # calculate the target Q value, only if not done
            # we do not want unavailable actions to be the best next action
            Qbest = agent.select_adapting_action_value(observations = s_new,available_actions_bool = a_action, opponent_level = opponent_level, game_balance = game_balance, unavailable = tf.constant(False))

            # calculate q value of this state action pair
            Qsa_estimated = tf.gather(self(s, training = True, agent = agent, opponent_level = opponent_level, game_balance = game_balance)[0],indices = tf.cast(a,tf.int32),axis=1,batch_dims=1)
            
            # calculate the future change in game_balance
            new_game_balance = tf.add( game_balance, tf.divide(r, agent.max_balance_length))
            target = new_game_balance * (done) + self.gamma*(Qbest)*(1-done)

            losses = self.loss(Qsa_estimated, target)

            self.metric[0].update_state(losses)

        # get the gradients
        gradients = tape.gradient(losses,self.trainable_variables)

        # apply the gradient
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        if summary_writer != None:
            with summary_writer.as_default():
                tf.summary.scalar('prediction', tf.reduce_mean(Qsa_estimated), step=step)
                tf.summary.scalar('target', tf.reduce_mean(target), step=step)

        return{m.name : m.result() for m in self.metric}