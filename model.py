import tensorflow as tf

class MyMLP(tf.keras.Model):
    """ an ANN created to train on the mnist dataset """
    
    def __init__(self,hidden_units : list = [128,64,32],output_units : int = 10,hidden_activation = tf.nn.relu, output_activation = tf.nn.softmax, optimizer = tf.keras.optimizers.Adam(), loss = tf.keras.losses.CategoricalCrossentropy()):
        """ Constructor 
        
        Parameters: 
            hidden_units (list) = list containing one element for each hidden layer and the values are the units of each layer 
            output_units (int) = the number of wanted output units
            hidden_activation (function)= the activation function for the hidden layers
            output_activation (function)= the activation fuction for the output layer
        """

        super(MyMLP, self).__init__()
        self.dense_list = [ tf.keras.layers.Dense(units, activation=hidden_activation) for units in hidden_units ]
        self.out = tf.keras.layers.Dense(output_units, activation=output_activation)

        self.optimizer = optimizer
        self.loss = loss
        self.output_units = output_units
        self.model_target = None

        self.metric = [tf.keras.metrics.Mean(name="loss")]

    def reset_metrics(self):
        """ resets all the metrices that are observed during training and testing """
        for m in self.metric:
            m.reset_states()

    @tf.function
    def call(self, states):
        """ forward propagation of the ANN """
        x = tf.keras.layers.Flatten()(states)
        for layer in self.dense_list:
            x = layer(x)
        x = self.out(x)
        return x

    @tf.function
    def train_step(self, inputs):
        """ 
        one train step of the model

        inputs (list): state, action, reward, new_state, done (axis 0 is the batch dim)
        """

        if not self.model_target:
            raise Exception("This model does not have a target model.")

        s,a,r,s_new, done = inputs

        with tf.GradientTape() as tape: 
            
            # calculate the target Q value
            Qmax = tf.math.reduce_max(self.model_target(s_new),axis=1)
            # calculate q value of this state action pair
            Qsa = tf.gather(self(s),indices = tf.cast(a,tf.int32),axis=1,batch_dims=1)

            losses = self.loss(Qsa, r + (tf.constant(0.99)*Qmax)*(1-done))

            self.metric[0].update_state(value=losses)

        # get the gradients
        gradients = tape.gradient(losses,self.trainable_variables)

        # apply the gradient
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return{m.name : m.result() for m in self.metric}

        
