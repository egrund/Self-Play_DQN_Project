import tensorflow as tf

class MyModel(tf.keras.Model):
    """ an ANN created to train on the mnist dataset """
    
    def __init__(self,hidden_units : list,output_units : int = 10,hidden_activation = tf.nn.relu):
        """ Constructor 
        
        Parameters: 
            hidden_units (list) = list containing one element for each hidden layer and the values are the units of each layer 
            output_units (int) = the number of wanted output units
            hidden_activation = the activation function for the hidden layers
        """

        super(MyModel, self).__init__()
        self.dense_list = [ tf.keras.layers.Dense(units, activation=hidden_activation) for units in hidden_units ]
        self.out = tf.keras.layers.Dense(output_units, activation=tf.nn.softmax)

        self.metrics = [tf.keras.metrics.Mean(name="loss"), tf.keras.metrics.CategoricalAccuracy(name="accuracy")]

    def reset_metrics(self):
        """ resets all the metrices that are observed during training and testing """
        for m in self.metrics:
            m.reset_states()

    @tf.function
    def call(self, inputs):
        """ forward propagation of the ANN """
        x = inputs
        for layer in self.dense_list:
            x = layer(x)
        x = self.out(x)
        return x

    @tf.function
    def train_step(self, inputs):

        images, targets = inputs

        with tf.GradientTape() as tape: 

            predictions = model(images)
            loss = loss_function(targets, predictions)

            self.metrics[0].update_state(value=loss)
            self.metrics[1].update_state(predictions, targets)

        # get the gradients
        gradients = tape.gradient(loss,model.trainable_variables)

        # apply the gradient
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return{m.name : m.result() for m in self.metrics}

    @tf.function
    def test_step(self,data):
        """ does one test step in one episode given a batch of data 
        
        Parameters: 
            data (tuple) = shape (img, target)
        
        returns a dictionary of the metrics
        """

        img, targets = data

        predictions = self(img,training=False)
        loss = self.loss_function(targets, tf.squeeze(predictions))

        self.metrics[0].update_state(values = loss) # loss
        self.metrics[1].update_state(predictions, targets) # accuracy

        return{m.name : m.result() for m in self.metrics}
