from design.SingletonPattern import SingletonInstance
#from tensorflow import keras
import tensorflow as tf

class RNN(SingletonInstance):
    def __init__(self, input_size, timestep, output_size):
        self.input_size = input_size
        self.timestep = timestep
        self.output_size = output_size
        
        build_network()

    def build_network(self, hd_size, l_rate=1e-2):
        self._X = tf.placeholder(tf.float32, [None, self.input_size, self.timestep])
        self._Y = tf.placeholder(tf.float32, [None, self.output_size])

        #for multi layered cell
        cell = []
        for i in range(24):
            cell.append(tf.contrib.rnn.LSTMCell(hd_size, state_is_tuple=True, initializer=tf.contrib.layers.xavier_initializer()))
        cell = tf.contrib.rnn.MultiRNNCell(cell,state_is_tuple=True)

        outputs, _states = tf.nn.dynamic_rnn(cell, self._X, dtype=tf.float32)
        self.Y_pred = tf.con

