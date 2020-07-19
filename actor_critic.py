"""Actor and Critic for DDPG"""

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xav_init, xavier_initializer_conv2d as xav_conv_init
from tensorflow.contrib.layers import l2_regularizer
from utils import unif_initializer, collection_add, clip_grads

class Actor:
    """Actor for DDPG.

    Attributes:
        sess (): Tensorflow session.
        lr (float): Learning rate.
        tau (float): Parameter for soft update.
        batch_size (int): Size of batch for gradient descent.
        clip_val (float): Value to clip all gradients.
        s_dim (array): Array containing input dimensions to neural networks.
        a_dim (array): Array containing input dimensions to neural networks.
        input (): Input to network.
        output (): Output of network.
        target_input (): Input to target network.
        target_output (): Output of target network.
        vars (): List of network parameters.
        target_vars (): List of target network parameters.
        action_gradients (): Placeholder for action gradients.
    """

    def __init__(self, sess, lr, tau, batch_size, clip_val, s_dim, a_dim):
        """Initialize DDPG actor.

        Args:
            sess (): Tensorflow session.
            lr (float): Learning rate.
            tau (float): Parameter for soft update.
            batch_size (int): Size of batch for gradient descent.
            clip_val (float): Value to clip all gradients.
            s_dim (list): Array containing input dimensions to neural networks.
            a_dim (list): Array containing input dimensions to neural networks.
        """
        self.sess = sess
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size
        self.clip_val = clip_val
        self.s_dim = s_dim
        self.a_dim = a_dim

        # networks
        with tf.variable_scope('actor_network'):
            self.input, self.output = self._set_network()
            collection_add([self.input, self.output])
        with tf.variable_scope('actor_target_network'):
            self.target_input, self.target_output = self._set_network()
            collection_add([self.target_input, self.target_output])

        # variables
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_network')
        self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_target_network')

        # copy vars
        self._copy_vars = [self.target_vars[i].assign(self.vars[i]) for i in range(len(self.target_vars))]

        # update vars
        self._update_vars = [self.target_vars[i].assign(tf.multiply(self.vars[i], self.tau) +
                                                       tf.multiply(self.target_vars[i], 1. - self.tau))
                            for i in range(len(self.target_vars))]

        # optimizer
        self.action_gradients = tf.placeholder(shape=a_dim, dtype=tf.float32)
        self.grads = tf.gradients(self.output, self.vars, -self.action_gradients)
        self.normalized_grads = list(map(lambda x: tf.div(x, self.batch_size), self.grads))
        self.clipped_grads = clip_grads(zip(self.normalized_grads, self.vars), self.clip_val)
        self.optimize = tf.train.AdamOptimizer(learning_rate=self.lr).apply_gradients(self.clipped_grads)

    def _set_network(self):
        """Set up network.

        Args:
            void

        Returns:
            input (): Placeholder for network input.
            output (): Output operation.
        """

        # input
        input = tf.placeholder(shape=self.s_dim, dtype=tf.float32)

        # initializers for fc layers
        unif_init = unif_initializer(-0.003, 0.003)

        # flatten
        net = tf.contrib.layers.flatten(input)

        # fc layers
        net = tf.layers.dense(net, 400, kernel_initializer=xav_init())
        net = tf.nn.relu(net)
        net = tf.layers.dense(net, 300, kernel_initializer=xav_init())
        net = tf.nn.relu(net)
        net = tf.layers.dense(net, self.a_dim[1], kernel_initializer=unif_init)
        output = tf.nn.tanh(net)

        return input, output

    def train(self, input, action_gradients):
        """Do a training run on the actor network.

        Args:
            input (): Input to feed placeholder.
            action_gradients (): Input to feed placeholder.

        Returns:
            void
        """
        self.sess.run(self.optimize, feed_dict={self.input: input, self.action_gradients: action_gradients})

    def predict(self, input):
        """Perform forward run with actor.

        Args:
            input (): Input to placeholder.

        Returns:
            - (): Action prediction.
        """
        return self.sess.run(self.output, feed_dict={self.input: input})

    def predict_target(self, input):
        """Perform forward run with target actor.

        Args:
            input (): Input to placeholder.

        Returns:
            - (): Action prediction.
        """
        return self.sess.run(self.target_output, feed_dict={self.target_input: input})

    def copy_vars(self):
        """Copy parameters from the network to the target network.

        Args:
            void

        Returns:
            void
        """
        self.sess.run(self._copy_vars)

    def update_vars(self):
        """Soft update parameters from the network to the target network.

        Args:
            void

        Returns:
            void
        """
        self.sess.run(self._update_vars)


class Critic:
    """Critic for DDPG.

    Attributes:
        sess (): Tensorflow session.
        lr (float): Learning rate.
        tau (float): Parameter for soft update.
        clip_val (float): Value to clip all gradients.
        s_dim (array): Array containing input dimensions to neural networks.
        a_dim (array): Array containing input dimensions to neural networks.
        input (): Input to network.
        output (): Output of network.
        target_input (): Input to target network.
        target_output (): Output of target network.
        vars (): List of network parameters.
        target_vars (): List of target network parameters.
    """
    def __init__(self, sess, lr, tau, clip_val, s_dim, a_dim):
        """Initialize DDPG actor.

        Args:
            sess (): Tensorflow session.
            lr (float): Learning rate.
            tau (float): Parameter for soft update.
            clip_val (float): Value to clip all gradients.
            s_dim (list): Array containing input dimensions to neural networks.
            a_dim (list): Array containing input dimensions to neural networks.
        """

        # session
        self.sess = sess
        self.lr = lr
        self.tau = tau
        self.clip_val = clip_val
        self.s_dim = s_dim
        self.a_dim = a_dim

        # networks
        with tf.variable_scope('critic_network'):
            self.input, self.actions, self.output = self._set_network()
            collection_add([self.input, self.output])
        with tf.variable_scope('critic_target_network'):
            self.target_input, self.target_actions, self.target_output = self._set_network()
            collection_add([self.target_input, self.target_output])

        # variables
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_network')
        self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_target_network')

        # copy vars
        self._copy_vars = [self.target_vars[i].assign(self.vars[i]) for i in range(len(self.target_vars))]

        # update vars
        self._update_vars = [self.target_vars[i].assign(tf.multiply(self.vars[i], self.tau) +
                                                       tf.multiply(self.target_vars[i], 1. - self.tau))
                            for i in range(len(self.target_vars))]

        # optimizer
        self.q_target = tf.placeholder(shape=None, dtype=tf.float32)
        self.loss = tf.losses.mean_squared_error(labels=self.q_target, predictions=self.output)
        self.grads = tf.gradients(self.loss, self.vars)
        self.clipped_grads = clip_grads(zip(self.grads, self.vars), self.clip_val)
        self.optimize = tf.train.AdamOptimizer(learning_rate=self.lr).apply_gradients(self.clipped_grads)

        # action gradients
        self.action_gradients = tf.gradients(self.output, self.actions)

    def _set_network(self):
        """Set up network.

        Args:
            void

        Returns:
            input (): Placeholder for network input.
            actions (): Placeholder for action inputs.
            output (): Output operation.
        """

        # regularizer
        regularizer = l2_regularizer(scale=0.01)

        # inputs
        input = tf.placeholder(shape=self.s_dim, dtype=tf.float32)
        actions = tf.placeholder(shape=self.a_dim, dtype=tf.float32)

        # initializers for fc layers
        unif_init = unif_initializer(-0.003, 0.003)

        # flatten
        net = tf.contrib.layers.flatten(input)

        # fc layers
        net = tf.layers.dense(net, 400, kernel_initializer=xav_init(),
                              kernel_regularizer=regularizer)
        net = tf.nn.relu(net)
        net = tf.layers.dense(net, 300, kernel_initializer=xav_init(),
                              kernel_regularizer=regularizer, use_bias=False)
        action_net = tf.layers.dense(actions, 300, kernel_initializer=xav_init(),
                              kernel_regularizer=regularizer, use_bias=True)
        net += action_net
        net = tf.nn.relu(net)
        output = tf.layers.dense(net, 1, kernel_initializer=unif_init, kernel_regularizer=regularizer)

        return input, actions, output

    def train(self, input, actions, q_target):
        """Do a training run on the critic network.

        Args:
            input (): Input to feed placeholder.
            actions (): Input to feed placeholder.
            q_target (): Target for loss.

        Returns:
            void
        """
        self.sess.run(self.optimize, feed_dict={self.input: input, self.actions: actions, self.q_target: q_target})

    def predict(self, input, actions):
        """Perform forward run with target critic.

        Args:
            input (): Input to placeholder.
            actions (): Input to placeholder.

        Returns:
            - (): Output of the target network.
        """
        return self.sess.run(self.target_output, feed_dict={self.target_input: input, self.target_actions: actions})

    def get_gradients(self, input, actions):
        """Get the gradients of the outputs wrt. inputs actions.

        Args:
            input (): Input to placeholder
            actions (): Action inputs to placeholder.

        Returns:
            - (): Actions gradients.
        """
        return self.sess.run(self.action_gradients, feed_dict={self.input: input, self.actions: actions})[0]

    def copy_vars(self):
        """Copy parameters from the network to the target network.

        Args:
            void

        Returns:
            void
        """
        self.sess.run(self._copy_vars)

    def update_vars(self):
        """Soft update parameters from the network to the target network.

        Args:
            void

        Returns:
            void
        """
        self.sess.run(self._update_vars)
