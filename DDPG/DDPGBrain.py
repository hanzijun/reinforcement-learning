"""
This part code is Deep Deterministic Policy Gradient Brain.
Using tensorflow to build neural networks
Version : tensorflow1.0
"""


from collections import deque
import tensorflow as tf
import numpy as np
import pickle
import random
import os


class DDPG(object):
    def __init__(
            self,
            n_actions,
            n_features,
            bound_actions = 1,
            learning_rate_A=0.01,
            learning_rate_C = 0.02,
            reward_decay=0.9,
            e_greedy=0.9,
            soft_replacement=0.01,
            memory_size=500,
            batch_size=32,
            output_graph=False,
            param_file=None
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.bound_actions = bound_actions
        self.lrA = learning_rate_A
        self.lrC = learning_rate_C
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.a_replace_counter, self.c_replace_counter = 0, 0
        self.soft_replacement = soft_replacement
        self.replace_target_iter = 300
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.param_file = param_file

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        #self.memory = self.loadReplayMemory()
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self.sess = tf.Session()
        self.buildNetwork()
        self.saver = tf.train.Saver()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
        if self.param_file is not None:
            self.saver.restore(self.sess, "./eval_network/rlnew.ckpt")
            print  'loading previous neural network params'

    def buildNetwork(self):
        self.state = tf.placeholder(tf.float32, [None, self.n_features], 's')
        self.nextstate = tf.placeholder(tf.float32, [None, self.n_features], 's_')
        self.reward = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self.buildActorNetwork(self.state, scope='eval', trainable=True)
            a_ = self.buildActorNetwork(self.nextstate, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self.buildCriticNetwork(self.state, self.a, scope='eval', trainable=True)
            q_ = self.buildCriticNetwork(self.nextstate, a_, scope='target', trainable=False)

            # networks parameters
            self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
            self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
            self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
            self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

            # target net replacement
            self.soft_replace = [[tf.assign(ta, (1 - self.soft_replacement) * ta + self.soft_replacement * ea),
                                  tf.assign(tc, (1 - self.soft_replacement) * tc + self.soft_replacement * ec)]
                                 for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

            q_target = self.reward + self.gamma * q_
            # in the feed_dic for the td_error, the self.a should change to actions in memory
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(self.lrC).minimize(td_error, var_list=self.ce_params)

            a_loss = - tf.reduce_mean(q)  # maximize the q
            self.atrain = tf.train.AdamOptimizer(self.lrA).minimize(a_loss, var_list=self.ae_params)

    def buildActorNetwork(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.n_actions, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.bound_actions, name='scaled_a')

    def buildCriticNetwork(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.n_features, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.n_actions, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, s):
        return self.sess.run(self.a, {self.state: s[np.newaxis, :]})[0]

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.soft_replace)
            print('\ntarget_params_replaced\n')

        indices = np.random.choice(self.memory_size, size=self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.n_features]
        ba = bt[:, self.n_features: self.n_features + self.n_actions]
        br = bt[:, -self.n_features - 1: -self.n_features]
        bs_ = bt[:, -self.n_features:]

        self.sess.run(self.atrain, {self.state: bs})
        self.sess.run(self.ctrain, {self.state: bs, self.a: ba, self.reward: br, self.nextstate: bs_})