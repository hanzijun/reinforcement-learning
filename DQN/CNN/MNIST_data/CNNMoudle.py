"""
This part code is the Recurrent Nrural Network + DQN brain, which is a brain of the mobile station.
All decisions are made in here. Using Tensorflow to build the neural network.
Using:
Tensorflow: 1.0
"""""


import numpy as np
import tensorflow as tf
from collections import deque
import pickle
import os
import random


class RNNNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            param_file=None
    ):
        self.seqlen = 32
        self.num_layers = 3
        self.n_hidden_units = 10
        self.param_file = param_file
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0
        self.memory = self.loadReplayMemory()

        # consist of [target_net, evaluate_net]
        self.sess = tf.Session()
        self.buildCNNNetwork()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.saver = tf.train.Saver()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
        if self.param_file is not None:
            self.saver.restore(self.sess, "./eval_network/rlnew.ckpt")
            print  ('loading previous neural network params')

    def buildCNNNetwork(self):
        '''
        :param:  CNN networks consists of input layer (l1), multiple RNN cells (RNN cell) and output layer (l2). 
        :param:  collections (c_names) are the collections to store variables.
        :param:  state divided into two parts: the main line  and  sub-line.
        :param: outputs is a list that consists of all the result calculated in every step. 
        return: bulit cNN networks
        RNN network includes two parts as well, one is eval_network used to learn and update params
        another is target_network which is used to convert "nextstate" to "q_target", the loss of "q_target" 
        and "q_eval" updates the actions_value.  
        The main format is : 
        Q(s,a, t_) = (1-lanmda) * Q(s,a,t) + lambda(reward +gama * max Q(s_,a,t)       
        '''''

        # +++++++++++++++++++build eval_network+++++++++++++++++++++
        self.s = tf.placeholder(tf.float32, [None,  self.seqlen,  self.n_features], name='s')
        self.s_image = tf.reshape(self.s, [-1,self.seqlen,self.n_features,1])
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')

        with tf.variable_scope('eval_net'):
            c_names, n_l1 = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], self.n_hidden_units
            w_initializer = tf.random_normal_initializer(0., 0.3)
            b_initializer = tf.constant_initializer(0.1)

            with tf.variable_scope('conv_l1'):
                w_conv1 = tf.get_variable('w1',  [5,5,1,4], initializer=w_initializer,
                                          collections=c_names)
                b_conv1  = tf.get_variable('b1',  [4],initializer=b_initializer, collections=c_names)
                h_conv1 = tf.nn.relu(self.conv2d(self.s_image, w_conv1) + b_conv1)
                print (h_conv1.shape)
                h_pool1 = self.max_pool_2x2(h_conv1)
                print (h_pool1.shape)

            with tf.variable_scope("conv_l2"):
                w_conv2 = tf.get_variable('w2', [3, 3, 4, 8], initializer=w_initializer,
                                          collections=c_names)
                b_conv2 = tf.get_variable('b2', [8],initializer=b_initializer, collections=c_names)
                h_conv2 = tf.nn.relu(self.conv2d(h_pool1, w_conv2) + b_conv2)
                print (h_conv2.shape)
                h_pool2 = self.max_pool_2x2(h_conv2)
                print (h_pool2.shape)
                h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 3 * 8])
                print (h_pool2_flat.shape)
            with tf.variable_scope('full_connected_layer3'):
                w_fc1 = tf.get_variable('w3', [8*3*8, self.n_actions], initializer=w_initializer,
                                        collections=c_names)
                b_fc1 = tf.get_variable('b3',[self.n_actions],initializer=b_initializer, collections=c_names)
                result = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
                self.q_eval = result
                # self.q_eval = tf.nn.relu(self.q_eval)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # +++++++++++++++++++++build target_net +++++++++++++++++++++
        self.s_ = tf.placeholder(tf.float32, [None, self.seqlen,  self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('conv_l1'):
                X2 = tf.reshape(self.s_, [-1, self.seqlen,self.n_features,1])
                w_conv1 = tf.get_variable('w1',[5, 5, 1, 4], initializer=w_initializer,
                                          collections=c_names)
                b_conv1 = tf.get_variable('b1', [4],initializer=b_initializer, collections=c_names)
                h_conv1 = tf.nn.relu(self.conv2d(X2, w_conv1) + b_conv1)
                h_pool1 = self.max_pool_2x2(h_conv1)

            with tf.variable_scope("conv_l2"):
                w_conv2 = tf.get_variable('w2', [3, 3, 4, 8], initializer=w_initializer, collections=c_names)
                b_conv2 = tf.get_variable('b2',[8],initializer=b_initializer, collections=c_names)
                h_conv2 = tf.nn.relu(self.conv2d(h_pool1, w_conv2) + b_conv2)
                h_pool2 = self.max_pool_2x2(h_conv2)
                h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 3 * 8])

            with tf.variable_scope('full_connected_layer3'):
                w_fc1 = tf.get_variable('w3',[8 * 3 * 8, self.n_actions], initializer=w_initializer,
                                        collections=c_names)
                b_fc1 = tf.get_variable('b3', [self.n_actions],initializer=b_initializer, collections=c_names)
                result = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
                self.q_next = result
                # self.q_next = tf.nn.relu(self.q_next)

    def conv2d(self,x, W):
        # stride [1, x_movement, y_movement, 1]
        # Must have strides[0] = strides[3] = 1
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self,x):
        # stride [1, x_movement, y_movement, 1]
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    def store_transition(self, s, a, r, s_):
        """
        store the current memories in order to restore.
        :param s:  state
        :param a:  action
        :param r:  reward
        :param s_:  nextstate
        :return:  saved memories
        """
        self.memory.append((s,a,r,s_))
        if len(self.memory)> self.memory_size:
            self.memory.popleft()

    def choose_action(self, observation):
        ''''
         to have batch dimension when feed into tf placeholder
        choose actionID to adaptive various environments
        :param observation:  current state
        :return:  action ID
        '''
        observation = observation[np.newaxis, :]
        observation = observation.reshape([1, self.seqlen, self.n_features])
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        '''
        noteworthy: inputs should be converted  from "2D" to "3D" which  RNN needs
        :return:  updated params in DRQN neural networks
        '''
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory

        batch_memory = random.sample(self.memory,self.batch_size)

        nextstateList = np.squeeze([data[3] for data in batch_memory])

        stateList = np.squeeze([data[0] for data in batch_memory])

        nextstateList = nextstateList.reshape([self.batch_size, self.seqlen, self.n_features])

        stateList = stateList.reshape([self.batch_size, self.seqlen, self.n_features])
        # s_list = [i.reshape([1,1,2]) for i in batch_memory[:, -self.n_features:]]
        # slist = [i.reshape([1,1,2]) for i in batch_memory[:, :self.n_features:]]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                # self.s_: batch_memory[:, -self.n_features:],  # fixed params
                # self.s: batch_memory[:, :self.n_features]  # newest params
                self.s_: nextstateList,
                self.s: stateList
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # eval_act_index = batch_memory[:, self.n_features].astype(int)
        # reward = batch_memory[:, self.n_features + 1]
        eval_act_index = np.squeeze([data[1] for data in batch_memory]).astype(int)
        reward = np.squeeze([data[2] for data in batch_memory])

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: stateList,
                                             self.q_target: q_target,})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def saveReplayMemory(self):
        print ('Memory Size: ' + str(len(self.memory)))
        with open('./eval_network/replayMemory.pkl', 'wb') as handle:
            pickle.dump(self.memory, handle, -1)  # Using the highest protocol available
        pass

    def loadReplayMemory(self):
        if os.path.exists('./eval_network/replayMemory.pkl'):
            with open('./eval_network/replayMemory.pkl', 'rb') as handle:
                replayMemory = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
        else:
            replayMemory = deque()
        return replayMemory


if __name__ == "__main__":
    RNNNetwork(9,9)
