""""
This is part of code is the deep-Q-network brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.
action: [0,1,2,3] stands for handoff to ap1, ap2, ap3, ap4, respectively
Using:
Tensorflow: 1.0
"""
from collections import deque
import numpy as np
#import pandas as pd
import tensorflow as tf
import pickle
import os

#tf.set_random_seed(5)

OBSERVE = 100.  # timesteps to observe before training
EXPLORE = 200000.  # frames over which to anneal epsilon
INITIAL_EPSILON=0.80
FINAL_EPSILON=0.80
# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.0002,
            reward_decay=0.6,
            e_greedy=INITIAL_EPSILON,
            replace_target_iter=10,
            memory_size=5000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            param_file= None,
    ):
        self.timestep = 0
        self.param_file = param_file
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = e_greedy
        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = self.loadReplayMemory()

        # consist of [target_net, evaluate_net]
        self._build_net()
        self.t_params = tf.get_collection('target_net_params')
        self.e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        self.saver = tf.train.Saver() 
        self.sess = tf.Session()
        #if param_file != None:
        #    self.saver.restore(self.sess,"./eval_network/popseed12_17100.ckpt")
        #    print "loading nerou-network params..."

        if output_graph:
            # $ tensorboard --logdir=logs/0.0.0.0:6006/graphs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
        if param_file != None:
            self.saver.restore(self.sess,"./eval_network/rl13v1_3000.ckpt")
            print "loading nerou-network params..."

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input state  None: any samples
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss ,two input params

        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, n_l2, n_l3, n_l4, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 50, 80, 60, 30, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
            # third layer. collections is used later when assign to target net
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, n_l3], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, n_l3], initializer=b_initializer, collections=c_names)
                l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)
            # forth layer. collections is used later when assign to target net          
            with tf.variable_scope('l4'):
                w4 = tf.get_variable('w4', [n_l3, n_l4], initializer=w_initializer, collections=c_names)
                b4 = tf.get_variable('b4', [1, n_l4], initializer=b_initializer, collections=c_names)
                l4 = tf.nn.relu(tf.matmul(l3, w4) + b4)
            # fifth layer. collections is used later when assign to target net
            with tf.variable_scope('l5'):
                w5 = tf.get_variable('w5', [n_l4, self.n_actions], initializer=w_initializer, collections=c_names)
                b5 = tf.get_variable('b5', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l4, w5) + b5

                
            

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
            # third layer. collections is used later when assign to target net
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, n_l3], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, n_l3], initializer=b_initializer, collections=c_names)
                l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)
            # forth layer. collections is used later when assign to target net
            with tf.variable_scope('l4'):
                w4 = tf.get_variable('w4', [n_l3, n_l4], initializer=w_initializer, collections=c_names)
                b4 = tf.get_variable('b4', [1, n_l4], initializer=b_initializer, collections=c_names)
                l4 = tf.nn.relu(tf.matmul(l3, w4) + b4)
            # fifth layer. collections is used later when assign to target net
            with tf.variable_scope('l5'):
                w5 = tf.get_variable('w5', [n_l4, self.n_actions], initializer=w_initializer, collections=c_names)
                b5 = tf.get_variable('b5', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l4, w5) + b5


    def setPerception(self, s, a, r, s_):
        """
        store transition and when memory_counter reach OBSERVE , start to learning
        """
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        if s.tolist().count(0) < 2:
            transition = np.hstack((s, [a, r], s_))
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

        if self.timestep > OBSERVE:
            # Train the network
            self.learn()

        #print current state...
        state = ""
        if self.timestep <= OBSERVE:
            state = "observe"
        elif self.timestep > OBSERVE and self.timestep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print "timestep:", self.timestep, "/ state:", state, \
            "/ e-greedy: ", self.epsilon,"\n"
        self.timestep += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})

        if np.random.uniform() < self.epsilon:
            if observation[0].tolist().count(0) != 0:
                dictstr = []
                for index,value in enumerate(observation[0]):
                    if value != 0:
                        dictstr.append(index)
                MAX_V = actions_value[0][dictstr[0]]
                for item in dictstr:
                    if MAX_V < actions_value[0][item]:
                        MAX_V = actions_value[0][item]

                action = actions_value[0].tolist().index(MAX_V)
            else:
                action = np.argmax(actions_value)
        else:

            print "choose random action....."
            if observation[0].tolist().count(0) == 3:
                action = -1

            elif observation[0].tolist().count(0) == 2:
                for index,value in enumerate(observation[0]):
                    if value != 0:
                        action = index
            else:
                while True:
                    action = np.random.randint(0, self.n_actions)
                    if observation[0][action] !=0 and action != np.argmax(actions_value):
                        break

        return action, actions_value

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.timestep%500 == 0:
            self.saver.save(self.sess,"./eval_network/rl13v2_%04d.ckpt"%(self.timestep))

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                             self.q_target: q_target,})


        self.cost_his.append(self.cost)

        # increasing epsilon
        if self.epsilon <= FINAL_EPSILON and self.timestep > OBSERVE: 
            self.epsilon = self.epsilon + (FINAL_EPSILON-INITIAL_EPSILON)/EXPLORE
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def predict(self, observation):
        """

        :param observation:  next obse
        :return:
        """
        actions_value = self.sess.run(self.q_target, feed_dict={self.s: observation})
        #action = np.zeros(self.n_actions)
        action = np.argmax(actions_value)
        #action[action_index] = 1

        return action, actions_value

    def saveReplayMemory(self):
        print 'Memory Size: ' + str(len(self.memory))
        with open('./network_params/replayMemory.pkl', 'wb') as handle:
            pickle.dump(self.memory, handle, -1)  # Using the highest protocol available
        pass

    def loadReplayMemory(self):
        if os.path.exists('./network_params/replayMemory.pkl'):
            with open('./network_params/replayMemory.pkl', 'rb' ) as handle:
                replayMemory = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
                print "loading previous memories..."
        else:
            replayMemory = np.zeros((self.memory_size, self.n_features * 2 + 2))
        return replayMemory

