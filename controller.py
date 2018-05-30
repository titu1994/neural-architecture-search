import numpy as np
import time
import pprint
from collections import OrderedDict

from keras import backend as K
import tensorflow as tf

import os
if not os.path.exists('weights/'):
    os.makedirs('weights/')


class StateSpace:
    '''
    State Space manager

    Provides utilit functions for holding "states" / "actions" that the controller
    must use to train and predict.

    Also provides a more convenient way to define the search space
    '''
    def __init__(self):
        self.states = OrderedDict()
        self.state_count_ = 0

    def add_state(self, name, values):
        '''
        Adds a "state" to the state manager, along with some metadata for efficient
        packing and unpacking of information required by the RNN Controller.

        Stores metadata such as:
        -   Global ID
        -   Name
        -   Valid Values
        -   Number of valid values possible
        -   Map from value ID to state value
        -   Map from state value to value ID

        Args:
            name: name of the state / action
            values: valid values that this state can take

        Returns:
            Global ID of the state. Can be used to refer to this state later.
        '''
        index_map = {}
        for i, val in enumerate(values):
            index_map[i] = val

        value_map = {}
        for i, val in enumerate(values):
            value_map[val] = i

        metadata = {
            'id': self.state_count_,
            'name': name,
            'values': values,
            'size': len(values),
            'index_map_': index_map,
            'value_map_': value_map,
        }
        self.states[self.state_count_] = metadata
        self.state_count_ += 1

        return self.state_count_ - 1

    def embedding_encode(self, id, value):
        '''
        Embedding index encode the specific state value

        Args:
            id: global id of the state
            value: state value

        Returns:
            embedding encoded representation of the state value
        '''
        state = self[id]
        size = state['size']
        value_map = state['value_map_']
        value_idx = value_map[value]

        one_hot = np.zeros((1, size), dtype=np.float32)
        one_hot[np.arange(1), value_idx] = value_idx + 1
        return one_hot

    def get_state_value(self, id, index):
        '''
        Retrieves the state value from the state value ID

        Args:
            id: global id of the state
            index: index of the state value (usually from argmax)

        Returns:
            The actual state value at given value index
        '''
        state = self[id]
        index_map = state['index_map_']

        if (type(index) == list or type(index) == np.ndarray) and len(index) == 1:
            index = index[0]

        value = index_map[index]
        return value

    def get_random_state_space(self, num_layers):
        '''
        Constructs a random initial state space for feeding as an initial value
        to the Controller RNN

        Args:
            num_layers: number of layers to duplicate the search space

        Returns:
            A list of one hot encoded states
        '''
        states = []

        for id in range(self.size * num_layers):
            state = self[id]
            size = state['size']

            sample = np.random.choice(size, size=1)
            sample = state['index_map_'][sample[0]]
            state = self.embedding_encode(id, sample)
            states.append(state)
        return states

    def parse_state_space_list(self, state_list):
        '''
        Parses a list of one hot encoded states to retrieve a list of state values

        Args:
            state_list: list of one hot encoded states

        Returns:
            list of state values
        '''
        state_values = []
        for id, state_one_hot in enumerate(state_list):
            state_val_idx = np.argmax(state_one_hot, axis=-1)[0]
            value = self.get_state_value(id, state_val_idx)
            state_values.append(value)

        return state_values

    def print_state_space(self):
        ''' Pretty print the state space '''
        print('*' * 40, 'STATE SPACE', '*' * 40)

        pp = pprint.PrettyPrinter(indent=2, width=100)
        for id, state in self.states.items():
            pp.pprint(state)
            print()

    def print_actions(self, actions):
        ''' Print the action space properly '''
        print('Actions :')

        for id, action in enumerate(actions):
            if id % self.size == 0:
                print("*" * 20, "Layer %d" % (((id + 1) // self.size) + 1), "*" * 20)

            state = self[id]
            name = state['name']
            vals = [(n, p) for n, p in zip(state['values'], *action)]
            print("%s : " % name, vals)
        print()

    def __getitem__(self, id):
        return self.states[id % self.size]

    @property
    def size(self):
        return self.state_count_


class Controller:
    '''
    Utility class to manage the RNN Controller
    '''
    def __init__(self, policy_session, num_layers, state_space,
                 reg_param=0.001,
                 discount_factor=0.99,
                 exploration=0.8,
                 controller_cells=32,
                 embedding_dim=20,
                 clip_norm=0.0,
                 restore_controller=False):
        self.policy_session = policy_session  # type: tf.Session

        self.num_layers = num_layers
        self.state_space = state_space  # type: StateSpace
        self.state_size = self.state_space.size

        self.controller_cells = controller_cells
        self.embedding_dim = embedding_dim
        self.reg_strength = reg_param
        self.discount_factor = discount_factor
        self.exploration = exploration
        self.restore_controller = restore_controller
        self.clip_norm = clip_norm

        self.reward_buffer = []
        self.state_buffer = []

        self.cell_outputs = []
        self.policy_classifiers = []
        self.policy_actions = []
        self.policy_labels = []

        self.build_policy_network()

    def get_action(self, state):
        '''
        Gets a one hot encoded action list, either from random sampling or from
        the Controller RNN

        Args:
            state: a list of one hot encoded states, whose first value is used as initial
                state for the controller RNN

        Returns:
            A one hot encoded action list
        '''
        if np.random.random() < self.exploration:
            print("Generating random action to explore")
            actions = []

            for i in range(self.state_size * self.num_layers):
                state_ = self.state_space[i]
                size = state_['size']

                sample = np.random.choice(size, size=1)
                sample = state_['index_map_'][sample[0]]
                action = self.state_space.embedding_encode(i, sample)
                actions.append(action)
            return actions

        else:
            print("Prediction action from Controller")
            initial_state = self.state_space[0]
            size = initial_state['size']

            if state[0].shape != (1, size):
                state = state[0].reshape((1, size)).astype('int32')
            else:
                state = state[0]

            print("State input to Controller for Action : ", state.flatten())

            with self.policy_session.as_default():
                K.set_session(self.policy_session)

                with tf.name_scope('action_prediction'):
                    pred_actions = self.policy_session.run(self.policy_actions, feed_dict={self.state_input: state})

                return pred_actions


    def build_policy_network(self):
        with self.policy_session.as_default():
            K.set_session(self.policy_session)

            with tf.name_scope('controller'):
                with tf.variable_scope('policy_network'):

                    # state input is the first input fed into the controller RNN.
                    # the rest of the inputs are fed to the RNN internally
                    with tf.name_scope('state_input'):
                        state_input = tf.placeholder(dtype=tf.int32, shape=(1, None), name='state_input')

                    self.state_input = state_input

                    # we can use LSTM as the controller as well
                    nas_cell = tf.nn.rnn_cell.LSTMCell(self.controller_cells)
                    cell_state = nas_cell.zero_state(batch_size=1, dtype=tf.float32)

                    embedding_weights = []

                    # for each possible state, create a new embedding. Reuse the weights for multiple layers.
                    with tf.variable_scope('embeddings', reuse=tf.AUTO_REUSE):
                        for i in range(self.state_size):
                            state_ = self.state_space[i]
                            size = state_['size']

                            # size + 1 is used so that 0th index is never updated and is "default" value
                            weights = tf.get_variable('state_embeddings_%d' % i,
                                                      shape=[size + 1, self.embedding_dim],
                                                      initializer=tf.initializers.random_uniform(-1., 1.))

                            embedding_weights.append(weights)

                        # initially, cell input will be 1st state input
                        embeddings = tf.nn.embedding_lookup(embedding_weights[0], state_input)

                    cell_input = embeddings

                    # we provide a flat list of chained input-output to the RNN
                    for i in range(self.state_size * self.num_layers):
                        state_id = i % self.state_size
                        state_space = self.state_space[i]
                        size = state_space['size']

                        with tf.name_scope('controller_output_%d' % i):
                            # feed the ith layer input (i-1 layer output) to the RNN
                            outputs, final_state = tf.nn.dynamic_rnn(nas_cell,
                                                                     cell_input,
                                                                     initial_state=cell_state,
                                                                     dtype=tf.float32)

                            # add a new classifier for each layers output
                            classifier = tf.layers.dense(outputs[:, -1, :], units=size, name='classifier_%d' % (i),
                                                         reuse=False)
                            preds = tf.nn.softmax(classifier)

                            # feed the previous layer (i-1 layer output) to the next layers input, along with state
                            # take the class label
                            cell_input = tf.argmax(preds, axis=-1)
                            cell_input = tf.expand_dims(cell_input, -1, name='pred_output_%d' % (i))
                            cell_input = tf.cast(cell_input, tf.int32)
                            cell_input = tf.add(cell_input, 1)  # we avoid using 0 so as to have a "default" embedding at 0th index

                            # embedding lookup of this state using its state weights ; reuse weights
                            cell_input = tf.nn.embedding_lookup(embedding_weights[state_id], cell_input,
                                                           name='cell_output_%d' % (i))

                            cell_state = final_state

                        # store the tensors for later loss computation
                        self.cell_outputs.append(cell_input)
                        self.policy_classifiers.append(classifier)
                        self.policy_actions.append(preds)

            policy_net_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='policy_network')

            with tf.name_scope('optimizer'):
                self.global_step = tf.Variable(0, trainable=False)
                starter_learning_rate = 0.1
                learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                           500, 0.95, staircase=True)

                tf.summary.scalar('learning_rate', learning_rate)

                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

            with tf.name_scope('losses'):
                self.discounted_rewards = tf.placeholder(tf.float32, shape=(None,), name='discounted_rewards')
                tf.summary.scalar('discounted_reward', tf.reduce_sum(self.discounted_rewards))

                # calculate sum of all the individual classifiers
                cross_entropy_loss = 0
                for i in range(self.state_size * self.num_layers):
                    classifier = self.policy_classifiers[i]
                    state_space = self.state_space[i]
                    size = state_space['size']

                    with tf.name_scope('state_%d' % (i + 1)):
                        labels = tf.placeholder(dtype=tf.float32, shape=(None, size), name='cell_label_%d' % i)
                        self.policy_labels.append(labels)

                        ce_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=classifier, labels=labels)
                        tf.summary.scalar('state_%d_ce_loss' % (i + 1), tf.reduce_mean(ce_loss))

                    cross_entropy_loss += ce_loss

                policy_gradient_loss = tf.reduce_mean(cross_entropy_loss)
                reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in policy_net_variables])  # Regularization

                # sum up policy gradient and regularization loss
                self.total_loss = policy_gradient_loss + self.reg_strength * reg_loss
                tf.summary.scalar('total_loss', self.total_loss)

                self.gradients = self.optimizer.compute_gradients(self.total_loss)

                with tf.name_scope('policy_gradients'):
                    # normalize gradients so that they dont explode if argument passed
                    if self.clip_norm is not None and self.clip_norm != 0.0:
                        norm = tf.constant(self.clip_norm, dtype=tf.float32)
                        gradients, vars = zip(*self.gradients)  # unpack the two lists of gradients and the variables
                        gradients, _ = tf.clip_by_global_norm(gradients, norm)  # clip by the norm
                        self.gradients = list(zip(gradients, vars))  # we need to set values later, convert to list

                    # compute policy gradients
                    for i, (grad, var) in enumerate(self.gradients):
                        if grad is not None:
                            self.gradients[i] = (grad * self.discounted_rewards, var)

                # training update
                with tf.name_scope("train_policy_network"):
                    # apply gradients to update policy network
                    self.train_op = self.optimizer.apply_gradients(self.gradients, global_step=self.global_step)

            self.summaries_op = tf.summary.merge_all()

            timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
            filename = 'logs/%s' % timestr

            self.summary_writer = tf.summary.FileWriter(filename, graph=self.policy_session.graph)

            self.policy_session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(max_to_keep=1)

            if self.restore_controller:
                path = tf.train.latest_checkpoint('weights/')

                if path is not None and tf.train.checkpoint_exists(path):
                    print("Loading Controller Checkpoint !")
                    self.saver.restore(self.policy_session, path)

    def store_rollout(self, state, reward):
        self.reward_buffer.append(reward)
        self.state_buffer.append(state)

        # dump buffers to file if it grows larger than 50 items
        if len(self.reward_buffer) > 20:
            with open('buffers.txt', mode='a+') as f:
                for i in range(20):
                    state_ = self.state_buffer[i]
                    state_list = self.state_space.parse_state_space_list(state_)
                    state_list = ','.join(str(v) for v in state_list)

                    f.write("%0.4f,%s\n" % (self.reward_buffer[i], state_list))

                print("Saved buffers to file `buffers.txt` !")

            self.reward_buffer = [self.reward_buffer[-1]]
            self.state_buffer = [self.state_buffer[-1]]

    def discount_rewards(self):
        '''
        Compute discounted rewards over the entire reward buffer

        Returns:
            Discounted reward value
        '''
        rewards = np.asarray(self.reward_buffer)
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards[-1]

    def train_step(self):
        '''
        Perform a single train step on the Controller RNN

        Returns:
            the training loss
        '''
        states = self.state_buffer[-1]
        label_list = []

        # parse the state space to get real value of the states,
        # then one hot encode them for comparison with the predictions
        state_list = self.state_space.parse_state_space_list(states)
        for id, state_value in enumerate(state_list):
            state_one_hot = self.state_space.embedding_encode(id, state_value)
            label_list.append(state_one_hot)

        # the initial input to the controller RNN
        state_input_size = self.state_space[0]['size']
        state_input = states[0].reshape((1, state_input_size)).astype('int32')
        print("State input to Controller for training : ", state_input.flatten())

        # the discounted reward value
        reward = self.discount_rewards()
        reward = np.asarray([reward]).astype('float32')

        feed_dict = {
            self.state_input: state_input,
            self.discounted_rewards: reward
        }

        # prepare the feed dict with the values of all the policy labels for each
        # of the Controller outputs
        for i, label in enumerate(label_list):
            feed_dict[self.policy_labels[i]] = label

        with self.policy_session.as_default():
            K.set_session(self.policy_session)

            print("Training RNN (States ip) : ", state_list)
            print("Training RNN (Reward ip) : ", reward.flatten())
            _, loss, summary, global_step = self.policy_session.run([self.train_op, self.total_loss, self.summaries_op,
                                                                     self.global_step],
                                                                     feed_dict=feed_dict)

            self.summary_writer.add_summary(summary, global_step)
            self.saver.save(self.policy_session, save_path='weights/controller.ckpt', global_step=self.global_step)

            # reduce exploration after many train steps
            if global_step != 0 and global_step % 20 == 0 and self.exploration > 0.5:
                self.exploration *= 0.99

        return loss

    def remove_files(self):
        files = ['train_history.csv', 'buffers.txt']

        for file in files:
            if os.path.exists(file):
                os.remove(file)
