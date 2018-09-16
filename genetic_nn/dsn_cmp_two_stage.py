import numpy as np
import tensorflow as tf
import math
import os
import sys
import matplotlib.pyplot as plt
import random
import pickle

from framework.wrapper import TwoStageComplete as sim

# np.random.seed(10)
# random.seed(10)

eval_core = sim.EvaluationCore("./framework/yaml_files/two_stage_full.yaml")
def evaluate_individual(individual, verbose=False):
    # TODO
    # returns a scalar number representing the cost function of that individual
    # return (sum(individual),)
    mp1_idx = int(individual[0])
    mn1_idx = int(individual[1])
    mn3_idx = int(individual[2])
    mp3_idx = int(individual[3])
    mn5_idx = int(individual[4])
    mn4_idx = int(individual[5])
    cc_idx  = int(individual[6])
    result = eval_core.cost_fun(mp1_idx, mn1_idx, mp3_idx, mn3_idx, mn4_idx, mn5_idx, cc_idx, verbose=verbose)
    return result

def generate_data_set(n=10, evaluate=True):
    if evaluate:
        print("[info] generating %d random data" %n)
    data_set, cost_set, bw_set, gain_set , ibias_set = [], [], [], [], []

    for _ in range(n):

        mp1_idx = random.randint(0 ,len(eval_core.mp1_vec)-1)
        mn1_idx = random.randint(0 ,len(eval_core.mn1_vec)-1)
        mn3_idx = random.randint(0 ,len(eval_core.mn3_vec)-1)
        mp3_idx = random.randint(0 ,len(eval_core.mp3_vec)-1)
        mn5_idx = random.randint(0 ,len(eval_core.mn5_vec)-1)
        mn4_idx = random.randint(0 ,len(eval_core.mn4_vec)-1)
        cc_idx =  random.randint(0 ,len(eval_core.cc_vec)-1)

        sample_dsn = [mp1_idx, mn1_idx, mn3_idx, mp3_idx, mn5_idx, mn4_idx, cc_idx]
        data_set.append(sample_dsn)
        if evaluate:
            result =  evaluate_individual(sample_dsn, verbose=False)
            cost = result[0]
            cost_set.append(cost)


    if evaluate:
        return np.array(data_set), np.array(cost_set)
    else:
        return np.array(data_set), _

# data generation and preprocessing. for new environments these numbers should be readjusted
n_init_samples = 200
n_new_samples = 10
num_designs = 2
num_features_per_design = 7
num_classes = 2
num_nn_features = num_designs * num_features_per_design
valid_frac = 0.2
max_n_retraining = 15

k_top = 10 #during training only consider comparison between k_top ones and the others
ref_dsn_idx = k_top #during inference compare new randomly generated samples with this design in the sorted dataset

# training settings
num_epochs = 100
batch_size = 8
display_step = 10
ckpt_step = 10

summary_dir = 'genetic_nn/summary'


# nn hyper parameters
nhidden1 = 20 # number of hidden nodes
nhidden2 = 20
nhidden3 = 20

learning_rate = 0.03
decay_steps = 10000
decay_rate = 0.5

l2_reg_scale = 0.003
DROP_OUT_PROB = 0.5

graph = tf.Graph()

with graph.as_default():
    # tf.set_random_seed(10)
    tf_train_dataset = tf.placeholder(tf.float32, shape=(None, num_nn_features), name='train_in')
    tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_classes), name='train_labels')
    loss_weights = tf.placeholder(tf.float32, shape=[None, 1], name='adjustment_weights')
    keep_prob = tf.placeholder(tf.float32)

    with tf.variable_scope('normalizer'):
        mu = tf.Variable(tf.zeros([num_nn_features], dtype=tf.float32), name='training_set_mu', trainable=False)
        std = tf.Variable(tf.zeros([num_nn_features], dtype=tf.float32), name='training_set_std', trainable=False)
        tf_train_dataset_norm = (tf_train_dataset - mu) / (std + 1e-6)
    with tf.variable_scope("regulizer"):
        l2_reg_fn = tf.contrib.layers.l2_regularizer(l2_reg_scale, scope="l2_reg")


    def nn_model(input_data, name='nn_model', reuse=False, is_test=False):
        with tf.variable_scope(name):
            layer1 = tf.contrib.layers.fully_connected(input_data, nhidden1, reuse=reuse, scope='fc1',
                                                       weights_regularizer=l2_reg_fn)
            do1 = tf.nn.dropout(layer1, keep_prob)
            layer2 = tf.contrib.layers.fully_connected(do1, nhidden2, reuse=reuse, scope='fc2',
                                                       weights_regularizer=l2_reg_fn)
            do2 = tf.nn.dropout(layer2, keep_prob)
            layer3 = tf.contrib.layers.fully_connected(do2, nhidden3, reuse=reuse, scope='fc3',
                                                       weights_regularizer=l2_reg_fn)
            do3 = tf.nn.dropout(layer3, keep_prob)
            logits = tf.contrib.layers.fully_connected(do3, num_classes,
                                                       activation_fn=None,
                                                       weights_regularizer=l2_reg_fn,
                                                       reuse=reuse, scope='fc_out')
        return logits

    with tf.variable_scope('regulizer'):
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = tf.reduce_sum(reg_losses)

    train_logits = nn_model(tf_train_dataset_norm, name='train_nn')
    train_prediction = tf.nn.softmax(train_logits)
    with tf.variable_scope("loss"):
        likelihoods = tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,
                                                              logits=train_logits)
        weighted_likelihoods = tf.multiply(likelihoods, loss_weights)
        loss = tf.reduce_mean(weighted_likelihoods) + reg_loss

    with tf.variable_scope("optimizer"):
        # global_step = tf.Variable(0, name='global_step', trainable=False)
        # starter_learning_rate = learning_rate
        # lr = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=True)
        # optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)
        # tf.summary.scalar('learning_rate', learning_rate)
        # tf.summary.scalar('lr', lr)
        optimizer = tf.train.AdamOptimizer().minimize(loss)


    def accuracy(predictions, labels, name='accuracy'):
        with tf.variable_scope(name):
            # predicted_label = tf.cast(tf.greater(predictions, 0.99999), tf.float32)
            # correct_predictions = tf.equal(predicted_label, labels)
            correct_predictions = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(labels, axis=1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            return accuracy

    tf_training_accuracy = accuracy(train_prediction, tf_train_labels, name='train_accuracy')

    # validation prediction
    tf_valid_dataset = tf.placeholder(tf.float32, shape=(None, num_nn_features), name='valid_in')
    tf_valid_labels = tf.placeholder(tf.float32, shape=(None, num_classes), name='valid_labels')
    with tf.variable_scope('normalizer', reuse=True):
        tf_valid_dataset_norm = (tf_valid_dataset - mu) / (std + 1e-6)
    valid_logits = nn_model(tf_valid_dataset_norm, name='train_nn', reuse=True)

    valid_prediction = tf.nn.softmax(valid_logits)
    valid_accuracy = accuracy(valid_prediction, tf_valid_labels, name='validation_accuracy')
    # summarize a couple of things
    tf.summary.scalar("train_loss", loss)
    tf.summary.scalar("train_accuracy", tf_training_accuracy)
    tf.summary.scalar("validation_accuracy", valid_accuracy)

    # summarize weights and biases
    all_vars= tf.global_variables()
    def get_var(name):
        for i in range(len(all_vars)):
            if all_vars[i].name.startswith(name):
                return all_vars[i]
        return None
    tf.summary.histogram("fc1_weight", get_var('train_nn/fc1/weights'))
    tf.summary.histogram("fc1_biases", get_var('train_nn/fc1/biases'))
    tf.summary.histogram("fc2_weight", get_var('train_nn/fc2/weights'))
    tf.summary.histogram("fc2_biases", get_var('train_nn/fc2/biases'))
    tf.summary.histogram("fc3_weight", get_var('train_nn/fc3/weights'))
    tf.summary.histogram("fc3_biases", get_var('train_nn/fc3/biases'))
    tf.summary.histogram("fc_out_weight", get_var('train_nn/fc_out/weights'))
    tf.summary.histogram("fc_out_biases", get_var('train_nn/fc_out/biases'))

    # inference phase
    merged_summary = tf.summary.merge_all()

class BatchGenerator(object):
    def __init__(self, data_set, labels, weights, batch_size):
        self._data_set = data_set
        self._labels = labels
        self._weights = weights
        self._data_size = data_set.shape[0]
        self._batch_size = batch_size
        self._segment = self._data_size // batch_size
        self.last_index = 0

    def next(self):

        if ((self.last_index+1)*self._batch_size > self._data_size):
            data1 = self._data_set[self.last_index * self._batch_size:,:]
            data2 = self._data_set[:((self.last_index+1)*self._batch_size)%self._data_size, :]
            labels1 = self._labels[self.last_index * self._batch_size:, :]
            labels2 = self._labels[:((self.last_index+1)*self._batch_size)%self._data_size, :]
            weights1 = self._weights[self.last_index * self._batch_size:]
            weights2 = self._weights[:((self.last_index+1)*self._batch_size)%self._data_size]
            batch_data = np.concatenate((data1, data2), axis=0)
            batch_labels = np.concatenate((labels1, labels2), axis=0)
            batch_weights = np.concatenate((weights1, weights2), axis=0)
        else:
            batch_data = self._data_set[self.last_index * self._batch_size:(self.last_index + 1) * self._batch_size, :]
            batch_labels = self._labels[self.last_index * self._batch_size:(self.last_index + 1) * self._batch_size, :]
            batch_weights = self._weights[self.last_index * self._batch_size:(self.last_index + 1) * self._batch_size]

        self.last_index = (self.last_index+1) % (self._segment+1)
        return batch_data, batch_labels, batch_weights

def combine(dataset, cost, for_training=False):
    # this combine function is used when generating nn data for training
    # label [0 1] means design1 is "sufficiently" worse than than design2 (e.g cost1 is at
    # least %10 higher than cost2) and label [1 0] means it is not the case.
    # it is really important to combine those pairs that are going to be useful during inference
    # since during inference one of the designs is always good we should make sure that during training also
    # this bias on some level exists. For this reason we will sort the data set and produce pairs that always have at least
    # one design from top k_top designs.

    assert k_top < len(dataset)

    sorted_indices = sorted(range(len(cost)), key=lambda x: cost[x])
    sorted_dataset = dataset[sorted_indices]
    sorted_cost = cost[sorted_indices]

    category = []
    nn_dataset, nn_labels = [], []
    cost_arr = [] # just for debbuging purposes

    # adjustment weights are for adjusting the difference in the number of samples that are like (x0,x1) where
    # both x0 and x1 are good with the samples that have at least one bad x.
    adjustment_weights = []
    num_both_good_comparisons = k_top*(k_top-1)/2
    n = len(sorted_dataset)
    num_one_bad_comparisons = n*(n-1)/2 - (n-k_top) * (n-k_top-1)/2

    # weights_good = (num_one_bad_comparisons)#/(num_one_bad_comparisons+num_both_good_comparisons)
    weights_good = 1
    # weights_bad = (num_both_good_comparisons)#/(num_one_bad_comparisons+num_both_good_comparisons)
    weights_bad = 1

    for i in range(k_top):
        for j in range(i+1, len(sorted_dataset)):
            # (x0,x1) (x0,x2) ... x0 is always going to be the first one going into nn but we also want to have x0
            # to be the second term. so the probability of having (x0,x1) and (x1,x0) is equal.

            # if random.random() < 0.5:
            #     nn_dataset.append(list(sorted_dataset[i,:])+list(sorted_dataset[j,:]))
            #     cost_arr.append([sorted_cost[i], sorted_cost[j]])
            #     label = 1 if (sorted_cost[i] > sorted_cost[j]) else 0
            # else:
            #     nn_dataset.append(list(sorted_dataset[j,:])+list(sorted_dataset[i,:]))
            #     cost_arr.append([sorted_cost[j], sorted_cost[i]])
            #     label = 1 if (sorted_cost[j] > sorted_cost[i]) else 0

            # (x0,x1) part
            nn_dataset.append(list(sorted_dataset[i,:])+list(sorted_dataset[j,:]))
            cost_arr.append([sorted_cost[i], sorted_cost[j]])
            label = 1 if (sorted_cost[i] > sorted_cost[j]) else 0
            if j < k_top:
                adjustment_weights.append(weights_good)

            else:
                adjustment_weights.append(weights_bad)
            category.append(label)

            # (x1,x0) part
            nn_dataset.append(list(sorted_dataset[j,:])+list(sorted_dataset[i,:]))
            cost_arr.append([sorted_cost[j], sorted_cost[i]])
            label = 1 if (sorted_cost[j] > sorted_cost[i]) else 0
            if j < k_top:
                adjustment_weights.append(weights_good)
            else:
                adjustment_weights.append(weights_bad)
            category.append(label)



    nn_labels = np.zeros((len(category), num_classes))
    nn_labels[np.arange(len(category)), category] = 1
    return np.array(nn_dataset), np.array(nn_labels), np.array(cost_arr), np.array(adjustment_weights)

def shuffle_dataset(dataset, labels):
    """
    :param dataset: this is suppose to be a list of individuals
    :param labels:  this is a numpy array
    :return:
    """
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def train(session, dataset, cost, writer, num_epochs=10, batch_size=128):
    all_vars = tf.global_variables()
    saver = tf.train.Saver(all_vars)
    nn_dataset, nn_labels, cost_arr, adjustment_weights = combine(dataset, cost)

    print("[Debug_data] design1, design2, costs, label")

    # for i in range(len(nn_dataset)):
    #     design1 = nn_dataset[i, :num_features_per_design]
    #     design2 = nn_dataset[i, num_features_per_design:]
    #     costs = cost_arr[i,:]
    #     label = nn_labels[i,:]
    #     print("[Debug_data] {} -> {} -> {} -> {}".format(design1, design2, costs, label))
    # nn_dataset, nn_labels = shuffle_dataset(nn_dataset, nn_labels) # Do we really need to shuffle? yes we do :)


    permutation = np.random.permutation(nn_labels.shape[0])
    nn_dataset = nn_dataset[permutation]
    nn_labels = nn_labels[permutation]
    adjustment_weights = adjustment_weights[permutation]

    # print("[Debug_shuffle] sample -> label ")
    # for i in range(len(nn_dataset)):
    #     print("[Debug_shuffle] {} -> {} " .format(nn_dataset[i], nn_labels[i]))

    boundry_index = nn_dataset.shape[0] - int(nn_dataset.shape[0]*valid_frac)
    train_dataset = nn_dataset[:boundry_index]
    train_labels = nn_labels[:boundry_index]
    valid_dataset = nn_dataset[boundry_index:]
    valid_labels = nn_labels[boundry_index:]

    train_weights = adjustment_weights[:boundry_index]
    valid_weights = adjustment_weights[boundry_index:]

    data_dict = {'train_input': train_dataset,
                 'train_labels': train_labels,
                 'valid_input': valid_dataset,
                 'valid_labels': valid_labels}

    with open('genetic_nn/checkpoint/two_stage/train_valid_data.pkl', 'wb') as f:
        pickle.dump(data_dict, f)

    # find the mean and std of dataset for normalizing
    train_mean = np.mean(train_dataset, axis=0)
    train_std = np.std(train_dataset, axis=0)

    print("[info] dataset size:%d" %len(dataset))
    print("[info] combine size:%d" %len(nn_dataset))
    print("[info] train_dataset: positive_samples/total ratio : %d/%d" %(np.sum(train_labels, axis=0)[0], train_labels.shape[0]))
    print("[info] valid_dataset: positive_samples/total ratio : %d/%d" %(np.sum(valid_labels, axis=0)[0], valid_labels.shape[0]))

    batch_generator = BatchGenerator(train_dataset, train_labels, train_weights, batch_size)
    print("[info] training the model with dataset ....")

    total_n_batches = int(len(train_dataset) // batch_size)
    print("[info] number of total batches: %d" %total_n_batches)
    print(30*"-")

    tf.global_variables_initializer().run()
    mu.assign(train_mean).op.run()
    std.assign(train_std).op.run()

    for epoch in range(num_epochs):
        avg_loss = 0.
        avg_train_acc = 0.
        avg_valid_acc = 0.
        feed_dict = {}
        for iter in range(total_n_batches):
            batch_data, batch_labels, batch_weights = batch_generator.next()
            drop_out_prob = DROP_OUT_PROB
            feed_dict = {tf_train_dataset   :batch_data,
                         tf_train_labels    :batch_labels,
                         tf_valid_dataset   :valid_dataset,
                         tf_valid_labels    :valid_labels,
                         keep_prob          :drop_out_prob,
                         loss_weights       :batch_weights[:, None]}

            _, l, valid_acc, train_acc= session.run([optimizer, loss, valid_accuracy, tf_training_accuracy],
                                                    feed_dict=feed_dict)
            avg_loss += l / total_n_batches
            avg_train_acc += train_acc / total_n_batches
            avg_valid_acc += valid_acc / total_n_batches

        s = session.run(merged_summary, feed_dict=feed_dict)

        # sample_input = np.array([[19, 38, 220, 55], [43, 10, 26, 25]])
        # sample_predictions = session.run(train_prediction, feed_dict={tf_train_dataset: sample_input})
        # print("[Debug] sample_prediction {}, {}".format(sample_predictions[0], sample_predictions[1]))
        writer.add_summary(s, epoch)
        if epoch % ckpt_step == 0:
            saver.save(session, 'genetic_nn/checkpoint/two_stage/checkpoint.ckpt')
            dict_to_save = dict(dataset=dataset, cost=cost)
            with open('genetic_nn/checkpoint/two_stage/data.pkl', 'wb') as f:
                pickle.dump(dict_to_save, f)
        if epoch % display_step == 0:
            print("[epoch %d] loss: %f" %(epoch, avg_loss))
            print("train_acc = %.2f%%, valid_acc = %.2f%%" %(avg_train_acc*100, avg_valid_acc*100))

def plot_cost1d(range, dataset, cost):
    cost_min, cost_max = range
    plt.figure(1)
    plt.hist(cost, 50)
    plt.title('distribution of cost function')
    plt.figure(2)
    plt.hist(dataset[:,0], 50)
    plt.title('distribution of res_idx')
    plt.figure(3)
    plt.hist(dataset[:,1], 50)
    plt.title('distribution of mul_idx')
    plt.show()

def find_dsns_with_cost(cost_range, data_set, cost_set):
    """

    :param cost_range:
    :param data_set:
    :param cost_set:
    :return:

    >>> find_dsns_with_cost([0.6,0.7], dataset, cost)

    """
    indices = []
    for i in range(len(cost_set)):
        if cost_set[i] < cost_range[1] and cost_set[i] > cost_range[0]:
            indices.append(i)
    return indices

def combine2(design_arr1, cost_arr1, design_arr2, cost_arr2):
    # this combine function is used when doing inference
    # what this does it combines the two design arrs and produce the corresponding inputs to nn.
    # but also makes sure in the pairs generated, there is always one design from design_arr1, and one other is from
    # design_arr2

    nn_dataset = []
    cost_arr = []
    category = []

    for i in range(len(design_arr1)):
        for j in range(len(design_arr2)):

            if random.random() < 0.5:
                nn_dataset.append(list(design_arr1[i,:])+list(design_arr2[j,:]))
                cost_arr.append([cost_arr1[i], cost_arr2[j]])
                label = 1 if (cost_arr1[i] > cost_arr2[j]) else 0
            else:
                nn_dataset.append(list(design_arr2[j,:])+list(design_arr1[i,:]))
                cost_arr.append([cost_arr2[j], cost_arr1[i]])
                label = 1 if (cost_arr2[j] > cost_arr1[i]) else 0

            category.append(label)
    nn_labels = np.zeros((len(category), num_classes))
    nn_labels[np.arange(len(category)), category] = 1
    return np.array(nn_dataset), np.array(nn_labels), np.array(cost_arr)


def test_model2(session, training_dataset, training_cost):

    n_samples = 1000

    # generate the designs
    if os.path.exists('genetic_nn/checkpoint/two_stage/test_data.pkl'):
        with open('genetic_nn/checkpoint/two_stage/test_data.pkl', 'rb') as f:
            read_data = pickle.load(f)
            new_designs, new_cost = read_data['dataset'], read_data['cost']
    else:
        new_designs, new_cost = generate_data_set(n_samples)
        write_data = dict(dataset=new_designs, cost=new_cost)
        with open('genetic_nn/checkpoint/two_stage/test_data.pkl', 'wb') as f:
            pickle.dump(write_data, f)

    # sort training data_set
    sorted_indices = sorted(range(len(training_cost)), key=lambda x: training_cost[x])
    sorted_training_designs = training_dataset[sorted_indices]
    sorted_training_cost = training_cost[sorted_indices]

    nn_inputs, nn_labels, design_costs = combine2(sorted_training_designs[k_top-1:k_top],
                                                  sorted_training_cost[k_top-1:k_top],
                                                  new_designs, new_cost)
    drop_out_prob = 1
    feed_dict = {tf_train_dataset: nn_inputs,
                 tf_train_labels: nn_labels,
                 keep_prob: drop_out_prob}
    predictions, = session.run([train_prediction], feed_dict=feed_dict)

    for i in range(ref_dsn_idx):
        print("[Debug_test] dataset: {} -> {}".format(sorted_training_designs[i], sorted_training_cost[i]))
    print("[Debug_test] ref design in dataset {} cost {}".format(sorted_training_designs[ref_dsn_idx-1],
                                                                 sorted_training_cost[ref_dsn_idx-1]))
    print("[Debug_test] design1, design2, costs, label, prediction, correctness")

    good_design_miss_cnt = 0
    good_design_caught_cnt = 0
    bad_design_miss_cnt = 0
    bad_design_caught_cnt = 0
    for i in range(len(nn_inputs)):
        design1 = nn_inputs[i, :num_features_per_design]
        design2 = nn_inputs[i, num_features_per_design:]
        costs = design_costs[i, :]
        label = nn_labels[i, :]
        prediction = predictions[i, :]
        predicted_label = 0
        if np.argmax(prediction) == np.argmax(label):
            # this is a correct prediction (caught design)
            predicted_label = 1
            # cnt += 1
            if sorted_training_cost[k_top-1] < costs[0] or sorted_training_cost[k_top-1] < costs[1]:
                # bad_design_caught_cnt
                bad_design_caught_cnt += 1
            elif sorted_training_cost[k_top-1] > costs[0] or sorted_training_cost[k_top-1] > costs[1]:
                # good_design_caught_cnt
                good_design_caught_cnt += 1
        else:
            # this is a miss classification
            if sorted_training_cost[k_top-1] < costs[0] or sorted_training_cost[k_top-1] < costs[1]:
                # bad_design_miss_cnt
                bad_design_miss_cnt += 1
            elif sorted_training_cost[k_top-1] > costs[0] or sorted_training_cost[k_top-1] > costs[1]:
                # good_design_miss_cnt
                good_design_miss_cnt += 1

        print("[Debug_test] {} -> {} -> {} -> {} -> {} -> {} ".format(list(design1), list(design2), costs, label, prediction, predicted_label))


    print("[Debug_test] accuracy = {}/{} = {}".format(good_design_caught_cnt+bad_design_caught_cnt,n_samples,
                                                      1.0*(good_design_caught_cnt+bad_design_caught_cnt)/n_samples))

    print("[Debug_test] good_design recall accuracy = {}/{} = {}".format(good_design_caught_cnt,
                                                                             good_design_caught_cnt+good_design_miss_cnt,
                                                                             1.0*good_design_caught_cnt/(good_design_caught_cnt+good_design_miss_cnt)))
    print("[Debug_test] bad_design recall accuracy = {}/{} = {}".format(bad_design_caught_cnt,
                                                                             bad_design_caught_cnt+bad_design_miss_cnt,
                                                                             1.0*bad_design_caught_cnt/(bad_design_caught_cnt+bad_design_miss_cnt)))

    print("[Debug_test] good_design precision accuracy = {}/{} = {}".format(good_design_caught_cnt,
                                                                            good_design_caught_cnt+bad_design_miss_cnt,
                                                                            1.0*good_design_caught_cnt/(good_design_caught_cnt+bad_design_miss_cnt)))
    print("[Debug_test] bad_design precision accuracy = {}/{} = {}".format(bad_design_caught_cnt,
                                                                           bad_design_caught_cnt+good_design_miss_cnt,
                                                                           1.0*bad_design_caught_cnt/(bad_design_caught_cnt+good_design_miss_cnt)))


def test_model(session, training_dataset, training_cost):

    select_indices = []

    # plot_cost1d([0,3], training_dataset, training_cost)

    n_samples = 1000

    # generate the designs
    new_designs, new_cost = generate_data_set(n_samples)
    # sort them according to cost function for simplicity
    sorted_indices = sorted(range(len(new_cost)), key=lambda x: new_cost[x])
    sorted_designs = new_designs[sorted_indices]
    sorted_cost = new_cost[sorted_indices]

    # In here we should select the ones we care about and combine them to build the test_set for nn.
    # cluster1 = find_dsns_with_cost([1.9,3], sorted_designs, sorted_cost)
    # select_indices += random.sample(cluster1, 10)
    # select_indices += cluster1[-10:]
    # cluster2 = find_dsns_with_cost([2, 5], sorted_designs, sorted_cost)
    # select_indices += random.sample(cluster2, 5)
    # shuffle them randomly so there is no privilege because of the way we obtained them
    # random.shuffle(select_indices)

    #
    selected_designs =  sorted_designs[select_indices]
    selected_cost =     sorted_cost[select_indices]

    nn_inputs, nn_labels, design_costs, _ = combine(selected_designs, selected_cost)

    drop_out_prob = 1
    feed_dict = {tf_train_dataset: nn_inputs,
                 tf_train_labels: nn_labels,
                 keep_prob: drop_out_prob}
    predictions, = session.run([train_prediction], feed_dict=feed_dict)

    print("[Debug_test] design1, design2, costs, label, prediction, correctness")

    cnt = 0
    for i in range(len(nn_inputs)):
        design1 = nn_inputs[i, :2]
        design2 = nn_inputs[i, 2:]
        costs = design_costs[i, :]
        label = nn_labels[i, :]
        prediction = predictions[i, :]
        predicted_label = 0
        if np.argmax(prediction) == np.argmax(label):
            predicted_label = 1
            cnt += 1
        print("[Debug_test] {} -> {} -> {} -> {} -> {} -> {} ".format(design1, design2, costs, label, prediction, predicted_label))
    print("[Debug_test] accuracy = {}/{} = {}".format(cnt, len(nn_inputs), 1.0*cnt/len(nn_inputs)))

def run_model(session, design_pool, cost_pool, m_samples, ref_dsn, max_iter=1000):
    print(30*"-")
    print("[info] running model ... ")
    cnt = 0

    better_dsns = []
    better_dsns_costs, better_dsns_bw, better_dsns_gain, better_dsns_ibias = [], [], [], []
    better_dsns_pred = []
    for i in range(ref_dsn_idx):
        print("[Debug_test] dataset: {} -> {}".format(design_pool[i], cost_pool[i]))
    print("[debug] ref design {} with ref cost {}".format(design_pool[ref_dsn_idx], cost_pool[ref_dsn_idx]))
    for _ in range(max_iter):
        cnt += 1

        new_designs, _ = generate_data_set(1, evaluate=False)
        new_design = list(new_designs[0])
        # print("[debug] new_design = {}".format(new_design))
        if any((new_design == row).all() for row in design_pool):
            # if design is already in the design pool skip ...
            # print("[debug] design {} already exists".format(new_design))
            continue

        if  random.random() < 0.5:
            input_nn = np.array(new_design + list(design_pool[ref_dsn_idx]))
            ref_label = 0
        else:
            input_nn = np.array(list(design_pool[ref_dsn_idx]) + new_design)
            ref_label = 1

        drop_out_prob = 1
        feed_dict = {tf_train_dataset: input_nn[None, :], keep_prob: drop_out_prob}
        prediction = session.run(train_prediction, feed_dict=feed_dict).flatten()
        # print("[debug] design {} ->  {} -> {} ".format(new_design, design_pool[0], prediction))

        if np.argmax(prediction) == ref_label:
            # depending on the random ordering determine if the new design sample is better than the reference design
            better_dsns.append(new_design)
            better_dsns_pred.append(prediction)
            if len(better_dsns) == m_samples:
                break
        else:
            pass
            # just a sanity check for not too complicated circuit problems: run simulation for anything to make sure
            # I'm not doing anything too stupid
            # result = evaluate_individual(new_design)
            # cost = result[0]
            # if cost < cost_pool[ref_dsn_idx]:
            #     print("[debug] design {} with cost {} was better but missed with prediction {}".format(new_design, cost,
            #                                                                                            prediction))

    print("[info] new designs tried: %d" %cnt)
    print("[info] new candidates size: %d " %len(better_dsns))
    # now if we have enough number of new potential designs we do simulation for each
    if len(better_dsns) > 0.1*m_samples:
        for i in range(len(better_dsns)):
            result = evaluate_individual(better_dsns[i], verbose=False)
            cost = result[0]
            better_dsns_costs.append(cost)

    return np.array(better_dsns), np.array(better_dsns_costs), np.array(better_dsns_pred)

def test_swaping(session, dataset, cost):

    sorted_indices = sorted(range(len(dataset)), key=lambda x: cost[x])
    sorted_design_pool = dataset[sorted_indices]
    sorted_cost_pool = cost[sorted_indices]

    design1 = sorted_design_pool[ref_dsn_idx]
    cost1 = sorted_cost_pool[ref_dsn_idx]
    # design1 =
    # cost1=812
    cnt = 0
    n_samples = 1000

    print("[Debug_swapping] ref design: {} -> {}".format(list(design1), cost1))


    # generate the designs
    if os.path.exists('genetic_nn/checkpoint/two_stage/test_data.pkl'):
        with open('genetic_nn/checkpoint/two_stage/test_data.pkl', 'rb') as f:
            read_data = pickle.load(f)
            design2s, costs = read_data['dataset'], read_data['cost']
    else:
        design2s, costs = generate_data_set(n_samples)
        write_data = dict(dataset=design2s, cost=costs)
        with open('genetic_nn/checkpoint/two_stage/test_data.pkl', 'wb') as f:
            pickle.dump(write_data, f)

    for i in range(len(design2s)):
        design2 = design2s[i]
        cost2 = costs[i]
        nn_inputs = np.array([list(design1)+list(design2), list(design2)+list(design1)])

        drop_out_prob = 1
        feed_dict = {tf_train_dataset: nn_inputs,
                     keep_prob: drop_out_prob}
        predictions, = session.run([train_prediction], feed_dict=feed_dict)

        if np.argmax(predictions[0]) != np.argmax(predictions[1]):
            cnt += 1
        else:
            print("[Debug_swapping] found inconsistency in design: {} -> {}".format(list(design2), cost2))
            print("[Debug_swapping] [d1,d2] -> {}".format(predictions[0]))
            print("[Debug_swapping] [d2,d1] -> {}".format(predictions[1]))

    print("[Debug_swapping] cnt=%d" %cnt)

def test_swaping_on_train_valid(session, dataset, cost):

    cnt=0

    sorted_indices = sorted(range(len(dataset)), key=lambda x: cost[x])
    sorted_design_pool = dataset[sorted_indices]
    sorted_cost_pool = cost[sorted_indices]

    nn_dataset, nn_labels, cost_arr, adjustment_weights = combine(dataset, cost)

    drop_out_prob = 1
    feed_dict = {tf_train_dataset   :nn_dataset,
                 tf_train_labels    :nn_labels,
                 keep_prob          :drop_out_prob}

    predictions, = session.run([train_prediction], feed_dict=feed_dict)

    for i in range(len(nn_dataset)):
        design1 = nn_dataset[i, :num_features_per_design]
        design2 = nn_dataset[i, num_features_per_design:]
        costs = cost_arr[i, :]
        label = nn_labels[i, :]
        prediction = predictions[i, :]
        predicted_label = 0
        if np.argmax(prediction) == np.argmax(label):
            predicted_label = 1
            cnt += 1
        print("[Debug_test] {} -> {} -> {} -> {} -> {} -> {} ".format(design1, design2, costs, label, prediction, predicted_label))
    print("[Debug_test] accuracy = {}/{} = {}".format(cnt, len(nn_dataset), 1.0*cnt/len(nn_dataset)))


def test_two_designs(session, design1, design2):
    # print(type(design1))
    # print(type(design2))

    cost1 = evaluate_individual(list(design1))[0]
    cost2 = evaluate_individual(list(design2))[0]

    nn_inputs = np.array([list(design1)+list(design2), list(design2)+list(design1)])

    drop_out_prob = 1
    feed_dict = {tf_train_dataset: nn_inputs,
                 keep_prob: drop_out_prob}
    predictions, = session.run([train_prediction], feed_dict=feed_dict)

    print("[test_model] design1 {} | design2 {}".format(list(design1), list(design2)))
    print("[test_model] cost1 {} | cost2 {}".format(cost1, cost2))
    print("[test_model] (d1,d2): {} | (d2,d1): {}".format(predictions[0], predictions[1]))

def main():
    data_set_list,cost_set_list = [], []

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--model_dir', type=str, default='genetic_nn/checkpoint/two_stage')
    args = parser.parse_args()


    with tf.Session(graph=graph) as session:

        writer = tf.summary.FileWriter(summary_dir)
        writer.add_graph(graph)
        all_vars = tf.global_variables()
        saver = tf.train.Saver(all_vars)
        if not args.load_model:
            if os.path.exists('genetic_nn/checkpoint/two_stage/init_data.pkl'):
                with open('genetic_nn/checkpoint/two_stage/init_data.pkl', 'rb') as f:
                    read_data = pickle.load(f)
                    dataset, cost = read_data['dataset'], read_data['cost']
            else:
                dataset, cost = generate_data_set(n=n_init_samples)
                write_data = dict(dataset=dataset, cost=cost)
                with open('genetic_nn/checkpoint/two_stage/init_data.pkl', 'wb') as f:
                    pickle.dump(write_data, f)

            print("Initialized")
            train(session, dataset, cost, writer, num_epochs=num_epochs, batch_size=batch_size)
        else:
            print("Loading model from {}".format(args.model_dir))
            saver.restore(session, os.path.join(args.model_dir, 'checkpoint.ckpt'))
            with open(os.path.join(args.model_dir, 'data.pkl'), 'rb') as f:
                data = pickle.load(f)
                dataset = data['dataset']
                cost = data['cost']

        # change test model if you want to really see how the trained model works data
        # test_model(session, dataset, cost)
        # test_model2(session, dataset, cost)
        test_swaping(session, dataset, cost)
        # test_swaping_on_train_valid(session, dataset, cost)

        # sorted_indices = sorted(range(len(dataset)), key=lambda x: cost[x])
        # sorted_design_pool = dataset[sorted_indices]
        # test_two_designs(session, sorted_design_pool[ref_dsn_idx] , design2 = [45, 36, 47,77,35,9,49])



        # for i in range(max_n_retraining):
        #     # run_model() requires sorted design pool we have so far:
        #     sorted_indices = sorted(range(len(dataset)), key=lambda x: cost[x])
        #     sorted_design_pool = dataset[sorted_indices]
        #     sorted_cost_pool = cost[sorted_indices]
        #     # store the sorted design pool to a log file for later plotting
        #     data_set_list.append(sorted_design_pool)
        #     cost_set_list.append(sorted_cost_pool)
        #
        #     # get the reference design for comparison
        #     ref_design = sorted_design_pool[ref_dsn_idx]
        #     ref_cost = cost[sorted_indices[ref_dsn_idx]]
        #     print("[info] retraining step: {}, best design: {} -> {} ".format(i, ref_design, ref_cost))
        #     new_dataset, new_cost, new_predictions = run_model(session, sorted_design_pool, sorted_cost_pool,
        #                                                        n_new_samples,
        #                                                        ref_dsn_idx, max_iter=1000)
        #     if len(new_dataset) <= 0.1*n_new_samples :
        #         # there are new points found that are as good as the old solutions
        #         break
        #     for k in range(len(new_dataset)):
        #         print("[debug] {} -> {} -> {}".format(new_dataset[k], new_cost[k], new_predictions[k]))
        #
        #     dataset = np.concatenate((dataset, new_dataset), axis=0)
        #     cost = np.concatenate((cost, new_cost), axis=0)
        #     train(session, dataset, cost, writer, num_epochs=num_epochs, batch_size=batch_size)
        # print("[finished] best_solution = {}".format(dataset[sorted_indices[0]]))
        # print("[finished] cost = {}".format(cost[sorted_indices[0]]))
        # print("[finished] performance {} ".format(evaluate_individual(dataset[sorted_indices[0]]), verbose=False))
        #
        # write_data = dict(
        #     data_set_list=data_set_list,
        #     cost_set_list=cost_set_list,
        # )
        # with open('genetic_nn/log_files/two_stage_logbook.pickle', 'wb') as f:
        #     pickle.dump(write_data, f)

def plot_cost_hist():

    if os.path.exists('genetic_nn/checkpoint/two_stage/init_data.pkl'):
        with open('genetic_nn/checkpoint/two_stage/init_data.pkl', 'rb') as f:
            read_data = pickle.load(f)
            dataset, cost = read_data['dataset'], read_data['cost']
    else:
        dataset, cost = generate_data_set(n=10000)
        write_data = dict(dataset=dataset, cost=cost)
        with open('genetic_nn/checkpoint/two_stage/init_data.pkl', 'wb') as f:
            pickle.dump(write_data, f)


    sorted_indices = sorted(range(len(dataset)), key=lambda x: cost[x])
    sorted_design_pool = dataset[sorted_indices]
    sorted_cost_pool = cost[sorted_indices]
    print(sorted_design_pool[0], sorted_cost_pool[0])
    plt.figure(1)
    plt.hist(cost, bins=5000, range=[0,100000])
    plt.title('distribution of cost function')
    plt.show()

if __name__ == '__main__':
    main()
    # plot_cost_hist()