'''
The difference between this and cs_amp_lut.py is that some of the ea methods have been modified able us to plot the
characteristics of this circuit as the algorithm progresses. These changes are in evaluate_individual() and the
main eaMuPlusLambda() function.
'''

import random
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from deap.tools.support import Logbook

import numpy as np
import os
from scipy import interpolate

import genome.alg as alg
import tensorflow as tf
import math
import os
import matplotlib.pyplot as plt
import sys

log_file = './genetic_nn/log.txt'
file = open(log_file,'w')
# origin_stdout = sys.stdout
# sys.stdout = file


# from scoop import futures

######################################################################
## helper functions for working with files
def rel_path(fname):
    return os.path.join(os.path.dirname(__file__), fname)

def load_array(fname):
    with open(rel_path(fname), "rb") as f:
        arr = np.load(f)
    return arr
######################################################################
## function and classes related to this specific problem and dealing with the evaluation core
class EvaluationCore(object):

    def __init__(self, cir_yaml):
        # specs
        import yaml
        with open(cir_yaml, 'r') as f:
            yaml_data = yaml.load(f)

        # specs
        specs = yaml_data['target_specs']
        self.bw_min     = specs['bw_min']
        self.gain_min   = specs['gain_min']
        self.bias_max   = specs['ibias_max']


        self.res_vec = load_array("sweeps/res_vec.array")
        self.mul_vec = load_array("sweeps/mul_vec.array")
        self.bw_mesh = load_array("sweeps/bw.array")
        self.ibias_mesh = load_array("sweeps/ibias.array")
        self.gain_mesh = load_array("sweeps/gain.array")

        self.bw_fun = interpolate.interp2d(self.res_vec, self.mul_vec, self.bw_mesh, kind="cubic")
        self.bias_fun = interpolate.interp2d(self.res_vec, self.mul_vec, self.ibias_mesh, kind="cubic")
        self.gain_fun = interpolate.interp2d(self.res_vec, self.mul_vec, self.gain_mesh, kind="cubic")

    def cost_fun(self, res_idx, mul_idx, verbose=False):
        bw_cur = self.bw_fun(self.res_vec[res_idx], self.mul_vec[mul_idx])
        gain_cur = self.gain_fun(self.res_vec[res_idx], self.mul_vec[mul_idx])
        ibias_cur = self.bias_fun(self.res_vec[res_idx], self.mul_vec[mul_idx])
        if verbose:
            print('bw = %f vs. bw_min = %f' %(bw_cur, self.bw_min))
            print('gain = %f vs. gain_min = %f' %(gain_cur, self.gain_min))
            print('Ibias = %f vs. Ibias_max = %f' %(ibias_cur, self.bias_max))

        cost = 0
        if bw_cur < self.bw_min:
            cost += abs(bw_cur/self.bw_min - 1.0)
        if gain_cur < self.gain_min:
            cost += abs(gain_cur/self.gain_min - 1.0)
        cost += abs(ibias_cur/self.bias_max)/10

        return cost, bw_cur, gain_cur, ibias_cur


eval_core = EvaluationCore("./framework/yaml_files/cs_amp.yaml")

def init_inividual():
    # TODO
    # returns a vector representing a random individual
    # ind = [random.randint(1, 100) for _ in range(5)]
    # return creator.Individual(ind)
    res_idx = random.randint(0, len(eval_core.res_vec)-1)
    mul_idx = random.randint(0, len(eval_core.mul_vec)-1)
    return creator.Individual([res_idx, mul_idx])

def evaluate_individual(individual, verbose=False):
    # TODO
    # returns a scalar number representing the cost function of that individual
    # return (sum(individual),)
    res_idx = individual[0]
    mul_idx = individual[1]
    cost_val, bw_val, gain_val, ibias_val = eval_core.cost_fun(int(res_idx), int(mul_idx), verbose)
    return (cost_val, bw_val, gain_val, ibias_val)

######################################################################
## helper functions for opt_core
def print_best_ind(population):
    best_ind = None
    best_fitness = float('inf')
    print("--------- best_individual in the final population ---------")
    for ind in population:
        if ind.fitness.values[0] < best_fitness:
            best_fitness = ind.fitness.values[0]
            best_ind = ind
    cost, _, _, _= evaluate_individual(ind, verbose=True)
    print("cost = %f" %cost)

## learning algorithm combined with genetic algorithm
pop_size = 128
top = 40
bot = 40
ngen=10
evaluated_samples = []

num_inputs = 2
# input vector is going to be the index of the parameters within their vector
# so it might need to get normalized to have zero mean and one sigma. i.e (x-u)/sigma
# so what should the type of them be? int or float?
nhidden1 = 20 # number of hidden nodes
nhidden2 = 40
nhidden3 = 20
batch_size = 16
num_classes = 2
init_learning_rate = 0.5
l2_reg_scale = 0.0
valid_size = (top+bot) // 10
summary_dir = '/tmp/cs_nn/'


graph = tf.Graph()

with graph.as_default():

    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, num_inputs), name='train_in')
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_classes), name='train_labels')

    l2_reg_fn = tf.contrib.layers.l2_regularizer(l2_reg_scale, scope="l2_reg")

    def nn_model(input_data, name='nn_model', reuse=False, is_test=False):
        with tf.variable_scope(name):

            layer1 = tf.contrib.layers.fully_connected(input_data, nhidden1, weights_regularizer=l2_reg_fn,
                                                       # activation_fn=tf.nn.tanh,
                                                       # weights_initializer=tf.truncated_normal_initializer(1,2),
                                                       reuse=reuse, biases_regularizer=l2_reg_fn, scope='fc1')
            layer2 = tf.contrib.layers.fully_connected(layer1, nhidden2, weights_regularizer=l2_reg_fn,
                                                       # activation_fn=tf.nn.tanh,
                                                       # weights_initializer=tf.truncated_normal_initializer(1,2),
                                                       reuse=reuse, biases_regularizer=l2_reg_fn, scope='fc2')
            layer3 = tf.contrib.layers.fully_connected(layer2, nhidden3, weights_regularizer=l2_reg_fn,
                                                       # activation_fn=tf.nn.tanh,
                                                       # weights_initializer=tf.truncated_normal_initializer(1,2),
                                                       reuse=reuse, biases_regularizer=l2_reg_fn, scope='fc3')
            logits = tf.contrib.layers.fully_connected(layer3, num_classes, weights_regularizer=l2_reg_fn,
                                                       # weights_initializer=tf.truncated_normal_initializer(1,2),
                                                       activation_fn=None,reuse=reuse, biases_regularizer=l2_reg_fn,
                                                       scope='fc_out')

            if (is_test is not True):
                tf.summary.histogram('act1', layer1)
                tf.summary.histogram('act2', layer2)
                tf.summary.histogram('act3', layer3)
                tf.summary.histogram('logits', logits)

        return logits


    train_logits = nn_model(tf_train_dataset, name='train_nn')
    train_prediction = tf.nn.softmax(train_logits)

    with tf.variable_scope("xent"):
        # nn_variable_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="NN model")
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = np.sum(reg_losses)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=train_logits))+reg_loss

    with tf.variable_scope("optimizer"):
        global_step = tf.Variable(0)
        # learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, 300, 0.3, staircase=False)
        learning_rate = 0.5
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    def accuracy(predictions, labels, name='accuracy'):
        with tf.variable_scope(name):
            correct_predictions = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(labels, axis=1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            return accuracy

    tf_training_accuracy = accuracy(train_prediction, tf_train_labels, name='train_accuracy')


    # validation prediction
    tf_valid_dataset = tf.placeholder(tf.float32, shape=(valid_size, num_inputs), name='valid_in')
    tf_valid_labels = tf.placeholder(tf.float32, shape=(valid_size, num_classes), name='valid_labels')
    valid_logits = nn_model(tf_valid_dataset, name='train_nn', reuse=True)
    valid_prediction = tf.nn.softmax(valid_logits)
    valid_accuracy = accuracy(valid_prediction, tf_valid_labels, name='validation_accuracy')
    # summarize a couple of things
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("train_accuracy", tf_training_accuracy)
    tf.summary.scalar("validation_accuracy", valid_accuracy)
    tf.summary.scalar("learning rate", learning_rate)

    tf.summary.tensor_summary("train_prediction", train_prediction)
    tf.summary.tensor_summary("train_batch", tf_train_dataset)
    tf.summary.tensor_summary("train_labels", tf_train_labels)

    tf.summary.tensor_summary("valid_prediction", valid_prediction)
    tf.summary.tensor_summary("valid_batch", tf_valid_dataset)
    tf.summary.tensor_summary("valid_labels", tf_valid_labels)
    # summarize weights and biases
    all_vars= tf.global_variables()
    def get_var(name):
        for i in range(len(all_vars)):
            if all_vars[i].name.startswith(name):
                return all_vars[i]
        return None
    fc1_weight = get_var('train_nn/fc1/weights')
    fc1_biases = get_var('train_nn/fc1/biases')
    fc2_weight = get_var('train_nn/fc2/weights')
    fc2_biases = get_var('train_nn/fc2/biases')
    fc3_weight = get_var('train_nn/fc3/weights')
    fc3_biases = get_var('train_nn/fc3/biases')
    fc_out_weight = get_var('train_nn/fc_out/weights')
    fc_out_biases = get_var('train_nn/fc_out/biases')
    tf.summary.histogram("fc1_weight", fc1_weight)
    tf.summary.histogram("fc1_biases", fc1_biases)
    tf.summary.histogram("fc2_weight", fc2_weight)
    tf.summary.histogram("fc2_biases", fc2_biases)
    tf.summary.histogram("fc3_weight", fc3_weight)
    tf.summary.histogram("fc3_biases", fc3_biases)
    tf.summary.histogram("fc_out_weight", fc_out_weight)
    tf.summary.histogram("fc_out_biases", fc_out_biases)

    # inference phase
    tf_test_data = tf.placeholder(tf.float32, shape=(1, num_inputs), name='test_in')
    test_logits = nn_model(tf_test_data, name='train_nn', reuse=True, is_test=True)
    test_prediction = tf.nn.softmax(test_logits)

    merged_summary = tf.summary.merge_all()


class BatchGenerator(object):
    def __init__(self, data_set, labels, batch_size):
        self._data_set = data_set
        self._labels = labels
        self._data_size = data_set.shape[0]
        self._batch_size = batch_size
        self._segment = self._data_size // batch_size
        self.last_index = 0

    def next(self):

        if ((self.last_index+1)*self._batch_size > self._data_size):
            data1 = self._data_set[self.last_index * self._batch_size:,:]
            data2 = self._data_set[:((self.last_index+1)*self._batch_size)%self._data_size, :]
            labels1 = self._labels[self.last_index * self._batch_size:,:]
            labels2 = self._labels[:((self.last_index+1)*self._batch_size)%self._data_size, :]
            batch_data = np.concatenate((data1, data2), axis=0)
            batch_labels = np.concatenate((labels1, labels2), axis=0)
        else:
            batch_data = self._data_set[self.last_index * self._batch_size:(self.last_index + 1) * self._batch_size, :]
            batch_labels = self._labels[self.last_index * self._batch_size:(self.last_index + 1) * self._batch_size, :]

        self.last_index = (self.last_index+1) % (self._segment+1)
        return batch_data, batch_labels

def normalize_input(dataset):

    # outputs a dataset with features in range of (-0.5,0.5)
    min = np.array([0, 0])
    max = np.array([249, 99])
    min_array = np.tile(min, (len(dataset), 1))
    max_array = np.tile(max, (len(dataset), 1))
    mean_array = (min_array+max_array)/2
    dataset_normalized = (dataset - mean_array)/(max_array-min_array)

    return dataset_normalized


def train(session, dataset, labels, gen_num, max_iter=1000, batch_size=32):
    # 90% of the data is training data and 10% is validation data
    # split the dataset intp train and validation sets
    boundry_index = len(dataset) * 9 // 10
    train_dataset = dataset[:boundry_index]
    train_labels = labels[:boundry_index]
    valid_dataset = dataset[boundry_index:]
    valid_labels = labels[boundry_index:]

    train_dataset_norm = normalize_input(np.array(train_dataset))
    valid_dataset_norm = normalize_input(np.array(valid_dataset))
    print("-"*40)
    print(train_dataset_norm)
    print("-"*40)
    print("-"*40)
    print(valid_dataset_norm)
    print("-"*40)


    writer_name = os.path.join(summary_dir, str(gen_num))
    writer = tf.summary.FileWriter(writer_name)
    writer.add_graph(graph)

    # print("train_data_set")
    # print("-"*30)
    # for i, ind in enumerate(train_dataset):
    #     print("%10d %10s->%20s->%10s" %(i, ind, ind.fitness.values[0], train_labels[i]))
    # print("valid_data_set")
    # print("-"*30)
    # for i, ind in enumerate(valid_dataset):
    #     print("%10d %10s->%20s->%10s" %(i, ind, ind.fitness.values[0], valid_labels[i]))

    print("train_dataset bad_samples/total ratio : %d/%d" %(np.sum(train_labels, axis=0)[0], train_labels.shape[0]))
    print("valid_dataset bad_samples/total ratio : %d/%d" %(np.sum(valid_labels, axis=0)[0], valid_labels.shape[0]))

    data_set_size = len(train_dataset)
    num_segments = data_set_size // batch_size
    batch_generator = BatchGenerator(np.array(train_dataset_norm), train_labels, batch_size)
    print ("training the model with new dataset ....")

    tf.global_variables_initializer().run()
    print("Initialized")
    for iter in range(max_iter):
        batch_data, batch_labels = batch_generator.next()
        feed_dict = {tf_train_dataset   :batch_data,
                     tf_train_labels    :batch_labels,}

        _, l, predictions, train_acc = session.run([optimizer, loss, train_prediction, tf_training_accuracy], feed_dict=feed_dict)

        if iter%5 == 0:
            feed_dict[tf_valid_dataset] = valid_dataset_norm
            feed_dict[tf_valid_labels] = valid_labels
            valid_predict, valid_acc, s = session.run([valid_prediction, valid_accuracy, merged_summary], feed_dict=feed_dict)
            writer.add_summary(s, iter)
            print("loss at iter %d: %f" %(iter, l))
            print("Minibatch train and validation accuracy: %.2f%%, %.2f%%" %(train_acc*100, valid_acc*100))

threshold = 0.5
def is_bad(prediction):
    return prediction[0] > threshold

def sample_trained_model(session, toolbox):
    print("sampling with trained model ....")
    num_new_samples = 0
    max_num_new_samples = pop_size - top - bot
    new_samples = list()
    bad_samples = list()
    bad_sample_count = 0
    while num_new_samples < max_num_new_samples:

        ind = toolbox.individual()
        if ind in evaluated_samples: # only sample individuals that are new
            continue
        np_ind = np.array(ind).reshape(1, num_inputs)
        np_ind = normalize_input(np_ind)
        ind_prediction = test_prediction.eval(feed_dict={tf_test_data: np_ind})[0]
        if is_bad(ind_prediction):
            print("bad %3d ind:%s" %(bad_sample_count, ind))
            print("prediction=%40s , evaluation=%10s, is considered as bad" %(ind_prediction, toolbox.evaluate(ind)))
            bad_samples.append(ind)
            bad_sample_count += 1
            if bad_sample_count == 20000:
                break
            continue
        print("good %3d ind:%s" %(num_new_samples, ind))
        print("prediction=%40s , evaluation=%10s, is considered as not bad" %(ind_prediction, toolbox.evaluate(ind)))
        new_samples.append(ind)
        num_new_samples += 1

    return new_samples


def shuffle_dataset(dataset, labels):
    """
    :param dataset: this is suppose to be a list of individuals
    :param labels:  this is a numpy array
    :return:
    """
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = [dataset[i] for i in permutation]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def split_pipulation(population):
    good_samples    = sorted(population, key=lambda x: x.fitness.values[0])[:top]
    bad_samples     = sorted(population, key=lambda x: x.fitness.values[0])[-bot:]
    # good_samples_values = [ind.fitness.values[0] for ind in good_samples]
    # bad_samples_values = [ind.fitness.values[0] for ind in bad_samples]


    dataset = good_samples+bad_samples
    labels_bool = np.concatenate((np.zeros(shape=(top, 1), dtype=np.float32), np.ones(shape=(bot, 1), dtype=np.float32)), axis=0)
    labels = np.concatenate((labels_bool, 1-labels_bool), axis=1)
    print("-"*30)
    for i, ind in enumerate(dataset):
        print("%10d   %10s->%20s->%10s->%10s" %(i, ind, ind.fitness.values[0], labels_bool[i], labels[i]))

    dataset, labels = shuffle_dataset(dataset, labels)
    return dataset, labels




class VisualizeTraining(object):
    def __init__(self):
        self.rows = 4
        self.cols = 5
        self.cur_dataset_sub_id = 1
        self.cur_classification_sub_id = 1
        self.dataset_fig = plt.figure(1)
        self.dataset_fig.suptitle('Dataset of good and bad data')
        self.classification_fig = plt.figure(2)
        # self.classification_fig.suptitle('Bad regions learned from nn')

        self.threshold=0.5 #threshold for making a decsion between good and bad regions


        self.bw_mat =    self._load_array("./genome/sweeps/bw.array")
        self.gain_mat =  self._load_array("./genome/sweeps/gain.array")
        self.Ibias_mat = self._load_array("./genome/sweeps/ibias.array")
        self.mul_vec =   self._load_array("./genome/sweeps/mul_vec.array")
        self.res_vec =   self._load_array("./genome/sweeps/res_vec.array")

        self.cost_mat = [[self._cost_fc(self.Ibias_mat[mul_idx][res_idx],
                                        self.gain_mat[mul_idx][res_idx],
                                        self.bw_mat[mul_idx][res_idx]) for res_idx in \
                          range(len(self.res_vec))] for mul_idx in range(len(self.mul_vec))]

        self.cost_min = np.min(np.min(self.cost_mat))
        self.cost_max = np.max(np.max(self.cost_mat))

    def _cost_fc(self, ibias_cur, gain_cur, bw_cur):
        bw_min = 1e9
        gain_min = 3
        bias_max = 1e-3

        cost = 0
        if bw_cur < bw_min:
            cost += abs(bw_cur/bw_min - 1.0)
        if gain_cur < gain_min:
            cost += abs(gain_cur/gain_min - 1.0)
        cost += abs(ibias_cur/bias_max)/10

        return cost

    def _load_array(self, fname):
        with open(fname, "rb") as f:
            arr = np.load(f)
        return arr


    def visualize_dataset(self, dataset, labels):
        if self.cur_dataset_sub_id > (self.rows*self.cols):
            print("figure doesn't have that many rows and cols for illustrating %d" %(self.cur_dataset_sub_id))
        else:
            ax = self.dataset_fig.add_subplot(self.rows, self.cols, self.cur_dataset_sub_id)
            # plot background cost function
            mappable = ax.pcolormesh(self.cost_mat, cmap='autumn', vmin=self.cost_min, vmax=self.cost_max)
            plt.colorbar(mappable,ax=ax)

            positive = [dataset[i] for i in range(len(dataset)) if np.array_equal(labels[i], np.array([1, 0]))]
            negative = [dataset[i] for i in range(len(dataset)) if np.array_equal(labels[i], np.array([0, 1]))]

            positive_x = [ind[0] for ind in positive]
            positive_y = [ind[1] for ind in positive]
            negative_x = [ind[0] for ind in negative]
            negative_y = [ind[1] for ind in negative]
            ax.scatter(positive_x, positive_y, marker = 'x', color='b')
            ax.scatter(negative_x, negative_y, marker = '*', color='g')
            ax.axis([0, 250, 0, 100])
            self.cur_dataset_sub_id+=1

    def is_bad(self, prediction):
        return prediction[0] > self.threshold

    def visualize_decision_bnd(self, session, toolbox):
        if self.cur_classification_sub_id > (self.rows*self.cols):
            print("figure doesn't have that many rows and cols for illustrating %d" %(self.cur_classification_sub_id))
        else:
            print("visualizing decsion boundry")
            max_num_bad_samples = 1000
            bad_samples = list()
            bad_sample_count = 0
            while bad_sample_count < max_num_bad_samples:
                ind = toolbox.individual()
                np_ind = np.array(ind).reshape(1, num_inputs)
                np_ind = normalize_input(np_ind)
                ind_prediction = test_prediction.eval(feed_dict={tf_test_data: np_ind})[0]
                if self.is_bad(ind_prediction):
                    bad_samples.append(ind)
                    bad_sample_count += 1
            ax = self.classification_fig.add_subplot(self.rows, self.cols, self.cur_classification_sub_id)
            # plot background cost function
            mappable = ax.pcolormesh(self.cost_mat, cmap='autumn', vmin=self.cost_min, vmax=self.cost_max)
            plt.colorbar(mappable, ax=ax)

            x = [ind[0] for ind in bad_samples]
            y = [ind[1] for ind in bad_samples]
            ax.scatter(x, y, marker = 'x', color='b')
            ax.axis([0, 250, 0, 100])
            self.cur_classification_sub_id+=1

def optimize(population, toolbox, top, bot, stats=None, halloffame=None, verbose=__debug__):

    population_size = len(population)
    sample_size = top+bot
    new_samples_size = population_size - sample_size

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'pop'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    results = toolbox.map(toolbox.evaluate, invalid_ind)
    evaluated_samples.append(population)
    for ind, res in zip(invalid_ind, results):
        ind.fitness.values = (res[0],)
        ind.bw =    res[1]
        ind.gain =  res[2]
        ind.ibias = res[3]

    if halloffame is not None:
        halloffame.update(population)

    good_samples    = sorted(population, key=lambda x: x.fitness.values[0])[:top]
    record = stats.compile(good_samples) if stats is not None else {}
    # print (population)
    logbook.record(gen=0, nevals=len(invalid_ind), pop=good_samples.copy(),  **record)
    if verbose:
        print(logbook.stream)

    vs_data = VisualizeTraining()

    # Begin the generational process
    with tf.Session(graph=graph) as session:
        for gen in range(1, ngen+1):
            # make dataset for training
            dataset, labels = split_pipulation(population)
            # plot positive and negative samples
            vs_data.visualize_dataset(dataset, labels)
            # train the model
            train(session, dataset, labels, gen, batch_size=batch_size)
            # plot the decision boundry for better understanding of what's happening
            vs_data.visualize_decision_bnd(session, toolbox)
            # generate new sample after learning the model
            invalid_ind = sample_trained_model(session, toolbox)
            # update the population with new sample and evaluate the newbies
            results = toolbox.map(toolbox.evaluate, invalid_ind)
            evaluated_samples.append(invalid_ind)
            for ind, res in zip(invalid_ind, results):
                ind.fitness.values = (res[0],)
                ind.bw =    res[1]
                ind.gain =  res[2]
                ind.ibias = res[3]

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(invalid_ind)

            population[:] = dataset+invalid_ind

            good_samples    = sorted(population, key=lambda x: x.fitness.values[0])[:top]
            # Update the statistics with the new population
            record = stats.compile(good_samples) if stats is not None else {}
            print(record)
            logbook.record(gen=gen, nevals=len(invalid_ind), pop=good_samples.copy(), **record)
            if verbose:
                print(logbook.stream)

    return population, logbook

## optimization core
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax,
               gain=float, bw=float, ibias=float)

toolbox = base.Toolbox()

stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
stats_res = tools.Statistics(key=lambda ind: eval_core.res_vec[ind[0]])
stats_mul = tools.Statistics(key=lambda ind: eval_core.mul_vec[ind[1]])
stats_gain = tools.Statistics(key=lambda ind: ind.gain)
stats_bw = tools.Statistics(key=lambda ind: ind.bw)
stats_ibias = tools.Statistics(key=lambda ind: ind.ibias)
mStat = tools.MultiStatistics(fit=stats_fit, res=stats_res, mul=stats_mul, gain=stats_gain,
                              bw=stats_bw, ibias=stats_ibias)
mStat.register("avg", np.mean)
mStat.register("std", np.std)
mStat.register("min", np.min)
mStat.register("max", np.max)

history = tools.History()

toolbox.register("individual", init_inividual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate_individual)

def main():

    random.seed(10)
    np.random.seed(10)
    pop = toolbox.population(n=pop_size)
    history.update(pop)
    pop, logbook = optimize(pop, toolbox, top, bot, verbose=False, stats=mStat)
    import pprint
    # pprint.pprint(logbook)
    print_best_ind(pop)
    gen = logbook.select('gen')
    fit_avg, fit_min, fit_max, fit_std = np.array(logbook.chapters["fit"].select('avg', 'min', 'max', 'std'))
    bw_avg, bw_min, bw_max, bw_std = np.array(logbook.chapters["bw"].select('avg', 'min', 'max', 'std'))
    gain_avg, gain_min, gain_max, gain_std = np.array(logbook.chapters["gain"].select('avg', 'min', 'max', 'std'))
    ibias_avg, ibias_min, ibias_max, ibias_std = np.array(logbook.chapters["ibias"].select('avg', 'min', 'max', 'std'))
    mul_avg, mul_min, mul_max, mul_std = np.array(logbook.chapters["mul"].select('avg', 'min', 'max', 'std'))
    res_avg, res_min, res_max, res_std = np.array(logbook.chapters["res"].select('avg', 'min', 'max', 'std'))

    # import pdb
    # pdb.set_trace()
    good_samples = logbook.select('pop')

    fig = plt.figure()
    ax = fig.add_subplot(321)
    ax.plot(gen, fit_avg)
    ax.plot(gen, fit_min, '--', color='r')
    ax.plot(gen, fit_max, '--', color='r')
    ax.plot(gen, fit_avg+fit_std, ':', color='m')
    ax.plot(gen, fit_avg-fit_std, ':', color='m')
    ax.set_ylabel('cost function')

    ax = fig.add_subplot(322)
    ax.plot(gen, bw_avg)
    ax.plot(gen, bw_min, '--', color='r')
    ax.plot(gen, bw_max, '--', color='r')
    ax.plot(gen, bw_avg+bw_std, ':', color='m')
    ax.plot(gen, bw_avg-bw_std, ':', color='m')
    ax.set_ylabel('bandwidth')

    ax = fig.add_subplot(323)
    ax.plot(gen, gain_avg)
    ax.plot(gen, gain_min, '--', color='r')
    ax.plot(gen, gain_max, '--', color='r')
    ax.plot(gen, gain_avg+gain_std, ':', color='m')
    ax.plot(gen, gain_avg-gain_std, ':', color='m')
    ax.set_ylabel('gain')

    ax = fig.add_subplot(324)
    ax.plot(gen, ibias_avg)
    ax.plot(gen, ibias_min, '--', color='r')
    ax.plot(gen, ibias_max, '--', color='r')
    ax.plot(gen, ibias_avg+ibias_std, ':', color='m')
    ax.plot(gen, ibias_avg-ibias_std, ':', color='m')
    ax.set_ylabel('ibias')

    ax = fig.add_subplot(325)
    ax.plot(gen, mul_avg)
    ax.plot(gen, mul_min, '--', color='r')
    ax.plot(gen, mul_max, '--', color='r')
    ax.plot(gen, mul_avg+mul_std, ':', color='m')
    ax.plot(gen, mul_avg-mul_std, ':', color='m')
    ax.set_ylabel('mul')
    ax.set_xlabel('number of generation')

    ax = fig.add_subplot(326)
    ax.plot(gen, res_avg)
    ax.plot(gen, res_min, '--', color='r')
    ax.plot(gen, res_max, '--', color='r')
    ax.plot(gen, res_avg+res_std, ':', color='m')
    ax.plot(gen, res_avg-res_std, ':', color='m')
    ax.set_ylabel('res')
    ax.set_xlabel('number of generation')


    fig.tight_layout()

    ## helper function for demonstration

    def cost_fc(ibias_cur, gain_cur, bw_cur):
        bw_min = 1e9
        gain_min = 3
        bias_max = 1e-3

        cost = 0
        if bw_cur < bw_min:
            cost += abs(bw_cur/bw_min - 1.0)
        if gain_cur < gain_min:
            cost += abs(gain_cur/gain_min - 1.0)
        cost += abs(ibias_cur/bias_max)/10

        return cost

    def load_array(fname):
        with open(fname, "rb") as f:
            arr = np.load(f)
        return arr

    bw_mat =    load_array("./genome/sweeps/bw.array")
    gain_mat =  load_array("./genome/sweeps/gain.array")
    Ibias_mat = load_array("./genome/sweeps/ibias.array")
    mul_vec =   load_array("./genome/sweeps/mul_vec.array")
    res_vec =   load_array("./genome/sweeps/res_vec.array")

    cost_mat = [[cost_fc(Ibias_mat[mul_idx][res_idx],
                         gain_mat[mul_idx][res_idx],
                         bw_mat[mul_idx][res_idx]) for res_idx in \
                 range(len(res_vec))] for mul_idx in range(len(mul_vec))]

    cost_min = np.min(np.min(cost_mat))
    cost_max = np.max(np.max(cost_mat))

    fig = plt.figure()
    for i in range(min(ngen, 20)):
        ax = fig.add_subplot(4, 5, i+1)
        mappable = ax.pcolormesh(cost_mat, cmap='OrRd', vmin=cost_min, vmax=cost_max)
        plt.colorbar(mappable)
        if i>=15:
            ax.set_xlabel('res_idx')
        if i%5==0:
            ax.set_ylabel('mul_idx')
        # good_samples    = sorted(populations[i], key=lambda x: x.fitness.values[0])[:top]
        x = [ind[0] for ind in good_samples[i]]
        y = [ind[1] for ind in good_samples[i]]
        ax.scatter(x, y, marker = 'x')
        ax.axis([0, 250, 0, 100])

    # fig.tight_layout()

    import pickle
    with open("./genetic_nn/log/cs_logbook.pickle", 'wb') as f:
        pickle.dump(logbook, f)


if __name__ == '__main__':
    main()
