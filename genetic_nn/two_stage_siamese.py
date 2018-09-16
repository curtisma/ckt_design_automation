import numpy as np
import tensorflow as tf
import math
import os
import sys
import matplotlib.pyplot as plt
import random
import pickle

from genetic_nn.util import BatchGenerator

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
n_new_samples = 20
num_designs = 2
num_params_per_design = 7
num_classes = 2
num_nn_features = num_designs * num_params_per_design
valid_frac = 0.2
max_n_retraining = 20

k_top = 20 #during training only consider comparison between k_top ones and the others
ref_dsn_idx = 20 #during inference compare new randomly generated samples with this design in the sorted dataset
max_data_set_size = 800

# training settings
num_epochs = 50
batch_size = 16
display_step = 10
ckpt_step = 10

summary_dir = 'genetic_nn/summary'


# nn hyper parameters
num_features = 20
feat_ext_dim_list = [num_params_per_design, 20, num_features]
compare_nn_dim_list = [2*num_features, num_classes]

learning_rate = 0.03
decay_steps = 10000
decay_rate = 0.5

l2_reg_scale = 0.003
DROP_OUT_PROB = 0.5

graph = tf.Graph()

with graph.as_default():
    input1 = tf.placeholder(tf.float32, shape=[None, num_params_per_design], name='in1')
    input2 = tf.placeholder(tf.float32, shape=[None, num_params_per_design], name='in2')
    true_labels = tf.placeholder(tf.float32, shape=[None, num_classes], name='labels')

    with tf.variable_scope('normalizer'):
        mu = tf.Variable(tf.zeros([num_params_per_design], dtype=tf.float32), name='mu', trainable=False)
        std = tf.Variable(tf.zeros([num_params_per_design], dtype=tf.float32), name='std', trainable=False)
        input1_norm = (input1 - mu) / (std + 1e-6)
        input2_norm = (input2 - mu) / (std + 1e-6)

    def feature_extraction_model(input_data, name='feature_model', reuse=False):
        layer = input_data
        with tf.variable_scope(name):
            for i, layer_dim in enumerate(feat_ext_dim_list[1:]):
                layer = tf.contrib.layers.fully_connected(layer, layer_dim, reuse=reuse, scope='feat_fc'+str(i))

        return layer

    features1 = feature_extraction_model(input1_norm, name='feat_model', reuse=False)
    features2 = feature_extraction_model(input2_norm, name='feat_model', reuse=True)


    def sym_fc_layer(input_data, layer_dim, activation_fn = None, reuse=False, scope='sym_fc'):
        assert input_data.shape[1]%2==0

        with tf.variable_scope(scope):
            weight_elements = tf.get_variable(name = 'W', shape=[input_data.shape[1]//2, layer_dim],
                                              initializer=tf.random_normal_initializer)
            bias_elements = tf.get_variable(name='b', shape=[layer_dim//2],
                                           initializer=tf.zeros_initializer)

            Weight = tf.concat([weight_elements, weight_elements[::-1, ::-1]], axis=0, name='Weights')
            Bias = tf.concat([bias_elements, bias_elements[::-1]], axis=0, name='Bias')

            out = tf.add(tf.matmul(input_data, Weight), Bias)
            if activation_fn == None:
                pass
            elif activation_fn == 'Relu':
                out = tf.nn.relu(out)
            elif activation_fn == 'tanh':
                out = tf.nn.tanh(out)
            else:
                print('activation does not exist')

            return out, Weight


    def comparison_model(input_data, name='compare_model', reuse=False):
        layer = input_data
        w_list = []
        with tf.variable_scope(name):
            for i, layer_dim in enumerate(compare_nn_dim_list[1:-1]):
                layer, w = sym_fc_layer(layer, layer_dim, activation_fn='Relu', reuse=reuse, scope='cmp_fc'+str(i))
                w_list.append(w)

            logits, w = sym_fc_layer(layer, num_classes, reuse=reuse, scope='fc_out')
            w_list.append(w)

        return logits, w_list

    input_features = tf.concat([features1, features2[:, ::-1]], axis=1)
    out_logits, w_list = comparison_model(input_features, 'compare_model', reuse=False)
    out_predictions = tf.nn.softmax(out_logits)

    with tf.variable_scope("loss"):
        likelihoods = tf.nn.softmax_cross_entropy_with_logits(labels=true_labels,
                                                              logits=out_logits)
        loss = tf.reduce_mean(likelihoods)
    with tf.variable_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer().minimize(loss)
        # optimizer = tf.train.GradientDescentOptimizer().minimize(loss)

    def accuracy(predictions, labels, name='accuracy'):
        with tf.variable_scope(name):
            correct_predictions = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(labels, axis=1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            return accuracy

    accuracy = accuracy(out_predictions, true_labels)

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary = tf.summary.merge_all()


    writer = tf.summary.FileWriter(summary_dir)
    writer.add_graph(graph)

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
    nn_input1, nn_input2 , nn_labels = [], [], []
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
    # i = k_top
        for j in range(i+1, len(sorted_dataset)):
            # if j == k_top: continue
            if random.random() < 0.5:
                nn_input1.append(list(sorted_dataset[i,:]))
                nn_input2.append(list(sorted_dataset[j,:]))
                cost_arr.append([sorted_cost[i], sorted_cost[j]])
                label = 1 if (sorted_cost[i] > sorted_cost[j]) else 0
            else:
                nn_input1.append(list(sorted_dataset[j,:]))
                nn_input2.append(list(sorted_dataset[i,:]))
                cost_arr.append([sorted_cost[j], sorted_cost[i]])
                label = 0 if (sorted_cost[i] > sorted_cost[j]) else 1

            if j < k_top:
                adjustment_weights.append(weights_good)
            else:
                adjustment_weights.append(weights_bad)
            category.append(label)


    nn_labels = np.zeros((len(category), num_classes))
    nn_labels[np.arange(len(category)), category] = 1
    return np.array(nn_input1), np.array(nn_input2), np.array(nn_labels),\
           np.array(cost_arr), np.array(adjustment_weights)

def train(session, dataset, cost, writer, num_epochs=10, batch_size=128):
    all_vars = tf.global_variables()
    saver = tf.train.Saver(all_vars)

    nn_input1, nn_input2, nn_labels, cost_arr, adjustment_weights = combine(dataset, cost)

    # print("[Debug_data] design1, design2, costs, label")
    #
    # for i in range(len(nn_labels)):
    #     design1 = nn_input1[i]
    #     design2 = nn_input2[i]
    #     costs = cost_arr[i,:]
    #     label = nn_labels[i,:]
    #     print("[Debug_data] {} -> {} -> {} -> {}".format(design1, design2, costs, label))

    # nn_dataset, nn_labels = shuffle_dataset(nn_dataset, nn_labels) # Do we really need to shuffle? yes we do :)
    permutation = np.random.permutation(nn_labels.shape[0])
    nn_input1 = nn_input1[permutation]
    nn_input2 = nn_input2[permutation]
    nn_labels = nn_labels[permutation]
    cost_arr = cost_arr[permutation]
    adjustment_weights = adjustment_weights[permutation]

    # print("[Debug_shuffle] sample -> label ")
    # for i in range(len(nn_dataset)):
    #     print("[Debug_shuffle] {} -> {} " .format(nn_dataset[i], nn_labels[i]))

    boundry_index = nn_labels.shape[0] - int(nn_labels.shape[0]*valid_frac)
    train_input1 = nn_input1[:boundry_index]
    train_input2 = nn_input2[:boundry_index]
    train_labels = nn_labels[:boundry_index]
    valid_input1 = nn_input1[boundry_index:]
    valid_input2 = nn_input2[boundry_index:]
    valid_labels = nn_labels[boundry_index:]

    train_weights = adjustment_weights[:boundry_index]
    valid_weights = adjustment_weights[boundry_index:]

    # find the mean and std of dataset for normalizing

    train_mean = np.mean(np.concatenate([train_input1, train_input2], axis=0), axis=0)
    train_std = np.std(np.concatenate([train_input1, train_input2], axis=0), axis=0)
    # print(train_mean)
    # print(train_std)

    print("[info] dataset size:%d" %len(dataset))
    print("[info] combine size:%d" %len(nn_labels))
    print("[info] train_dataset: positive_samples/total ratio : %d/%d" %(np.sum(train_labels, axis=0)[0], train_labels.shape[0]))
    print("[info] valid_dataset: positive_samples/total ratio : %d/%d" %(np.sum(valid_labels, axis=0)[0], valid_labels.shape[0]))

    # although we have shuffled the training dataset once, there is going to be another shuffle inside the batch generator
    batch_generator = BatchGenerator(len(train_labels), batch_size)
    print("[info] training the model with dataset ....")

    total_n_batches = int(len(train_labels) // batch_size)
    print("[info] number of total batches: %d" %total_n_batches)
    print(30*"-")

    tf.global_variables_initializer().run()
    mu.assign(train_mean).op.run()
    std.assign(train_std).op.run()

    for epoch in range(num_epochs):

        # w_list_print, = session.run([w_list], feed_dict={})
        # for i, w in enumerate(w_list_print):
        #     print("w[%d]:" %i)
        #     print("{}".format(w))

        avg_loss = 0.
        avg_train_acc = 0.
        avg_valid_acc = 0.
        feed_dict = {}
        for iter in range(total_n_batches):
            index = batch_generator.next()
            batch_input1, batch_input2 = train_input1[index], train_input2[index]
            batch_labels, batch_weights = train_labels[index], train_weights[index]
            feed_dict = {input1         :batch_input1,
                         input2         :batch_input2,
                         true_labels    :batch_labels}

            _, l, train_acc = session.run([optimizer, loss, accuracy], feed_dict=feed_dict)

            feed_dict = {input1         :valid_input1,
                         input2         :valid_input2,
                         true_labels    :valid_labels}

            valid_acc, = session.run([accuracy], feed_dict=feed_dict)
            avg_loss += l / total_n_batches
            avg_train_acc += train_acc / total_n_batches
            avg_valid_acc += valid_acc / total_n_batches


        s = session.run(merged_summary, feed_dict=feed_dict)

        writer.add_summary(s, epoch)

        # feed_dict = {
        #     input1:     np.array([[1,2,3,4,5,6,7],[10,11,12,13,14,15,16]]),
        #     input2:     np.array([[10,11,12,13,14,15,16],[1,2,3,4,5,6,7]]),
        # }

        # print(30*"-")
        # features_1, features_2, features, logit = session.run([features1, features2, input_features, out_logits], feed_dict=feed_dict)
        # print("features1")
        # print("{}".format(features_1))
        # print("features2")
        # print("{}".format(features_2))
        # print("features")
        # print("{}".format(features))
        # print(30*"-")
        # w_list_print, = session.run([w_list], feed_dict={})
        # for i, w in enumerate(w_list_print):
        #     print("w[%d]:" %i)
        #     print("{}".format(w))
        #
        # print(40*"-")
        # print(np.matmul(features, w_list_print[0]))
        # print("logit", logit)

        # exit()
        if epoch % ckpt_step == 0:
            saver.save(session, 'genetic_nn/checkpoint/checkpoint.ckpt')
            dict_to_save = dict(dataset=dataset, cost=cost)
            with open('genetic_nn/checkpoint/data.pkl', 'wb') as f:
                pickle.dump(dict_to_save, f)
        if epoch % display_step == 0:
            print("[epoch %d] loss: %f" %(epoch, avg_loss))
            print("train_acc = %.2f%%, valid_acc = %.2f%%" %(avg_train_acc*100, avg_valid_acc*100))

    # test_valid_training(session,
    #                     [train_input1, train_input2, cost_arr[:boundry_index],train_labels],
    #                     [valid_input1, valid_input2, cost_arr[boundry_index:],valid_labels])

def run_model(session, design_pool, cost_pool, m_samples, ref_dsn, max_iter=1000):
    print(30*"-")
    print("[info] running model ... ")
    cnt = 0

    better_dsns = []
    better_dsns_costs, better_dsns_bw, better_dsns_gain, better_dsns_ibias = [], [], [], []
    better_dsns_pred = []
    for i in range(ref_dsn_idx):
        print("[Debug_test] dataset: {} -> {}".format(design_pool[i], cost_pool[i]))
    print("[debug] ref design {} with ref cost {}".format(design_pool[ref_dsn_idx-1], cost_pool[ref_dsn_idx-1]))
    while (True):
        cnt += 1
        new_designs, _ = generate_data_set(1, evaluate=False)
        new_design = list(new_designs[0])
        if any((new_design == row).all() for row in design_pool):
            # if design is already in the design pool skip ...
            # print("[debug] design {} already exists".format(new_design))
            continue

        nn_input1 = np.array(new_design)
        nn_input2 = design_pool[ref_dsn_idx-1]

        feed_dict = {input1: nn_input1[None, :],
                     input2: nn_input2[None, :],
                     }
        prediction,  = session.run([out_predictions], feed_dict=feed_dict)

        if np.argmax(prediction) == 0:
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


def test_valid_training(session, train_list, valid_list):

    train_input1, train_input2, train_cost, train_labels = train_list
    valid_input1, valid_input2, valid_cost, valid_labels = valid_list

    feed_dict = {input1:    train_input1,
                 input2:    train_input2,
                 }

    train_predictions, = session.run([out_predictions], feed_dict=feed_dict)
    train_cnt = 0
    for i in range(len(train_labels)):
        design1 = train_input1[i,:]
        design2 = train_input2[i,:]
        costs = train_cost[i]
        true_label = 0 if costs[0] < costs[1] else 1
        prediction = train_predictions[i, :]
        correct_flag = 0
        if np.argmax(prediction) == true_label:
            # this is a correct prediction (caught design)
            correct_flag = 1
        print("[Debug_train] {} -> {} -> {} -> {} -> {} -> {} ".format(list(design1), list(design2), costs,
                                                                       true_label, prediction, correct_flag))

    feed_dict = {input1:    valid_input1,
                 input2:    valid_input2,
                 }

    valid_predictions, = session.run([out_predictions], feed_dict=feed_dict)
    valid_cnt = 0
    for i in range(len(valid_labels)):
        design1 = valid_input1[i,:]
        design2 = valid_input2[i,:]
        costs = valid_cost[i]
        true_label = 0 if costs[0] < costs[1] else 1
        prediction = valid_predictions[i, :]
        correct_flag = 0
        if np.argmax(prediction) == true_label:
            # this is a correct prediction (caught design)
            correct_flag = 1
        print("[Debug_valid] {} -> {} -> {} -> {} -> {} -> {} ".format(list(design1), list(design2), costs,
                                                                       true_label, prediction, correct_flag))




def test_swaping(session, dataset, cost):

    sorted_indices = sorted(range(len(dataset)), key=lambda x: cost[x])
    sorted_design_pool = dataset[sorted_indices]
    sorted_cost_pool = cost[sorted_indices]

    design1 = sorted_design_pool[ref_dsn_idx]
    cost1 = sorted_cost_pool[ref_dsn_idx]
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

        # print(np.array([list(design1),list(design2)]))
        # print(np.array([list(design2),list(design1)]))
        feed_dict = {input1         :np.array([list(design1),list(design2)]),
                     input2         :np.array([list(design2),list(design1)]),
                     }
        predictions, = session.run([out_predictions], feed_dict=feed_dict)
        print("[Debug_swapping] {}".format(predictions))
        if np.argmax(predictions[0]) == np.argmax(predictions[1]):
            print("[Debug_swapping] found inconsistency in design: {} -> {}".format(list(design2), cost2))
            print("[Debug_swapping] [d1,d2] -> {}".format(predictions[0]))
            print("[Debug_swapping] [d2,d1] -> {}".format(predictions[1]))
            cnt += 1

    print("[Debug_swapping] cnt=%d" %cnt)

def test_model(session, training_dataset, training_cost):

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

    nn_input1 = np.repeat(sorted_training_designs[ref_dsn_idx-1][None,:], len(new_designs), axis=0)

    feed_dict = {input1:    nn_input1,
                 input2:    new_designs,}

    predictions, = session.run([out_predictions], feed_dict=feed_dict)

    for i in range(ref_dsn_idx):
        print("[Debug_test] dataset: {} -> {}".format(sorted_training_designs[i], sorted_training_cost[i]))
    print("[Debug_test] ref design in dataset {} cost {}".format(sorted_training_designs[ref_dsn_idx-1],
                                                                 sorted_training_cost[ref_dsn_idx-1]))
    print("[Debug_test] design1, design2, costs, label, prediction, correctness")

    good_design_miss_cnt = 0
    good_design_caught_cnt = 0
    bad_design_miss_cnt = 0
    bad_design_caught_cnt = 0
    for i in range(len(new_designs)):
        design1 = nn_input1[i,:]
        design2 = new_designs[i,:]
        costs = [sorted_training_cost[ref_dsn_idx-1], new_cost[i]]
        true_label = 0 if sorted_training_cost[ref_dsn_idx-1] < new_cost[i] else 1
        prediction = predictions[i, :]
        correct_flag = 0
        if np.argmax(prediction) == true_label:
            # this is a correct prediction (caught design)
            correct_flag = 1
            # cnt += 1
            if sorted_training_cost[ref_dsn_idx-1] < new_cost[i]:
                # bad_design_caught_cnt
                bad_design_caught_cnt += 1
            elif sorted_training_cost[ref_dsn_idx-1] > new_cost[i]:
                # good_design_caught_cnt
                good_design_caught_cnt += 1
        else:
            # this is a miss classification
            if sorted_training_cost[ref_dsn_idx-1] < new_cost[i]:
                # bad_design_miss_cnt
                bad_design_miss_cnt += 1
            elif sorted_training_cost[ref_dsn_idx-1] > new_cost[i]:
                # good_design_miss_cnt
                good_design_miss_cnt += 1

        # print("[Debug_test] {} -> {} -> {} -> {} -> {} -> {} ".format(list(design1), list(design2), costs, true_label, prediction, correct_flag))


    print("[Debug_test] accuracy = {}/{} = {}".format(good_design_caught_cnt+bad_design_caught_cnt,n_samples,
                                                      1.0*(good_design_caught_cnt+bad_design_caught_cnt)/n_samples))

    if good_design_caught_cnt+good_design_miss_cnt != 0 :
        print("[Debug_test] good_design recall accuracy = {}/{} = {}".format(good_design_caught_cnt,
                                                                             good_design_caught_cnt+good_design_miss_cnt,
                                                                             1.0*good_design_caught_cnt/(good_design_caught_cnt+good_design_miss_cnt)))
    else:
        print("[Debug_test] good_design recall accuracy = {}/{}".format(good_design_caught_cnt,
                                                                             good_design_caught_cnt+good_design_miss_cnt,))
    if (bad_design_caught_cnt+bad_design_miss_cnt) != 0:
        print("[Debug_test] bad_design recall accuracy = {}/{} = {}".format(bad_design_caught_cnt,
                                                                            bad_design_caught_cnt+bad_design_miss_cnt,
                                                                            1.0*bad_design_caught_cnt/(bad_design_caught_cnt+bad_design_miss_cnt)))
    else:
        print("[Debug_test] bad_design recall accuracy = {}/{}".format(bad_design_caught_cnt,
                                                                            bad_design_caught_cnt+bad_design_miss_cnt,))

    if good_design_caught_cnt+bad_design_miss_cnt != 0:
        print("[Debug_test] good_design precision accuracy = {}/{} = {}".format(good_design_caught_cnt,
                                                                                good_design_caught_cnt+bad_design_miss_cnt,
                                                                               1.0*good_design_caught_cnt/(good_design_caught_cnt+bad_design_miss_cnt)))
    else:
        print("[Debug_test] good_design precision accuracy = {}/{}".format(good_design_caught_cnt,
                                                                                good_design_caught_cnt+bad_design_miss_cnt,))
    if bad_design_caught_cnt+good_design_miss_cnt != 0:
        print("[Debug_test] bad_design precision accuracy = {}/{} = {}".format(bad_design_caught_cnt,
                                                                               bad_design_caught_cnt+good_design_miss_cnt,
                                                                               1.0*bad_design_caught_cnt/(bad_design_caught_cnt+good_design_miss_cnt)))
    else:
        print("[Debug_test] bad_design precision accuracy = {}/{}".format(bad_design_caught_cnt,
                                                                               bad_design_caught_cnt+good_design_miss_cnt,))


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

        # test_swaping(session, dataset, cost)

        for i in range(max_n_retraining):
            print(30*"-")
            print("[info] testing model ...")
            test_model(session, dataset, cost)
            print(30*"-")

            # run_model() requires sorted design pool we have so far:
            sorted_indices = sorted(range(len(dataset)), key=lambda x: cost[x])
            sorted_design_pool = dataset[sorted_indices]
            sorted_cost_pool = cost[sorted_indices]
            # store the sorted design pool to a log file for later plotting
            data_set_list.append(sorted_design_pool)
            cost_set_list.append(sorted_cost_pool)

            # get the reference design for comparison
            ref_design = sorted_design_pool[ref_dsn_idx]
            ref_cost = cost[sorted_indices[ref_dsn_idx]]
            # print("[info] retraining step: {}, ref design: {} -> {} ".format(i, ref_design, ref_cost))
            new_dataset, new_cost, new_predictions = run_model(session, sorted_design_pool, sorted_cost_pool,
                                                               n_new_samples,
                                                               ref_dsn_idx, max_iter=10000)
            if len(dataset) >=  max_data_set_size :
                # reached the maximum allowable number of evaluations
                break
            for k in range(len(new_dataset)):
                print("[debug] {} -> {} -> {}".format(new_dataset[k], new_cost[k], new_predictions[k]))

            dataset = np.concatenate((dataset, new_dataset), axis=0)
            cost = np.concatenate((cost, new_cost), axis=0)
            train(session, dataset, cost, writer, num_epochs=num_epochs, batch_size=batch_size)
        print("[finished] best_solution = {}".format(dataset[sorted_indices[0]]))
        print("[finished] cost = {}".format(cost[sorted_indices[0]]))
        print("[finished] performance {} ".format(evaluate_individual(dataset[sorted_indices[0]]), verbose=False))

        write_data = dict(
            data_set_list=data_set_list,
            cost_set_list=cost_set_list,
        )
        with open('genetic_nn/log_files/two_stage_logbook.pickle', 'wb') as f:
            pickle.dump(write_data, f)

if __name__ == '__main__':
    main()
