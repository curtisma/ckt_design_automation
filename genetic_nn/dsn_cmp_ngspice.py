"""

>>> run genetic_nn/dsn_cmp_ngspice.py ./framework/yaml_files/two_stage_full.yaml two_stage_full --load_model
"""

from framework.wrapper import TwoStageComplete as sim
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import random
import pickle
import yaml



def evaluate_individual(individual, verbose=False):
    mp1_idx = int(individual[0])
    mn1_idx = int(individual[1])
    mn3_idx = int(individual[2])
    mp3_idx = int(individual[3])
    mn5_idx = int(individual[4])
    mn4_idx = int(individual[5])
    cc_idx  = int(individual[6])
    cost_val = eval_core.cost_fun(mp1_idx, mn1_idx, mp3_idx, mn3_idx, mn4_idx, mn5_idx, cc_idx, verbose=verbose)
    return cost_val


def construct_nn():
    pass

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('ckt_yaml_path', type='str')
    parser.add_argument('ckt_name', type='str')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--model_dir', type=str, default='genetic_nn/checkpoint')
    args = parser.parse_args()

    with open(args.ckt_yaml_path, 'r') as f:
        yaml_data = yaml.load(f)

    assert {''}
    eval_core = sim.EvaluationCore(args.ckt_yaml_path)

    graph = tf.Graph()


    dim_list =
    with tf.Session(graph=graph) as session:

        writer = tf.summary.FileWriter(summary_dir)
        writer.add_graph(graph)
        all_vars = tf.global_variables()
        saver = tf.train.Saver(all_vars)
        if not args.load_model:
            dataset, cost, _, _, _ = generate_data_set(n=n_init_samples)
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


        for i in range(max_n_retraining):
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
            print("[info] retraining step: {}, best design: {} -> {} ".format(i, ref_design, ref_cost))
            new_dataset, new_cost, _, _, _, new_predictions = run_model(session, sorted_design_pool, n_new_samples,
                                                                        ref_dsn_idx, max_iter=1000)
            if len(new_dataset) <= 0.1*n_new_samples :
                # there are new points found that are as good as the old solutions
                break
            for k in range(len(new_dataset)):
                print("[debug] {} -> {} -> {}".format(new_dataset[k], new_cost[k], new_predictions[k]))

            dataset = np.concatenate((dataset, new_dataset), axis=0)
            cost = np.concatenate((cost, new_cost), axis=0)
            nn_dataset, nn_labels, cost_arr = combine(dataset, cost)
            train(session, dataset, cost, writer, num_epochs=num_epochs, batch_size=batch_size)
        print("[finished] best_solution = {}".format(dataset[sorted_indices[0]]))
        print("[finished] cost = {}".format(cost[sorted_indices[0]]))
        _, bw, gain, ibias = sim(dataset[sorted_indices[0]][1], dataset[sorted_indices[0]][0])
        print("[finished] bw = {}, gain = {}, ibias = {}".format(bw, gain, ibias))

        cost_mat, res_vec, mul_vec = get_cost_surface()
        write_data = dict(
            data_set_list=data_set_list,
            cost_set_list=cost_set_list,
            cost_mat=cost_mat,
            x_vec=res_vec,
            y_vec=mul_vec,
        )
        with open('genetic_nn/log_files/cs_logbook.pickle', 'wb') as f:
            pickle.dump(write_data, f)

if __name__ == '__main__':
    main()