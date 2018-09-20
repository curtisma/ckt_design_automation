# import seaborn as sns
# import pandas as pd
import matplotlib.pyplot as plt
# import json
import os
import pickle
import numpy as np

from util import Design

def plot_avg_top(k, history):

    history_sorted = [sorted(pop, key=lambda ind: ind.cost) for pop in history]
    data = [[ind.cost for ind in pop[:k]] for pop in history_sorted]
    data = np.array(data)
    data_t = data.transpose()
    avg_data = np.mean(data_t, axis=0)
    low_data = avg_data - np.std(data_t, axis=0)
    up_data = avg_data + np.std(data_t, axis=0)
    plt.plot(avg_data)
    plt.plot(low_data, '--r')
    plt.plot(up_data, '--r')
    plt.ylabel("avg_top_%d" %k)
    plt.xlabel("generation")
    plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', type=str)
    parser.add_argument('--avg_top', type=int, default=20)
    args = parser.parse_args()

    assert os.path.exists(args.logdir)
    with open(args.logdir, 'rb') as f:
        history = pickle.load(f)

    plot_avg_top(args.avg_top, history)

if __name__ == '__main__':
    main()