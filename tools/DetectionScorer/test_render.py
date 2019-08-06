import random
import string
import argparse

import numpy as np
import pandas as pd

from sklearn_metrics import roc_curve
import matplotlib.pyplot as plt

from datacontainer import DataContainer
from medifor_datacontainer import MediForDataContainer
from render import Render

old_printoptions = np.get_printoptions()
np.set_printoptions(suppress=True)

# Random system generation functions

def normal(size, mean=0, stdev=1):
    mu, sigma = mean, stdev
    return sigma * np.random.randn(size) + mu

def create_system(n, target_ratio, means, stdevs, random_seed=7):
    np.random.seed(random_seed)
    nb_target = int(n * target_ratio)
    nb_non_target = int(n * (1 - target_ratio))
    target_mean, non_target_mean = means
    target_stdev, non_target_stdev = stdevs
    target_scores = normal(nb_target, mean=target_mean, stdev=target_stdev)
    non_target_scores = normal(nb_non_target, mean=non_target_mean, stdev=non_target_stdev)
    scores = np.r_[target_scores,non_target_scores]
    labels = np.r_[np.ones(nb_target), np.zeros(nb_non_target)]
    return target_scores, non_target_scores, scores, labels

def random_string(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def gen_data_containers(n, set_metrics=False, labels=None, means_boudaries=[[-2,3],[-5,1]], stdevs_boundaries=[[1,3],[1,3]], random_seed=42):
    np.random.seed(random_seed)
    dm_list = []
    random_seeds = np.random.choice(2*n, size=n, replace=False)
    target_mean_bd, non_target_mean_bd = means_boudaries
    target_stdev_bd, non_target_stdev_bd = stdevs_boundaries

    if labels is None:
        labels = ["random_sys_{}".format(i) for i in range(1, n+1)]

    for i, (label, seed) in enumerate(zip(labels, random_seeds)):
        
        target_scores, non_target_scores, scores, labels = create_system(1000, 0.1, [np.random.randint(*target_mean_bd), np.random.randint(*non_target_mean_bd)], 
                                                                                    [np.random.randint(*target_stdev_bd), np.random.randint(*non_target_stdev_bd)], 
                                                                                    random_seed=seed)
        fpr, tpr, thresholds = roc_curve(labels, scores)

        line_opts = MediForDataContainer.get_default_line_options()
        line_opts["color"] = None

        dm = MediForDataContainer(fpr, 1-tpr, thresholds, label=label, line_options=line_opts)

        if set_metrics:
            dm.setter_standard(labels, scores, 1000, target_label=1, non_target_label=0, verbose=False)

        dm_list.append(dm)

    return dm_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Render tester utility')
    parser.add_argument('-n', '--sys_number', type=int, help='number of system to generate and plot', default=10)
    args = parser.parse_args()

    # label_extra_length = 60
    # long_labels = ["random_sys_{}_{}".format(i, random_string(label_extra_length)) for i in range(1, n+1)]

    # Data generation
    dm_list = gen_data_containers(args.sys_number, set_metrics=False, labels=None, means_boudaries=[[-2,3],[-5,1]], stdevs_boundaries=[[1,3],[1,3]], random_seed=42)

    # Plotting
    myRender = Render(plot_type="ROC", plot_options=None)

    plot_opts = Render.gen_default_plot_options("ROC")
    plot_opts["title"] = "ROC Title"
    plot_opts["figsize"] = (7, 6)

    myfigure = myRender.plot(dm_list, plot_options=plot_opts, display=True, auto_width=True)

    myfigure.savefig('test_figure.pdf')