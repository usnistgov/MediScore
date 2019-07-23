import random
import string
import argparse

import numpy as np
import pandas as pd

from sklearn import metrics
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Render tester utility')
    parser.add_argument('-n', '--sys_number', type=int, help='number of system to generate and plot', default=10)
    parser.add_argument('-s', '--label_length', type=int, help='length of the random string added to the generated label', default=10)
    args = parser.parse_args()

    dm_number = args.sys_number
    np.random.seed(42)

    dm_list = []
    sys_list = []

    # Data generation

    random_seeds = np.random.choice(2*dm_number, size=dm_number, replace=False)
    for i, seed in enumerate(random_seeds):
        target_scores, non_target_scores, scores, labels = create_system(1000, 0.1, [np.random.randint(-2,3),np.random.randint(-5,1)], [np.random.randint(1,3),np.random.randint(1,3)], random_seed=seed)
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
        line_opts = MediForDataContainer.get_default_line_options()
        line_opts["color"] = None
        dm = MediForDataContainer(fpr, 1-tpr, thresholds, label="random_sys_{}_{}".format(i, random_string(args.label_length)), line_options=line_opts)
    #     dm.setter_standard(labels, scores, 1000, target_label=1, non_target_label=0, verbose=False)
        dm_list.append(dm)
        sys_list.append([target_scores, non_target_scores, scores, labels])

    # Plotting
    myRender = Render(plot_type="ROC", plot_options=None)

    plot_opts = Render.gen_default_plot_options("ROC")
    plot_opts["title"] = "ROC Title"
    plot_opts["figsize"] = (7, 6)

    myfigure = myRender.plot(dm_list, plot_options=plot_opts, display=True)

    myfigure.savefig('test_figure.pdf')