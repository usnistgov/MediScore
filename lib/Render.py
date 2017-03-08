#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class Render:
    """Class implementing a renderer for DET and ROC curves:
    """

    def __init__(self,setRender):
        self.DM_list = setRender.DM_list
        self.opts_list = setRender.opts_list
        self.plot_opts = setRender.plot_opts

    def plot_curve(self, display=True, multi_fig=False):
        """ Return single figure or a list of figures depending on the multi_fig option
        display: to display the figure from command-line
        multi_fig: generate a single curve plot per partition
        """
        if multi_fig is True:
            fig_list = list()
            for i,dm in enumerate(self.DM_list):
                fig = self.plot_fig([dm],i,display,multi_fig)
                fig_list.append(fig)
            return fig_list
        else:
            fig = self.plot_fig(self.DM_list,1,display)
            return fig

    #TODO: add auc values to each legend
    def plot_fig(self, dm_list, fig_number, display=True, multi_fig=False):
        """Generate plot with the specified options
        dm_list: a list of detection metrics for partitions
        fig_number: a number of plot figures
        display: to display the figure from command-line
        multi_fig: generate a single curve plot per partition
        """
        fig = plt.figure(num=fig_number, figsize=(7,6), dpi=120, facecolor='w', edgecolor='k')
        nb_dm_objects = len(dm_list)
        # DET curve settings
        if self.plot_opts['plot_type'] == 'DET':
            fnrs = [dm.fnr for dm in dm_list]
            fprs = [dm.fpr for dm in dm_list]
            norm_fnrs = list(map(norm.ppf, fnrs))
            norm_fprs = list(map(norm.ppf, fprs))
            #xytick_labels = [.01, .02, .05, .10, .20, .40, .60, .80, .90, .95, .98, .99]
#            xytick_labels = [0.00001, 0.0001, 0.001, 0.004, .01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40, 60, 80, 90, 95, 98, 99, 99.5, 99.9]
            xytick_labels = [.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, .01, .02, .05, .10, .20, .40, .60, .80, .90, .95, .98, .99, .995, .999]
            xytick = norm.ppf(xytick_labels)
            x_tick_labels = ['0.01', '0.02', '0.05', '0.1', '0.2', '0.5', '1', '2', '5', '10', '20', '40', '60', '80', '90', '95', '98', '99', '99.5', '99.9']
            y_tick_labels = ['0.01', '0.02', '0.05', '0.1', '0.2', '0.5', '1', '2', '5', '10', '20', '40', '60', '80', '90', '95', '98', '99', '99.5', '99.9']

            if multi_fig:
                plt.plot(norm_fprs[0], norm_fnrs[0], **self.opts_list[fig_number])
            else:
                for i, points in enumerate(zip(norm_fprs, norm_fnrs)):
                    fpr, fnr = points
                    plt.plot(fpr, fnr, **self.opts_list[i])

            plt.xlim([0, 1])
            plt.ylim([0, 1])
#            plt.plot((1, 0), 'k--', lw=2)

            # display the eer when there is only one curve
            if nb_dm_objects == 1:
                DM = dm_list[0]
                norm_fnrs_pos_ci = list(map(norm.ppf, DM.fnr+DM.ci_tpr))
                norm_fnrs_neg_ci = list(map(norm.ppf, DM.fnr-DM.ci_tpr))
                norm_fpr = list(map(norm.ppf, DM.fpr))
                plt.plot(norm_fpr, norm_fnrs_pos_ci, 'k--')
                plt.plot(norm_fpr, norm_fnrs_neg_ci, 'k--')
                plt.annotate("EER = %.2f%%" %(DM.eer*100),xy=(norm.ppf(DM.eer), norm.ppf(DM.eer)), xycoords='data',
                             xytext=(norm.ppf(DM.eer+0.05)+0.5, norm.ppf(DM.eer+0.05)+0.5), textcoords='data',
                             arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3, rad=+0.2", fc="w"),
                             size=10, va='center', ha='center', bbox=dict(boxstyle="round4", fc="w"),)


        # ROC curve settings
        elif self.plot_opts['plot_type'] == 'ROC':
            xytick_labels = np.linspace(0, 1, 11)
            xytick = xytick_labels
            x_tick_labels = [str(int(x * 100)) for x in xytick_labels]
            y_tick_labels = [str(int(x * 100)) for x in xytick_labels]
            tprs = [dm.tpr for dm in dm_list]
            fprs = [dm.fpr for dm in dm_list]

            if multi_fig:
                plt.plot(fprs[0], tprs[0], **self.opts_list[fig_number])
            else:
                for i, points in enumerate(zip(fprs, tprs)):
                    fpr, tpr = points
                    plt.plot(fpr, tpr, **self.opts_list[i])

            plt.plot((0, 1), '--', lw=0.5) # plot bisector
            plt.xlim([0, 1])
            plt.ylim([0, 1])

            # We display the confidence interval when there is only one curve
            if nb_dm_objects == 1:
                DM = dm_list[0]
                plt.plot(DM.fpr, DM.tpr+DM.ci_tpr, 'k--')
                plt.plot(DM.fpr, DM.tpr-DM.ci_tpr, 'k--')
                #TODO: add number of data
                plt.annotate("AUC=%.2f at FAR=%.2f\n(T#: %d, NT#: %d, TRR: %.2f) " %(DM.auc,DM.fpr_stop, DM.t_num, DM.nt_num, DM.trr), xy=(0.7,0.2), xycoords='data', xytext=(0.7,0.2), textcoords='data',
                     size=10, va='center', ha='center', bbox=dict(boxstyle="round4", fc="w"))

#                plt.annotate("d = %.2f" %(DM.d), xy=(DM.dpoint[0], DM.dpoint[1]), xycoords='data', xytext=(0.9,0.5), textcoords='data',
#                     size=10, va='center', ha='center', bbox=dict(boxstyle="round4", fc="w"),)
                if DM.d is not None:
                    x = DM.dpoint[0]
                    y = DM.dpoint[1]
                    if (             y <= .5): x += .1;
                    elif (.5 < y and y < .9): x -= .1;
                    elif (         y >= .9): y -= .1;

                    plt.annotate("d' = %.2f" %(DM.d),xy=(DM.dpoint[0], DM.dpoint[1]), xycoords='data',
                                 xytext=(x, y), textcoords='data',
                                 arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=0"), #http://matplotlib.org/examples/pylab_examples/annotation_demo2.html
                                 size=8, va='center', ha='center', bbox=dict(boxstyle="round4", fc="w"))

#                plt.annotate("a' = %.2f" %(DM.a),xy=(DM.apoint[0], DM.apoint[1]), xycoords='data',
#                             xytext=(DM.apoint[0]+0.1, DM.apoint[1]+0.1), textcoords='data',
#                             arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3, rad=+0.1", fc="w"),
#                             size=10, va='center', ha='center', bbox=dict(boxstyle="round4", fc="w"),)

                #TODO: how to add variable here for fpr_stop
#                plt.annotate("PAUC = %.2f%% at FAR=" %(pauc*100), xy=(0.7,0.2), xycoords='data', xytext=(0.7,0.2), textcoords='data',
#                     size=12, va='center', ha='center', bbox=dict(boxstyle="round4", fc="w"),)


        plt.xticks(xytick, x_tick_labels, size=self.plot_opts['xticks_size'])
        plt.yticks(xytick, y_tick_labels, size=self.plot_opts['yticks_size'])
        plt.suptitle(self.plot_opts['title'], fontsize=self.plot_opts['title_fontsize'])
        plt.xlabel(self.plot_opts['xlabel'], fontsize=self.plot_opts['xlabel_fontsize'])
        if self.plot_opts['plot_type'] == 'ROC':
            plt.ylabel("Correct Detection Rate [%]", fontsize=self.plot_opts['ylabel_fontsize'])
        else:
            plt.ylabel(self.plot_opts['ylabel'], fontsize=self.plot_opts['ylabel_fontsize'])
        plt.grid()

        if self.opts_list[0]['label'] != None:
            plt.legend(bbox_to_anchor=(0., -0.35, 1., .102), loc='lower center', prop={'size':8}, shadow=True, fontsize='medium')
            # Put a nicer background color on the legend.
            #legend.get_frame().set_facecolor('#00FFCC')
            #plt.legend(loc='upper left', prop={'size':6}, bbox_to_anchor=(1,1))
            fig.tight_layout(pad=7)

        if display is True:
            plt.show()

        return fig


def gen_default_plot_options(path="./plotJsonFiles/plot_options.json",plot_type='DET'):
    """ This function generates JSON file to customize the plot.
        path: JSON file name along with the path
        plot_type: either DET or ROC"""
    from collections import OrderedDict
    mon_dict = OrderedDict([
        ('title', plot_type),
        ('plot_type', plot_type),
        ('title_fontsize', 15),
        ('xticks_size', 'medium'),
        ('yticks_size', 'medium'),
        ('xlabel', "False Alarm Rate [%]"),
        ('xlabel_fontsize', 12),
        ('ylabel', "Miss Detection Rate [%]"),
        ('ylabel_fontsize', 12)])
    with open(path, 'w') as f:
        f.write(json.dumps(mon_dict).replace(',', ',\n'))


def load_plot_options(path="./plotJsonFiles/plot_options.json"):
    """ Load JSON file for plot options"""
    with open(path, 'r') as f:
        opt_dict = json.load(f)
    return opt_dict

def open_plot_options(path="./plotJsonFiles/plot_options.json"):
    """ open JSON file for customizng plot options"""
    import os
    try:
        os.system("idle " + path)
    except IOError:
        print("ERROR: There was an error opening JSON file")
        exit(1)

class setRender:

    def __init__(self, DM_list, opts_list, plot_opts):
        self.DM_list = DM_list
        self.opts_list = opts_list
        self.plot_opts = plot_opts
