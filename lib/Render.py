#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
# the following two lines are necessary for remote access.

from scipy.stats import norm


class Render:
    """Class implementing a renderer for DET and ROC curves:
    """

    def __init__(self, setRender):
        self.DM_list = setRender.DM_list
        self.opts_list = setRender.opts_list
        self.plot_opts = setRender.plot_opts

    def plot_curve(self, display=False, multi_fig=False, isOptOut=False, isNoNumber=False, isCI=False):
        """ Return single figure or a list of figures depending on the multi_fig option
        display: to display the figure from command-line
        multi_fig: generate a single curve plot per partition
        """

        if multi_fig is True:
            fig_list = list()
            for i, dm in enumerate(self.DM_list):
                fig = self.plot_fig([dm], i, display, multi_fig, isOptOut, isNoNumber, isCI)
                fig_list.append(fig)
            return fig_list
        else:
            fig = self.plot_fig(self.DM_list, 1, display, multi_fig, isOptOut, isNoNumber, isCI)
            return fig

    # TODO: add auc values to each legend
    def plot_fig(self, dm_list, fig_number, display=False, multi_fig=False, isOptOut=False, isNoNumber=False, isCI=False):
        """Generate plot with the specified options
        dm_list: a list of detection metrics for partitions
        fig_number: a number of plot figures
        display: to display the figure from command-line
        multi_fig: generate a single curve plot per partition
        """

        if not display:
            import matplotlib
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig = plt.figure(num=fig_number, figsize=(7, 6.5), dpi=120, facecolor='w', edgecolor='k')
        nb_dm_objects = len(dm_list)
        # DET curve settings
        if self.plot_opts['plot_type'] == 'DET':
            fnrs = [dm.fnr for dm in dm_list]
            fprs = [dm.fpr for dm in dm_list]
            norm_fnrs = list(map(norm.ppf, fnrs))
            norm_fprs = list(map(norm.ppf, fprs))
            #xytick_labels = [.01, .02, .05, .10, .20, .40, .60, .80, .90, .95, .98, .99]
#            xytick_labels = [0.00001, 0.0001, 0.001, 0.004, .01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40, 60, 80, 90, 95, 98, 99, 99.5, 99.9]
            xytick_labels = [.0001, 0.0002, 0.0005, 0.001, 0.002,
                             0.005, .01, .02, .05, .10, .20, .40, .60, .80, .90, .95, .98, .99, .995, .999]
            xytick = norm.ppf(xytick_labels)
            x_tick_labels = ['0.01', '0.02', '0.05', '0.1', '0.2', '0.5', '1', '2',
                             '5', '10', '20', '40', '60', '80', '90', '95', '98', '99', '99.5', '99.9']
            y_tick_labels = ['0.01', '0.02', '0.05', '0.1', '0.2', '0.5', '1', '2',
                             '5', '10', '20', '40', '60', '80', '90', '95', '98', '99', '99.5', '99.9']

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
                ##TODO: need to change this
                #norm_fnrs_pos_ci = list(map(norm.ppf, DM.fnr + DM.tpr_at_fpr_ci_lower))
                #norm_fnrs_neg_ci = list(map(norm.ppf, DM.fnr - DM.tpr_at_fpr_ci_upper))
                norm_fnrs_pos_ci = list(map(norm.ppf, DM.fnr))
                norm_fnrs_neg_ci = list(map(norm.ppf, DM.fnr))
                norm_fpr = list(map(norm.ppf, DM.fpr))
                plt.plot(norm_fpr, norm_fnrs_pos_ci, 'k--')
                plt.plot(norm_fpr, norm_fnrs_neg_ci, 'k--')

                if isNoNumber:
                    if isOptOut:
                        plt.annotate("trEER = %.2f (TRR: %.2f)" % (DM.eer * 100, DM.trr), xy=(norm.ppf(DM.eer), norm.ppf(DM.eer)), xycoords='data',
                                     xytext=(norm.ppf(DM.eer + 0.05) + 0.5, norm.ppf(DM.eer + 0.05) + 0.5), textcoords='data',
                                     arrowprops=dict(arrowstyle="-|>",
                                                     connectionstyle="arc3, rad=+0.2", fc="w"),
                                     size=10, va='center', ha='center', bbox=dict(boxstyle="round4", fc="w"),)
                    else:
                        plt.annotate("EER = %.2f%%" % (DM.eer * 100), xy=(norm.ppf(DM.eer), norm.ppf(DM.eer)), xycoords='data',
                                     xytext=(norm.ppf(DM.eer + 0.05) + 0.5, norm.ppf(DM.eer + 0.05) + 0.5), textcoords='data',
                                     arrowprops=dict(arrowstyle="-|>",
                                                     connectionstyle="arc3, rad=+0.2", fc="w"),
                                     size=10, va='center', ha='center', bbox=dict(boxstyle="round4", fc="w"),)
                else:
                    if isOptOut:
                        plt.annotate("trEER = %.2f \n(TRR: %.2f, T#: %d, NT#: %d)" % (DM.eer * 100, DM.trr, DM.t_num, DM.nt_num), xy=(norm.ppf(DM.eer), norm.ppf(DM.eer)), xycoords='data',
                                     xytext=(norm.ppf(DM.eer + 0.05) + 0.5, norm.ppf(DM.eer + 0.05) + 0.5), textcoords='data',
                                     arrowprops=dict(arrowstyle="-|>",
                                                     connectionstyle="arc3, rad=+0.2", fc="w"),
                                     size=9, va='center', ha='center', bbox=dict(boxstyle="round4", fc="w"),)
                    else:
                        plt.annotate("EER = %.2f \n(T#: %d, NT#: %d)" % (DM.eer * 100, DM.t_num, DM.nt_num), xy=(norm.ppf(DM.eer), norm.ppf(DM.eer)), xycoords='data',
                                     xytext=(norm.ppf(DM.eer + 0.05) + 0.5, norm.ppf(DM.eer + 0.05) + 0.5), textcoords='data',
                                     arrowprops=dict(arrowstyle="-|>",
                                                     connectionstyle="arc3, rad=+0.2", fc="w"),
                                     size=9, va='center', ha='center', bbox=dict(boxstyle="round4", fc="w"),)

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

            plt.plot((0, 1), '--', lw=0.5)  # plot bisector
            plt.xlim([0, 1])
            plt.ylim([0, 1])

            # We display the confidence interval when there is only one curve
            if nb_dm_objects == 1:
                DM = dm_list[0]
                #TODO: This is just one point, we should change this for all the tpr CI points
                #tpr_ci_upper = DM.tpr_at_fpr_ci_upper - DM.tpr_at_fpr
                #tpr_ci_lower = DM.tpr_at_fpr - DM.tpr_at_fpr_ci_lower
                tpr_ci_upper = 0
                tpr_ci_lower = 0
                plt.plot(DM.fpr, DM.tpr + tpr_ci_upper ,'k--')
                plt.plot(DM.fpr, DM.tpr - tpr_ci_lower, 'k--')

                if isCI:
                    if isNoNumber:
                        if isOptOut:
                            plt.annotate("trAUC=%.2f\n(TRR: %.2f, CI_L: %.2f, CI_U: %.2f)" % (DM.auc, DM.trr, DM.auc_ci_lower,DM.auc_ci_upper), xy=(0.7, 0.2), xycoords='data', xytext=(0.7, 0.2), textcoords='data',
                                         size=10, va='center', ha='center', bbox=dict(boxstyle="round4", fc="w"))
                        else:
                            plt.annotate("AUC=%.2f (CI_L: %.2f, CI_U: %.2f)" % (DM.auc, DM.auc_ci_lower, DM.auc_ci_upper), xy=(0.7, 0.2), xycoords='data', xytext=(0.7, 0.2), textcoords='data',
                                         size=10, va='center', ha='center', bbox=dict(boxstyle="round4", fc="w"))

                    else:
                        if isOptOut:
                            plt.annotate("trAUC=%.2f\n(TRR: %.2f, CI_L: %.2f, CI_U: %.2f, T#: %d, NT#: %d) " % (DM.auc, DM.trr, DM.auc_ci_lower,DM.auc_ci_upper,DM.t_num, DM.nt_num), xy=(0.7, 0.2), xycoords='data', xytext=(0.7, 0.2), textcoords='data',
                                         size=10, va='center', ha='center', bbox=dict(boxstyle="round4", fc="w"))
                        else:
                            plt.annotate("AUC=%.2f\n(CI_L: %.2f, CI_U: %.2f, T#: %d, NT#: %d) " % (DM.auc, DM.auc_ci_lower,DM.auc_ci_upper, DM.t_num, DM.nt_num), xy=(0.7, 0.2), xycoords='data', xytext=(0.7, 0.2), textcoords='data',
                                         size=10, va='center', ha='center', bbox=dict(boxstyle="round4", fc="w"))
                else:
                    if isNoNumber:
                        if isOptOut:
                            plt.annotate("trAUC=%.2f\n(TRR: %.2f)" % (DM.auc, DM.trr), xy=(0.7, 0.2), xycoords='data', xytext=(0.7, 0.2), textcoords='data',
                                         size=10, va='center', ha='center', bbox=dict(boxstyle="round4", fc="w"))
                        else:
                            plt.annotate("AUC=%.2f" % (DM.auc), xy=(0.7, 0.2), xycoords='data', xytext=(0.7, 0.2), textcoords='data',
                                         size=10, va='center', ha='center', bbox=dict(boxstyle="round4", fc="w"))

                    else:
                        if isOptOut:
                            plt.annotate("trAUC=%.2f\n(TRR: %.2f, T#: %d, NT#: %d) " % (DM.auc, DM.trr, DM.t_num, DM.nt_num), xy=(0.7, 0.2), xycoords='data', xytext=(0.7, 0.2), textcoords='data', size=10, va='center', ha='center', bbox=dict(boxstyle="round4", fc="w"))
                        else:
                            plt.annotate("AUC=%.2f\n(T#: %d, NT#: %d) " % (DM.auc, DM.t_num, DM.nt_num), xy=(0.7, 0.2), xycoords='data', xytext=(0.7, 0.2), textcoords='data',size=10, va='center', ha='center', bbox=dict(boxstyle="round4", fc="w"))


#                plt.annotate("d = %.2f" %(DM.d), xy=(DM.dpoint[0], DM.dpoint[1]), xycoords='data', xytext=(0.9,0.5), textcoords='data',
#                     size=10, va='center', ha='center', bbox=dict(boxstyle="round4", fc="w"),)
                if DM.d is not None:
                    x = DM.dpoint[0]
                    y = DM.dpoint[1]
                    if (y <= .5):
                        x += .1
                    elif (.5 < y and y < .9):
                        x -= .1
                    elif (y >= .9):
                        y -= .1

                    plt.annotate("d' = %.2f" % (DM.d), xy=(DM.dpoint[0], DM.dpoint[1]), xycoords='data',
                                 xytext=(x, y), textcoords='data',
                                 # http://matplotlib.org/examples/pylab_examples/annotation_demo2.html
                                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                                 size=8, va='center', ha='center', bbox=dict(boxstyle="round4", fc="w"))

                # TODO: how to add variable here for fpr_stop
#                plt.annotate("PAUC = %.2f%% at FAR=" %(pauc*100), xy=(0.7,0.2), xycoords='data', xytext=(0.7,0.2), textcoords='data',
#                     size=12, va='center', ha='center', bbox=dict(boxstyle="round4", fc="w"),)

        plt.xticks(xytick, x_tick_labels, size=self.plot_opts['xticks_size'])
        plt.yticks(xytick, y_tick_labels, size=self.plot_opts['yticks_size'])
        plt.suptitle(self.plot_opts['title'], fontsize=self.plot_opts['title_fontsize'])
        plt.title(self.plot_opts['subtitle'], fontsize=self.plot_opts['subtitle_fontsize'])
        plt.xlabel(self.plot_opts['xlabel'], fontsize=self.plot_opts['xlabel_fontsize'])

        if self.plot_opts['plot_type'] == 'ROC':
            plt.ylabel("Correct Detection Rate [%]", fontsize=self.plot_opts['ylabel_fontsize'])
        else:
            plt.ylabel(self.plot_opts['ylabel'], fontsize=self.plot_opts['ylabel_fontsize'])
        plt.grid()

        if any([curve_option.get("label",None) for curve_option in self.opts_list]):
            #            lgd = plt.legend(loc='lower right', prop={'size':8}, shadow=True, fontsize='medium', bbox_to_anchor=(0., -0.35, 1., .102))
            # Put a nicer background color on the legend.
            # legend.get_frame().set_facecolor('#00FFCC')
            #plt.legend(loc='upper left', prop={'size':6}, bbox_to_anchor=(1,1))
            fig.tight_layout(pad=2.5)

            plt.legend(loc='upper left', bbox_to_anchor=(0.6, 0.4), borderaxespad=0,
                       prop={'size': 8}, shadow=True, fontsize='small')

        if display is True:
            plt.show()

        plt.close()
        return fig


def gen_default_plot_options(plot_title='Performance', plot_subtitle='', plot_type='ROC'):
    """ This function generates JSON file to customize the plot.
        path: JSON file name along with the path
        plot_type: either DET or ROC"""
    from collections import OrderedDict
    plot_opts = OrderedDict([
        ('title', plot_title),
        ('subtitle', plot_subtitle),
        ('plot_type', plot_type),
        ('title_fontsize', 13),  # 15
        ('subtitle_fontsize', 11),  # 15
        ('xticks_size', 'medium'),
        ('yticks_size', 'medium'),
        ('xlabel', "False Alarm Rate [%]"),
        ('xlabel_fontsize', 11),
        ('ylabel', "Miss Detection Rate [%]"),
        ('ylabel_fontsize', 11)])
    return plot_opts

def gen_default_curve_options(number):
    """ Creation of defaults plot curve options dictionnary (line style opts)
    """
    from itertools import cycle
    from collections import OrderedDict
    Curve_opt = OrderedDict([('color', 'red'),
                             ('linestyle', 'solid'),
                             ('marker', '.'),
                             ('markersize', 5),
                             ('markerfacecolor', 'red'),
                             ('label',None),
                             ('antialiased', 'False')])

    # Creating the list of curves options dictionnaries (will be automatic)
    opts_list = list()
    colors = ['red','blue','green','cyan','magenta','yellow','black','sienna','navy','grey','darkorange', 'c', 'peru','y','pink','purple', 'lime', 'magenta', 'olive', 'firebrick']
    linestyles = ['solid','dashed','dashdot','dotted']
    markerstyles = ['.','+','x','d','*','s','p']
    # Give a random rainbow color to each curve
    #color = iter(cm.rainbow(np.linspace(0,1,number))) #YYL: error here
    color = cycle(colors)
    lty = cycle(linestyles)
    mkr = cycle(markerstyles)

    for i in range(number):
        new_curve_option = OrderedDict(Curve_opt)
        col = next(color)
        new_curve_option['color'] = col
        new_curve_option['marker'] = next(mkr)
        new_curve_option['markerfacecolor'] = col
        new_curve_option['linestyle'] = next(lty)
        opts_list.append(new_curve_option)

    return opts_list


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
