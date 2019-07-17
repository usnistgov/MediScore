# -*- coding: utf-8 -*-

import sys
import json
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from collections import OrderedDict

class Render:
    """Class implementing a renderer for DET and ROC curves:
    """

    def __init__(self, plot_type=None, plot_options=None):
        self.plot_type = plot_type
        self.plot_options = plot_options

    def get_plot_type(self, plot_type=None):
        if plot_type is not None:
            return plot_type.lower()
        elif self.plot_type is not None:
            return self.plot_type.lower()
        else:
            print("Error: No plot type has been set {'ROC' or 'DET'}. Either instance a specifif Render with Render(plot_type='roc') or provide a type to the plot method")
            sys.exit(1)

    def get_plot_options(self, plot_type, plot_options=None):
        if plot_options is not None:
            return plot_options
        elif self.plot_options is not None:
            return self.plot_options
        else:
            print("No plot options provided, setting default paramaters")
            return self.gen_default_plot_options(plot_type)

    def plot(self, data_list, annotations=[], plot_type=None, plot_options=None, display=True, multi_fig=False):
        plot_type = self.get_plot_type(plot_type=plot_type)
        plot_options = self.get_plot_options(plot_type, plot_options=plot_options)

        if not display:
            matplotlib.use('Agg')
      
        if multi_fig is True:
            fig_list = list()
            for i, data in enumerate(data_list):
                fig = self.plotter([data], annotations, plot_type, plot_options, display)
                fig_list.append(fig)
            return fig_list
        else:
            fig = self.plotter(data_list, annotations, plot_type, plot_options, display)
            return fig


    def plotter(self, data_list, annotations, plot_type, plot_options, display):
        fig = plt.figure(figsize=plot_options["figsize"], dpi=120, facecolor='w', edgecolor='k')

        get_y = lambda fn, plot_type: fn if plot_type == "det" else 1 - fn
        
        for obj in data_list:
            plt.plot(obj.fa, get_y(obj.fn, plot_type), **obj.line_options)

        # if plot_type.lower() == "det":
        #     plt.plot((1, 0), 'b--', lw=1)
        # elif plot_type.lower() == "roc":
        #     plt.plot((0, 1), 'b--', lw=1)

        if len(data_list) == 1:
            for annotation in annotations:
                plt.annotate(annotation.text, **annotation.paramaters)

        plt.xlim(plot_options["xlim"])
        plt.ylim(plot_options["ylim"])
        plt.xlabel(plot_options['xlabel'], fontsize=plot_options['xlabel_fontsize'])
        plt.ylabel(plot_options['ylabel'], fontsize=plot_options['ylabel_fontsize'])
        plt.xscale(plot_options["xscale"])
        plt.xticks(plot_options["xticks"], plot_options["xticks_labels"], fontsize=plot_options['xticks_label_size'])
        plt.yticks(plot_options["yticks"], plot_options["yticks_labels"], fontsize=plot_options['yticks_label_size'])
        plt.title(plot_options['title'], fontsize=plot_options['title_fontsize'])
        plt.suptitle(plot_options['subtitle'], fontsize=plot_options['subtitle_fontsize'])
        plt.grid()

        # If any label has been provided
        if any([obj.line_options.get("label",None) for obj in data_list]):
            plt.legend(loc='upper left', bbox_to_anchor=(0.6, 0.4), borderaxespad=0, prop={'size': 8}, shadow=True, fontsize='small')
            fig.tight_layout(pad=2.5)

        if display is True:
            plt.show()

        return fig

    def set_plot_options_from_file(path):
        """ Load JSON file for plot options"""
        with open(path, 'r') as f:
            opt_dict = json.load(f)
            self.plot_options = opt_dict

    def close_fig(self, figure):
        plt.close(figure)

    @staticmethod
    def gen_default_plot_options(plot_type,  plot_title=None):
        """ This function generates JSON file to customize the plot.
            path: JSON file name along with the path
            plot_type: either DET or ROC"""
        
        plot_opts = OrderedDict([
            ('title', "Performance" if plot_title is None else plot_title),
            ('subtitle', ''),
            ('figsize', (7, 6.5)),
            ('title_fontsize', 13), 
            ('subtitle_fontsize', 11), 
            ('xlim', [0,1]),
            ('ylim', [0,1]),
            ('xticks_label_size', 'medium'),
            ('yticks_label_size', 'medium'),
            ('xlabel', "False Alarm Rate [%]"),
            ('xlabel_fontsize', 11),
            ('ylabel_fontsize', 11)])

        if plot_type.lower() == "det":
            plot_opts["xscale"] = "log"
            plot_opts["ylabel"] = "Miss Detection Rate [%]"
            # plot_opts["xticks"] = norm.ppf([.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, .01, .02, .05, .10, .20, .40, .60, .80, .90, .95, .98, .99, .995, .999])
            plot_opts["xticks"] = [0.01, 0.1, 1, 10]
            plot_opts["yticks"] = norm.ppf([.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, .01, .02, .05, .10, .20, .40, .60, .80, .90, .95, .98, .99, .995, .999])
            plot_opts["xlim"] = (plot_opts["xticks"][0], plot_opts["xticks"][-1])
            plot_opts["ylim"] = (plot_opts["yticks"][0], plot_opts["yticks"][-1])
            # plot_opts["xticks_labels"] = ['0.01', '0.02', '0.05', '0.1', '0.2', '0.5', '1', '2', '5', '10', '20', '40', '60', '80', '90', '95', '98', '99', '99.5', '99.9']
            plot_opts["xticks_labels"] = ["0.01", "0.1", "1", "10"]
            plot_opts["yticks_labels"] = ['0.01', '0.02', '0.05', '0.1', '0.2', '0.5', '1', '2', '5', '10', '20', '40', '60', '80', '90', '95', '98', '99', '99.5', '99.9']

        elif plot_type.lower() == "roc":
            plot_opts["xscale"] = "linear"
            plot_opts["ylabel"] = "Correct Detection Rate [%]"
            plot_opts["xticks"] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            plot_opts["yticks"] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            plot_opts["yticks_labels"] = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']
            plot_opts["xticks_labels"] = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']

        return plot_opts

    

class Annotation:
    def __init__(self, text, parameters):
        self.text = text
        self.parameters = parameters

