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

    def plot(self, data_list, annotations=None, plot_type=None, plot_options=None, display=True, multi_fig=False):
        plot_type = self.get_plot_type(plot_type=plot_type)
        plot_options = self.get_plot_options(plot_type, plot_options=plot_options)

        if not display:
            matplotlib.use('Agg')
      
        if multi_fig is True:
            fig_list = list()
            for i, data in enumerate(data_list):
                fig = plotter([data], annotations, plot_type, plot_options, display)
                fig_list.append(fig)
            return fig_list
        else:
            fig = plotter(data_list, annotations, plot_type, plot_options, display)
            return fig


    def plotter(self, data_list, annotations, plot_type, plot_options, display):
        fig = plt.figure(figsize=plot_options["figsize"], dpi=120, facecolor='w', edgecolor='k')

        get_y = lambda fn, plot_type: fn if plot_type == "det" else 1 - fn
        
        for obj in data_list:
            plt.plot(obj.fa, get_y(obj.y, plot_type), **obj.line_options)

        if plot_type.lower() == "det":
            plt.plot((1, 0), 'b--', lw=1)
        elif plot_type.lower() == "roc":
            plt.plot((0, 1), 'b--', lw=1)

        if len(data_list) == 1:
            for annotation in annotations:
                plt.annotate(annotation.to_dict())

        plt.xlim(plot.options["xlim"])
        plt.ylim(plot.options["ylim"])
        plt.xlabel(plot_opts['xlabel'], fontsize=plot_opts['xlabel_fontsize'])
        plt.ylabel(plot_opts['ylabel'], fontsize=plot_opts['ylabel_fontsize'])
        plt.xticks(ticks=plot_options["xticks"], labels=plot_options["xticks_labels"], fontsize=plot_opts['xticks_label_size'])
        plt.yticks(ticks=plot_options["yticks"], labels=plot_options["yticks_labels"], fontsize=plot_opts['yticks_label_size'])
        plt.title(plot_opts['title'], fontsize=plot_opts['title_fontsize'])
        plt.suptitle(plot_opts['subtitle'], fontsize=plot_opts['subtitle_fontsize'])
        plt.grid()

        # If any label has been provided
        if any([obj.curve_option.get("label",None) for obj in data_list]):
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

    @staticmethod
    def gen_default_plot_options(plot_type):
    """ This function generates JSON file to customize the plot.
        path: JSON file name along with the path
        plot_type: either DET or ROC"""
    
    plot_opts = OrderedDict([
        ('title', "Performance"),
        ('subtitle', ''),
        ('figsize', (7, 6.5))
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
    def __init__(self, text, xy, xytext, 
                 xycoords="data", 
                 textcoords=None,
                 arrowprops=None,
                 text_args=None,
                 bbox=None):
        self.text = text
        self.xy = xy
        self.xytext = xytext
        self.xycoords = xycoords
        self.textcoords = textcoords
        self.arrowprops = arrowprops
        self.text_args = text_args
        self.bbox = bbox

    def to_dict(self):
        return {"text":self.text, "xy":self.xy, "xytext":self.xytext}


