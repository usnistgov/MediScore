#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Date: 03/07/2017
Authors: Yooyoung Lee and Timothee Kheyrkhah

Description: this code loads DM files and renders plots.
In addition, the code generates a JSON file that allows user
to customize curves and points.

"""

import argparse
import numpy as np
#import pandas as pd
import os # os.system("pause") for windows command line
import sys
import json
from ast import literal_eval

#import matplotlib.pyplot as plt
#from matplotlib.pyplot import cm
from collections import OrderedDict
from itertools import cycle

#lib_path = "../../lib"
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib")
sys.path.append(lib_path)
import Render as p
import detMetrics as dm


########### Command line interface ########################################################

if __name__ == '__main__':

    # Boolean for disabling the command line
    debug_mode_ide = False

    # Command-line mode
    if not debug_mode_ide:

        parser = argparse.ArgumentParser(description='NIST detection scorer.', formatter_class=argparse.RawTextHelpFormatter)
 
        input_help = ("Supports the following inputs:\n- .txt file containing one file path per line\n- .dm file\n",
                      "- a list of pair ['path/to/dm/file', **{any matplotlib.lines.Line2D properties}].\n",
                      "Example:\n  [[{'path':'path/to/file_1.dm','label':'sys_1','show_label':True}, {'color':'red','linestyle':'solid'}],\n",
                      "             [{'path':'path/to/file_2.dm','label':'sys_2','show_label':False}, {}]",
                      "Note: Use an empty dict for default behavior.")

        parser.add_argument('-i', '--input', required=True,metavar = "str",
                            help=''.join(input_help))

        parser.add_argument("--outputFolder", default='.',
                            help="Path to the output folder. (default: %(default)s)",metavar='')

        parser.add_argument("--outputFileNameSuffix", default='plot',
                            help="Output file name suffix. (default: '%(default)s')",metavar='')

        # Plot Options
        parser.add_argument("--plotOptionJsonFile", help="Path to a json file containing plot options", metavar='path')

        parser.add_argument("--curveOptionJsonFile", help="Path to a json file containing a list of matplotlib.lines.Line2D dictionnaries properties (One per curve)", metavar='path')

        parser.add_argument("--plotType", default="ROC", choices=["ROC", "DET"],
                            help="Plot type (default: %(default)s)", metavar='')

        parser.add_argument("--plotTitle",default="Performance",
                            help="Define the plot title (default: '%(default)s')", metavar='')

        parser.add_argument("--plotSubtitle",default='',
                            help="Define the plot subtitle (default: '%(default)s')", metavar='')

        parser.add_argument("--display", action="store_true",
                            help="Display plots")

        parser.add_argument("--multiFigs", action="store_true",
                            help="Generate plots (with only one curve) per a partition")

        parser.add_argument('--noNum', action="store_true",
                            help="Do not print the number of target trials and non-target trials on the legend of the plot")

        parser.add_argument('-v', "--verbose", action="store_true",
                            help="Increase output verbosity")

        args = parser.parse_args()
    

        if not args.display:
            import matplotlib
            matplotlib.use('Agg')
  
        # Input processing
        custom_input_metadata = False

        DM_list = list()
        # Case 1: text file containing one path per line
        if args.input.endswith('.txt'):
            with open(args.input) as f:
                fp_list = f.read().splitlines()

            for dm_file_path in fp_list:
                # We handle a potential label provided
                if ':' in filename:
                    dm_file_path, label = filename.rsplit(':', 1)

                dm_obj = dm.load_dm_file(dm_file_path)
                dm_obj.path = dm_file_path
                dm_obj.label = label
                dm_obj.show_label = True
                DM_list.append(dm_obj)

        # Case 2: One dm pickled file
        elif args.input.endswith('.dm'):
            dm_obj = dm.load_dm_file(args.input)
            dm_obj.path = args.input
            dm_obj.label = None
            dm_obj.show_label = None
            DM_list = [dm_obj]

        # Case 3: String containing a list of input with their metadata
        elif args.input.startswith('[[') and args.input.endswith(']]'):
            custom_input_metadata = True
            
            try:
                DM_list = list()
                DM_opts_list = list()
                input_list = literal_eval(args.input)
                for dm_data, dm_opts in input_list:
                    dm_file_path = dm_data['path']
                    dm_obj = dm.load_dm_file(dm_file_path)
                    dm_obj.path = dm_file_path
                    dm_obj.label = dm_data['label']
                    dm_obj.show_label = dm_data['show_label']
                    DM_list.append(dm_obj)
                    DM_opts_list.append(dm_opts)

            except ValueError as e:
                if not all([len(x) == 2 for x in input_list]):
                    print("ValueError: Invalid input format. All sub-lists must be a pair of two dictionnaries.\n-> {}".format(str(e)))
                else:
                    print("ValueError: {}".format(str(e)))
                sys.exit(1)

            except SyntaxError as e:
                print("SyntaxError: The input provided is invalid.\n-> {}".format(str(e)))
                sys.exit(1)

        # Verbosity option
        if args.verbose:
            def v_print(*args):
                for arg in args:
                   print (arg),
                print
        else:
            v_print = lambda *a: None      # do-nothing function

        # Output result directory
        if args.outputFolder != '.' and not os.path.exists(args.outputFolder):
            os.makedirs(args.outputFolder)

        # If no custom option file provided, we dump the generated default ones 
        if not args.plotOptionJsonFile or not args.curveOptionJsonFile:
            output_json_path = os.path.join(args.outputFolder, "plotJsonFiles")
            if not os.path.exists(output_json_path):
                os.makedirs(output_json_path)

        # Plot Options
        if not args.plotOptionJsonFile:
            v_print("Generating the default plot options...")
            plot_options_json_path = os.path.join(output_json_path, "plot_options.json")
            # Generating the default_plot_options json config file
            p.gen_default_plot_options(plot_options_json_path, plot_title = args.plotTitle, plot_subtitle = args.plotSubtitle, plot_type = args.plotType)
            # Loading of the plot_options json config file
            plot_opts = p.load_plot_options(plot_options_json_path)
            
        else:
            if os.path.isfile(args.plotOptionJsonFile):
                # Loading of the plot_options json config file
                plot_opts = p.load_plot_options(args.plotOptionJsonFile)
        
        # Curve options 
        curve_options_json_path = os.path.join(output_json_path, "curve_options.json")

        if custom_input_metadata:
            opts_list = DM_opts_list
            with open(curve_options_json_path, 'w') as f:
                f.write(json.dumps(opts_list))

        elif not args.curveOptionJsonFile:
            v_print("Generating the default curves options...")

            # Creation of defaults plot curve options dictionnary (line style opts)
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
            #color = iter(cm.rainbow(np.linspace(0,1,len(DM_list)))) #YYL: error here
            color = cycle(colors)
            lty = cycle(linestyles)
            mkr = cycle(markerstyles)

            for i in range(len(DM_list)):
                new_curve_option = OrderedDict(Curve_opt)
                col = next(color)
                new_curve_option['color'] = col
                new_curve_option['marker'] = next(mkr)
                new_curve_option['markerfacecolor'] = col
                new_curve_option['linestyle'] = next(lty)
                opts_list.append(new_curve_option)

            with open(curve_options_json_path, 'w') as f:
                f.write(json.dumps(opts_list, indent=2, separators=(',', ':')))
        else:
            if os.path.isfile(args.curveOptionJsonFile):
                opts_list = p.load_plot_options(curve_options_json_path)

        print(opts_list)
        # if the length of curve options is less than the length of the dm file list, then print ERROR
        if len(opts_list) < len(DM_list):
            print("ERROR: the number of the curve options is different with the number of the DM objects: ({} < {})".format(len(opts_list), len(DM_list)))
            exit(1)
        else:
            optout = False
            for curve_opts, dm_obj in zip(opts_list, DM_list):
                
                if plot_opts['plot_type'] == 'ROC':
                    met_str = "AUC: {}".format(round(dm_obj.auc,2))
                elif plot_opts['plot_type'] == 'DET':
                    met_str = "EER: {}".format(round(dm_obj.eer,2))

                trr_str = ""
                optout = False
                if dm_obj.sys_res == 'tr':
                    optout = True
                    trr_str = "TRR: {}".format(dm_obj.trr)
                    if plot_opts['plot_type'] == 'ROC':
                        #plot_opts['title'] = "trROC"
                        met_str = "trAUC: {}".format(round(dm_obj.auc,2))
                    elif plot_opts['plot_type'] == 'DET':
                        #plot_opts['title'] = "trDET"
                        met_str = "trEER: {}".format(round(dm_obj.eer,2))

                curve_opts["label"] = None
                if dm_obj.show_label:
                    curve_label = dm_obj.label if dm_obj.label else dm_obj.path
                    measures_list = [_ for _ in [met_str, trr_str] if _ != ''] 

                    if args.noNum:
                        curve_opts["label"] = "{label} ({measures})".format(label=curve_label, measures=', '.join(measures_list))
                    else:
                        curve_opts["label"] = "{label} ({measures}, T#: {nb_target}, NT#: {nb_nontarget})".format(label=curve_label, 
                                                                                                                  measures=', '.join(measures_list),
                                                                                                                  nb_target=dm_obj.t_num, 
                                                                                                                  nb_nontarget=dm_obj.nt_num)

            # Creation of the object setRender (~DetMetricSet)
            configRender = p.setRender(DM_list, opts_list, plot_opts)
            # Creation of the Renderer
            myRender = p.Render(configRender)
            # Plotting
            myfigure = myRender.plot_curve(args.display, multi_fig=args.multiFigs, isOptOut = optout, isNoNumber = args.noNum)

            # save multiple figures if multi_fig == True
            if isinstance(myfigure,list):
                for i,fig in enumerate(myfigure):
                    fig.savefig(args.outputFolder + '_' + args.plotType + '_' + str(i) + '.pdf', bbox_inches='tight')
            else:
                myfigure.savefig(args.outputFolder + '_' + args.plotType + '_all.pdf', bbox_inches='tight')


