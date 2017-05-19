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

        def is_file_specified(x):
            if x == '':
                raise argparse.ArgumentTypeError("{0} not provided".format(x))
            return x

        parser = argparse.ArgumentParser(description='NIST detection scorer.')

        parser.add_argument('-i','--input', type=is_file_specified,
                            help='Input file type [.txt or .dm]',metavar='character')

        parser.add_argument('--outRoot',default='.',
                            help='Report output file (plot and table) path along with the file suffix: [e.g., temp/xx_sys] (default: %(default)s)',metavar='character')

        # Plot Options
        parser.add_argument('--plotType',default='roc', choices=['roc', 'det'],
                            help="Plot option:[roc] and [det] (default: %(default)s)", metavar='character')

        parser.add_argument('--plotTitle',default='Performance',
                            help="Define the plot title (default: %(default)s)", metavar='character')

        parser.add_argument('--plotSubtitle',default='',
                            help="Define the plot subtitle (default: %(default)s)", metavar='character')

        parser.add_argument('--display', action='store_true',
                            help="display plots")

        parser.add_argument('--multiFigs', action='store_true',
                            help="Generate plots (with only one curve) per a partition ")

        parser.add_argument('-c','--curveOption', action='store_true',
                            help="Generate a JSON file for defalut curve options ")

        parser.add_argument('--noNum', action='store_true',
                            help="Do not print the number of target trials and non-target trials on the legend of the plot")

        parser.add_argument('-v', '--verbose', action='store_true',
                            help="Increase output verbosity")

        args = parser.parse_args()

        if not args.display:
            import matplotlib
            matplotlib.use('Agg')
        #import matplotlib.pyplot as plt

        # Verbosity option
        if args.verbose:
            def _v_print(*args):
                for arg in args:
                   print (arg),
                print
        else:
            _v_print = lambda *a: None      # do-nothing function

        global v_print
        v_print = _v_print

        f_list = list()
        try:

            fname = args.input
            if fname.endswith('.txt'):
                #check the file path for loading DM files
#                if '/' not in fname:
#                    file_path = '.'
#                    file_name = fname
#                else:
#                    file_path, file_name = fname.rsplit('/', 1)

                with open(fname) as f:
                    f_list = f.read().splitlines()

                DM_List = list()
                for filename in f_list:
                    if ':' in filename:
                        first_fname, second_fname = filename.rsplit(':', 1)
                        filename = first_fname

                    if not os.path.isfile(filename):
                        print("Filename: {} does not exist".format(filename))
                    else:
                        DM_List.append(dm.load_dm_file(filename))
                    #DM_List = [dm.load_dm_file(file_path +'/'+filename) for filename in f_list]

            elif fname.endswith('.dm'):
                #f_list = [os.path.basename(fname)]
                f_list = [fname]
                DM_List = [dm.load_dm_file(fname)]

        except IOError:
            print("ERROR: There was an error opening the input file [allowed .txt and .dm only]")
            exit(1)

        # the performant result directory
        if '/' not in args.outRoot:
            root_path = '.'
            file_suffix = args.outRoot
        else:
            root_path, file_suffix = args.outRoot.rsplit('/', 1)

        if root_path != '.' and not os.path.exists(root_path):
            os.makedirs(root_path)

        # Generation automatic of a default plot_options json config file
        p_json_path = "./plotJsonFiles"
        if not os.path.exists(p_json_path):
            os.makedirs(p_json_path)

        dict_plot_options_path_name = "./plotJsonFiles/plot_options.json"
        if os.path.isfile(dict_plot_options_path_name):
            # Loading of the plot_options json config file
            plot_opts = p.load_plot_options(dict_plot_options_path_name)
            args.plotType = plot_opts['plot_type']
            plot_opts['title'] = args.plotTitle
            plot_opts['subtitle'] = args.plotSubtitle
            plot_opts['subtitle_fontsize'] = 11
            #print("test plot title1 {}".format(plot_opts['title']))
        else:
            # Generating the default_plot_options json config file
            p.gen_default_plot_options(dict_plot_options_path_name, plot_title = args.plotTitle, plot_subtitle = args.plotSubtitle, plot_type = args.plotType.upper())
            # Loading of the plot_options json config file
            plot_opts = p.load_plot_options(dict_plot_options_path_name)

        # Save the default curve options in JSON format
        dict_curve_options_path_name = "./plotJsonFiles/curve_options.json"

        # if JSON file does not exist or it triggers the defaultOption command,
        # then generates the new JSON curve option file
        if not os.path.isfile(dict_curve_options_path_name) or args.curveOption: #or args.default
            v_print("Generating the default curve options ..")

            # Creation of defaults plot curve options dictionnary (line style opts)
            Curve_opt = OrderedDict([('color', 'red'),
                                     ('linestyle', 'solid'),
                                     ('marker', '.'),
                                     ('markersize', 8),
                                     ('markerfacecolor', 'red'),
                                     ('label',None),
                                     ('antialiased', 'False')])

            # Creating the list of curves options dictionnaries (will be automatic)
            opts_list = list()
            colors = ['red','blue','green','cyan','magenta','yellow','black']
            linestyles = ['solid','dashed','dashdot','dotted']
            # Give a random rainbow color to each curve
            #color = iter(cm.rainbow(np.linspace(0,1,len(DM_List)))) #YYL: error here
            color = cycle(colors)
            lty = cycle(linestyles)
            for i in range(len(DM_List)):
                new_curve_option = OrderedDict(Curve_opt)
                col = next(color)
                new_curve_option['color'] = col
                new_curve_option['markerfacecolor'] = col
                new_curve_option['linestyle'] = next(lty)
                opts_list.append(new_curve_option)

            with open(dict_curve_options_path_name, 'w') as f:
                f.write(json.dumps(opts_list).replace(',', ',\n'))

        # Loading of the curve_options json config file
        opts_list = p.load_plot_options(dict_curve_options_path_name)

        # if the length of curve options is less than the length of the dm file list, then print ERROR
        if len(opts_list) < len(DM_List):
            print("ERROR: the number of the curve options is different with the number of the DM objects: ({} < {})".format(len(opts_list), len(DM_List)))
            print("Please run with '--curveOption'")
            exit(1)
        else:
            for curve_opts, fname, dm_list in zip(opts_list, f_list, DM_List):
                #fname = os.path.basename(fname)
                if ':' in fname:
                    first_fname, second_fname = fname.rsplit(':', 1)
                    fname = second_fname

                if plot_opts['plot_type'] == 'ROC':
                    met_str = " (AUC: " + str(round(dm_list.auc,2))
                elif plot_opts['plot_type'] == 'DET':
                    met_str = " (EER: " + str(round(dm_list.eer,2))

                trr_str = ""
                optout = False
                if dm_list.sys_res == 'tr':
                    optout = True
                    trr_str = ", TRR: " + str(dm_list.trr)
                    if plot_opts['plot_type'] == 'ROC':
                        #plot_opts['title'] = "trROC"
                        met_str = " (trAUC: " + str(round(dm_list.auc,2))
                    elif plot_opts['plot_type'] == 'DET':
                        #plot_opts['title'] = "trDET"
                        met_str = " (trEER: " + str(round(dm_list.eer,2))

                if args.noNum:
                    curve_opts["label"] = fname + met_str + trr_str + ")"
                else:
                    curve_opts["label"] = fname + met_str + trr_str +", T#: "+ str(dm_list.t_num) + ", NT#: "+ str(dm_list.nt_num) +")"
            #curve_opts["label"] = fname + " (AUC: " + str(round(d.auc,2)) + ", T#: "+ str(d.t_num) + ", NT#: "+ str(d.nt_num) + ")"
            # Creation of the object setRender (~DetMetricSet)
            configRender = p.setRender(DM_List, opts_list, plot_opts)
            # Creation of the Renderer
            myRender = p.Render(configRender)
            # Plotting
            myfigure = myRender.plot_curve(args.display, multi_fig=args.multiFigs, isOptOut = optout, isNoNumber = args.noNum)

            # save multiple figures if multi_fig == True
            if isinstance(myfigure,list):
                for i,fig in enumerate(myfigure):
                    fig.savefig(args.outRoot + '_' + args.plotType + '_' + str(i) + '.pdf', bbox_inches='tight')
            else:
                myfigure.savefig(args.outRoot + '_' + args.plotType + '_all.pdf', bbox_inches='tight')

    # Debugging mode
    else:

        print('Starting debug mode ...\n')

#        f = "testcases/NC16_001_query_0.dm" #"test.dm"
#        f_list = ["testcases/NC16_001_query_0.dm", "testcases/NC16_001_query_0.dm"]
        outRoot = './test/sys_01'
        plotType = 'det'
        display = True
        multiFigs = False

#        file = open('testcases/dm_list.txt', 'r')
#        f_list = file.readlines()

        fname = 'testcases/dm_list.txt'
        #fname = "testcases/NC16_001_query_0.dm"

        if fname.endswith('.txt'):
            #check the file path for loading DM files
            if '/' not in fname:
                file_path = '.'
                file_name = fname
            else:
                file_path, file_name = fname.rsplit('/', 1)

            with open(fname) as f:
                f_list = f.read().splitlines()

            DM_List = [dm.load_dm_file(file_path +'/'+filename) for filename in f_list]
        elif fname.endswith('.dm'):
            f_list = [os.path.basename(fname)]
            DM_List = [dm.load_dm_file(fname)]
        else:
            print("ERROR: the file format specified is allowed in this option")

        #if f.endswith('.lst'):

        ### Starting plot options
        # Generation automatic of a default plot_options json config file
        p_json_path = "./plotJsonFiles"
        if not os.path.exists(p_json_path):
            os.makedirs(p_json_path)

        dict_plot_options_path_name = "./plotJsonFiles/plot_options.json"

         # Generating the default_plot_options json config file
        p.gen_default_plot_options(dict_plot_options_path_name, plotType.upper())

        # Loading of the plot_options json config file
        plot_opts = p.load_plot_options(dict_plot_options_path_name)
        ### Ending plot option

        # Save the default curve options at JSON format
        dict_curve_options_path_name = "./plotJsonFiles/curve_options2.json"

        default = False
        if not os.path.exists(dict_curve_options_path_name) or default: #or args.defaultOption
            print("Generating the default curve options")

            # Creation of defaults curve options dictionnary (line style opts)
            Curve_opt = OrderedDict([('color', 'red'),
                                     ('linestyle', 'solid'),
                                     ('marker', '.'),
                                     ('markersize', 8),
                                     ('markerfacecolor', 'red'),
                                     ('antialiased', 'False')])

            opts_list = list()
            colors = ['red','blue','green','cyan','magenta','yellow','black']
            linestyles = ['solid','dashed','dashdot','dotted']
            # Give a random rainbow color to each curve
            #color = iter(cm.rainbow(np.linspace(0,1,len(DM_List)))) #YYL: error here
            color = cycle(colors)
            lty = cycle(linestyles)
            for i in range(len(DM_List)):
                new_curve_option = OrderedDict(Curve_opt)
                col = next(color)
                new_curve_option['color'] = col
                new_curve_option['markerfacecolor'] = col
                new_curve_option['linestyle'] = next(lty)
                new_curve_option["label"] = f_list[i]
                opts_list.append(new_curve_option)

    #        #TODO: How to add key to opts_list
#            json_dict = dict()
#            for i, curve_opts in enumerate(opts_list):
#                json_dict[i]=curve_opts
    #
            with open(dict_curve_options_path_name, 'w') as f:
                f.write(json.dumps(opts_list).replace(',', ',\n'))

##        # Renaming the curves for the legend
#        for curve_opts,query in zip(opts_list,selection.part_query_list):
#            curve_opts["label"] = query

        # Loading of the curve_options json config file
        opts_list = p.load_plot_options(dict_curve_options_path_name)

        # if the length of curve options is less than the length of the dm file list, then print ERROR
        if len(opts_list) < len(DM_List):
            print("ERROR: the number of the curve options is different with the number of the DM objects: ({} < {})".format(len(opts_list), len(DM_List)))
            print("Please run with '--defaultOption'")
            exit(1)
        else:
            for curve_opts, fname, d in zip(opts_list, f_list, DM_List):
                curve_opts["label"] = fname + " (AUC: " + str(round(d.auc,2)) + ", T#: "+ str(d.t_num) + ", NT#: "+ str(d.nt_num) + ")"
            # Creation of the object setRender (~DetMetricSet)
            configRender = p.setRender(DM_List, opts_list, plot_opts)
            # Creation of the Renderer
            myRender = p.Render(configRender)
            # Plotting
            myfigure = myRender.plot_curve(display,multi_fig=multiFigs)


            # save multiple figures if multi_fig == True
            if isinstance(myfigure,list):
                for i,fig in enumerate(myfigure):
                    fig.savefig(outRoot + '_' + plotType + '_' + str(i) + '.pdf')
            else:
                myfigure.savefig(outRoot + '_' + plotType + '_all.pdf')
