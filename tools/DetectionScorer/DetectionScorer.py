#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Date: 10/20/2016
Authors: Yooyoung Lee and Timothee Kheyrkhah

Description: this code calculates performance measures (for points, AUC, and EER)
on system outputs (confidence scores) and return report plot(s) and table(s).
"""

import argparse
import numpy as np
import pandas as pd
import os # os.system("pause") for windows command line
import sys

from matplotlib.pyplot import cm
from collections import OrderedDict
from itertools import cycle

#import time
#from sklearn.metrics import roc_auc_score

lib_path = "../../lib"
sys.path.append(lib_path)
import Render as p
import detMetrics as dm
import Partition as f



########### Command line interface ########################################################

if __name__ == '__main__':

    # Boolean for disabling the command line
    debug_mode_ide = False

    # Command-line mode
    if not debug_mode_ide:

        parser = argparse.ArgumentParser(description='NIST detection scorer.')

        parser.add_argument('-t','--task', default='manipulation', choices=['manipulation', 'removal', 'splice', 'clone'],
        help='Four different types of tasks: [manipulation], [removal], [splice], or [clone] (default: %(default)s)',metavar='character')

        parser.add_argument('--refDir',default='.',
        help='Reference and index file path: [e.g., ../NC2016_Test] (default: %(default)s)',metavar='character')

        parser.add_argument('-r','--inRef',default='reference/manipulation/reference.csv',
        help='Reference csv file name: [e.g., reference/manipulation/reference.csv] (default: %(default)s)',metavar='character')

        parser.add_argument('-x','--inIndex',default='indexes/index.csv',
        help='Task Index csv file name: [e.g., indexes/index.csv] (default: %(default)s)',metavar='character')

        parser.add_argument('--sysDir',default='.',
        help='System output file path: [e.g., /mySysOutputs] (default: %(default)s)',metavar='character')

        parser.add_argument('-s','--inSys',default="",
        help='System output csv file name: [e.g., ~/expid/system_output.csv] (default: %(default)s)',metavar='character')

#        parser.add_argument('-m','--metric',default='all',
#        help="Metric option: [all], [auc], [eer], or [rate] (default: %(default)s)",metavar='character')
        def restricted_float(x):
            x = float(x)
            if x < 0.0 or x > 1.0:
                raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
            return x

        parser.add_argument('--farStop',type=restricted_float, default = 1,
        help="FAR for calculating partial AUC, range [0,1] (default: %(default) for full AUC)",metavar='float')

        parser.add_argument('--outRoot',default='.',
        help='Report output file (plot and table) path along with the file suffix: [e.g., temp/xx_sys] (default: %(default)s)',metavar='character')

        parser.add_argument('--plotType',default='roc', choices=['roc', 'det'],
        help="Plot option:[roc] and [det] (default: %(default)s)", metavar='character')

        parser.add_argument('--dump', action='store_true',
        help="DetMetrics object dumping option")

        parser.add_argument('--display', action='store_true',
        help="display plots")

        factor_group = parser.add_mutually_exclusive_group()

        factor_group.add_argument('-f', '--factor', nargs='*',
        help="Evaluate algorithm performance by given queries.", metavar='character')

        factor_group.add_argument('-fp', '--factorp',
        help="Evaluate algorithm performance with partitions given by one query (syntax : '==[]','<','<=')", metavar='character')

        parser.add_argument('--multiFigs', action='store_true',
        help="Generate plots (with only one curve) per a partition ")

        parser.add_argument('-v', '--verbose', action='store_true',
        help="Increase output verbosity")

        parser.add_argument('--ci', action='store_true',
        help="Calculate Confidence Interval for AUC")

        args = parser.parse_args()

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


        if (not args.factor) and (not args.factorp) and (args.multiFigs is True):
            print("ERROR: The multiFigs option is not available without factors options.")
            exit(1)

       # print("Namespace :\n{}\n".format(args))


        # Loading the reference file
        try:
            myRefFname = args.refDir + "/" + args.inRef
            ref_dtype = {'TaskID':str,
                     'ProbeFileID':str,
                     'ProbeFileName':str,
                     'ProbeMaskFileName':str,
                     'ProbeMaskFileName':str,
                     'DonorFileID':str,
                     'DonorFileName':str,
                     'DonorMaskFileName':str,
                     'IsTarget':str,
                     'ProbePostProcessed':str,
                     'DonorPostProcessed':str,
                     'ManipulationQuality':str,
                     'IsManipulationTypeRemoval':str,
                     'IsManipulationTypeSplice':str,
                     'IsManipulationTypeCopyClone':str,
                     'Collection':str,
                     'BaseFileName':str,
                     'Lighting':str,
                     'IsControl':str,
                     'CorrespondingControlFileName':str,
                     'SemanticConsistency':str}
            myRef = pd.read_csv(myRefFname, sep='|', dtype = ref_dtype)
        except IOError:
            print("ERROR: There was an error opening the reference file")
            exit(1)

        # Loading Index file and system output for SSD and DSD
        # different columns between SSD and DSD
        if args.task in ['manipulation','removal','clone']:
            index_dtype = {'TaskID':str,
                     'ProbeFileID':str,
                     'ProbeFileName':str,
                     'ProbeWidth':np.int64,
                     'ProbeHeight':np.int64}
            sys_dtype = {'ProbeFileID':str,
                     'ConfidenceScore':str, #this should be "string" due to the "nan" value, otherwise "nan"s will have different unique numbers
                     'ProbeOutputMaskFileName':str}
        elif args.task == 'splice':
            index_dtype = {'TaskID':str,
                     'ProbeFileID':str,
                     'ProbeFileName':str,
                     'ProbeWidth':np.int64,
                     'ProbeHeight':np.int64,
                     'DonorFileID':str,
                     'DonorFileName':str,
                     'DonorWidth':np.int64,
                     'DonorHeight':np.int64}
            sys_dtype = {'ProbeFileID':str,
                     'DonorFileID':str,
                     'ConfidenceScore':str, #this should be "string" due to the "nan" value, otherwise "nan"s will have different unique numbers
                     'ProbeOutputMaskFileName':str,
                     'DonorOutputMaskFileName':str}

        try:
            myIndexFname = args.refDir + "/" + args.inIndex
            myIndex = pd.read_csv(myIndexFname, sep='|', dtype = index_dtype)
        except IOError:
            print("ERROR: There was an error opening the index file")
            exit(1)

        try:
            mySysFname = args.sysDir + "/" + args.inSys
            v_print("Sys File Name {}".format(mySysFname))
            mySys = pd.read_csv(mySysFname, sep='|', dtype = sys_dtype)
            #mySys['ConfidenceScore'] = mySys['ConfidenceScore'].astype(str)
            #mySysDir = os.path.dirname(sysFname)
        except IOError:
            print("ERROR: There was an error opening the reference file")
            exit(1)

        # Merge the reference and system output for SSD/DSD reports
        if args.task in ['manipulation','removal','clone']:
            m_df = pd.merge(myRef, mySys, how='left', on='ProbeFileID')
        elif args.task == 'splice':
            m_df = pd.merge(myRef, mySys, how='left', on=['ProbeFileID','DonorFileID'])

        # if the confidence score are 'nan', replace the values with the mininum score
        m_df[pd.isnull(m_df['ConfidenceScore'])] = mySys['ConfidenceScore'].min()
        # convert to the str type to the float type for computations
        m_df['ConfidenceScore'] = m_df['ConfidenceScore'].astype(np.float)

        # the performant result directory
        if '/' not in args.outRoot:
            root_path = '.'
            file_suffix = args.outRoot
        else:
            root_path, file_suffix = args.outRoot.rsplit('/', 1)

        if root_path != '.' and not os.path.exists(root_path):
            os.makedirs(root_path)

        # Partition Mode
        if args.factor or args.factorp:
            v_print("Partition Mode \n")
            partition_mode = True

            if args.task in ['manipulation','removal','clone']:
                subIndex = myIndex[['ProbeFileID', 'ProbeWidth', 'ProbeHeight']] # subset the columns due to duplications
                pm_df = pd.merge(m_df, subIndex, how='left', on= 'ProbeFileID')
            elif args.task == 'splice':
                subIndex = myIndex[['ProbeFileID', 'DonorFileID', 'ProbeWidth', 'ProbeHeight', 'DonorWidth', 'DonorHeight']] # subset the columns due to duplications
                pm_df = pd.merge(m_df, subIndex, how='left', on= ['ProbeFileID','DonorFileID'])


            if args.factor:
                factor_mode = 'f'
                query = args.factor
            elif args.factorp:
                factor_mode = 'fp'
                query = args.factorp

            v_print("Query : {}\n".format(query))
            v_print("Creating partitions...\n")
            selection = f.Partition(pm_df, query, factor_mode, fpr_stop=1, isCI=args.ci)
            DM_List = selection.part_dm_list
            v_print("Number of partitions generated = {}\n".format(len(DM_List)))
            v_print("Rendering csv tables...\n")
            table_df = selection.render_table()
            if isinstance(table_df,list):
                v_print("Number of table DataFrame generated = {}\n".format(len(table_df)))
            if args.factor:
                for i,df in enumerate(table_df):
                    df.to_csv(args.outRoot + '_f_query_' + str(i) + '.csv', index = False)
            elif args.factorp:
                table_df.to_csv(args.outRoot + '_fp_query.csv')

        # No partitions
        else:
            DM = dm.detMetrics(m_df['ConfidenceScore'], m_df['IsTarget'], fpr_stop = 1, isCI=args.ci)

            DM_List = [DM]
            table_df = DM.render_table()
            table_df.to_csv(args.outRoot + '_all.csv', index = False)

        if isinstance(table_df,list):
            print("\nReport tables...\n")
            for i, table in enumerate (table_df):
                print("\nPartition {}:".format(i))
                print(table)
        else:
            print("Report table:\n{}".format(table_df))


        # Generation automatic of a default plot_options json config file
        p_json_path = "./plotJsonFiles"
        if not os.path.exists(p_json_path):
            os.makedirs(p_json_path)

        dict_plot_options_path_name = "./plotJsonFiles/plot_options.json"

         # Generating the default_plot_options json config file
        p.gen_default_plot_options(dict_plot_options_path_name, args.plotType.upper())

        # Loading of the plot_options json config file
        plot_opts = p.load_plot_options(dict_plot_options_path_name)

        # Dumping DetMetrics objects
        if args.dump:
            for i,DM in enumerate(DM_List):
                DM.write(root_path + '/' + file_suffix + '_query_' + str(i) + '.dm')

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


        # Renaming the curves for the legend
        if args.factor or args.factorp: #added by LEE
            for curve_opts,query in zip(opts_list,selection.part_query_list):
                curve_opts["label"] = query


        # Creation of the object setRender (~DetMetricSet)
        configRender = p.setRender(DM_List, opts_list, plot_opts)
        # Creation of the Renderer
        myRender = p.Render(configRender)
        # Plotting
        myfigure = myRender.plot_curve(args.display,multi_fig=args.multiFigs)

        # save multiple figures if multi_fig == True
        if isinstance(myfigure,list):
            for i,fig in enumerate(myfigure):
                fig.savefig(args.outRoot + '_' + args.plotType + '_' + str(i) + '.pdf')
        else:
            myfigure.savefig(args.outRoot + '_' + args.plotType + '_all.pdf')

    # Debugging mode
    else:

        print('Starting debug mode ...\n')

        data_path = "../../data"
        nc_path = "NC2016_Test0613"
        myRefDir = data_path + "/" + nc_path
        task = 'manipulation'
        outRoot = './test/sys_01'
        plotType = 'det'
        display = True
        multiFigs = False
        dump = True
        #fQuery = None
        fpQuery = None
        #fQuery = "Collection==['Nimble-SCI','Nimble-WEB']" "300 <= ProbeWidth"
        #fQuery = "Collection==['Nimble-SCI', 'Nimble-WEB']"
        fQuery = "Collection==['Nimble-WEB'] & 300 <= ProbeWidth"


        if task == 'manipulation':
            refFname = "reference/manipulation/NC2016-manipulation-ref.csv"
            indexFname = "indexes/NC2016-manipulation-index.csv"
            sysFname = "../../data/SystemOutputs/results/dct02.csv"
        if task == 'splice':
            refFname = "reference/splice/NC2016-splice-ref.csv"
            indexFname = "indexes/NC2016-splice-index.csv"
            sysFname = "../../data/SystemOutputs/splice0608/results.csv"

        # Loading the reference file
        try:
            myRefFname = myRefDir + "/" + refFname
            ref_dtype = {'TaskID':str,
                     'ProbeFileID':str,
                     'ProbeFileName':str,
                     'ProbeMaskFileName':str,
                     'ProbeMaskFileName':str,
                     'DonorFileID':str,
                     'DonorFileName':str,
                     'DonorMaskFileName':str,
                     'IsTarget':str,
                     'ProbePostProcessed':str,
                     'DonorPostProcessed':str,
                     'ManipulationQuality':str,
                     'IsManipulationTypeRemoval':str,
                     'IsManipulationTypeSplice':str,
                     'IsManipulationTypeCopyClone':str,
                     'Collection':str,
                     'BaseFileName':str,
                     'Lighting':str,
                     'IsControl':str,
                     'CorrespondingControlFileName':str,
                     'SemanticConsistency':str}
            myRef = pd.read_csv(myRefFname, sep='|', dtype = ref_dtype)
        except IOError:
            print("ERROR: There was an error opening the reference file")
            exit(1)

        # Loading Index file and system output for SSD and DSD
        # different columns between SSD and DSD
        if task in ['manipulation','removal','clone']:
            index_dtype = {'TaskID':str,
                     'ProbeFileID':str,
                     'ProbeFileName':str,
                     'ProbeWidth':np.int64,
                     'ProbeHeight':np.int64}
            sys_dtype = {'ProbeFileID':str,
                     'ConfidenceScore':str, #this should be "string" due to the "nan" value, otherwise "nan"s will have different unique numbers
                     'ProbeOutputMaskFileName':str}
        elif task == 'splice':
            index_dtype = {'TaskID':str,
                     'ProbeFileID':str,
                     'ProbeFileName':str,
                     'ProbeWidth':np.int64,
                     'ProbeHeight':np.int64,
                     'DonorFileID':str,
                     'DonorFileName':str,
                     'DonorWidth':np.int64,
                     'DonorHeight':np.int64}
            sys_dtype = {'ProbeFileID':str,
                     'DonorFileID':str,
                     'ConfidenceScore':str, #this should be "string" due to the "nan" value, otherwise "nan"s will have different unique numbers
                     'ProbeOutputMaskFileName':str,
                     'DonorOutputMaskFileName':str}

        try:
            myIndexFname = myRefDir + "/" + indexFname
            myIndex = pd.read_csv(myIndexFname, sep='|', dtype = index_dtype)
        except IOError:
            print("ERROR: There was an error opening the index file")
            exit(1)

        try:
            mySys = pd.read_csv(sysFname, sep='|', dtype = sys_dtype)
            #mySys['ConfidenceScore'] = mySys['ConfidenceScore'].astype(str)
            mySysDir = os.path.dirname(sysFname)
        except IOError:
            print("ERROR: There was an error opening the reference file")
            exit(1)

        # Create SSD/DSD reports
        if task in ['manipulation','removal','clone']:
            m_df = pd.merge(myRef, mySys, how='left', on='ProbeFileID')
        elif task== 'splice':
            m_df = pd.merge(myRef, mySys, how='left', on=['ProbeFileID','DonorFileID'])

        # if the values are 'nan', replace the values with the mininum score
        m_df[pd.isnull(m_df['ConfidenceScore'])] = mySys['ConfidenceScore'].min()
        # convert to the str type to the float type for computations
        m_df['ConfidenceScore'] = m_df['ConfidenceScore'].astype(np.float)

        root_path, file_suffix = outRoot.rsplit('/', 1)
        if root_path != '.' and not os.path.exists(root_path):
            os.makedirs(root_path)

        #TODO: add Error message if both f and fp are specified.
        # Partition Mode
        if fQuery or fpQuery:
            print("Partition Mode")
            partition_mode = True

            if task in ['manipulation','removal','clone']:
                subIndex = myIndex[['ProbeFileID', 'ProbeWidth', 'ProbeHeight']] # ubset the columns due to duplications
                pm_df = pd.merge(m_df, subIndex, how='left', on= 'ProbeFileID')
            elif task == 'splice':
                subIndex = myIndex[['ProbeFileID', 'DonorFileID', 'ProbeWidth', 'ProbeHeight', 'DonorWidth', 'DonorHeight']] # ubset the columns due to duplications
                pm_df = pd.merge(m_df, subIndex, how='left', on= ['ProbeFileID','DonorFileID'])

            if fQuery:
                factor_mode = 'f'
                query = [fQuery]
            elif fpQuery:
                factor_mode = 'fp'
                query = fpQuery

            print("Query : {}".format(query))
            print("Creating partitions...")
            selection = f.Partition(pm_df, query, factor_mode, fpr_stop=1, isCI=True)
            DM_List = selection.part_dm_list
            print("Number of partitions generated = {}".format(len(DM_List)))
            print("Rendering csv tables...")
            table_df = selection.render_table()
            if isinstance(table_df,list):
                print("Number of table DataFrame generated = {}".format(len(table_df)))
            if fQuery:
                for i,df in enumerate(table_df):
                    df.to_csv(outRoot + '_f_query_' + str(i) + '.csv', index = False)
            elif fpQuery:
                table_df.to_csv(outRoot + '_fp_query.csv')

        # No partitions
        else:
            DM = dm.detMetrics(m_df['ConfidenceScore'], m_df['IsTarget'], fpr_stop = 1, isCI=True)
            DM_List = [DM]
            table_df = DM.render_table()
            table_df.to_csv(outRoot + '_all.csv', index = False)

        if isinstance(table_df,list):
            print("\nReport tables...")
            for i, table in enumerate (table_df):
                print("\nPartition {}:".format(i))
                print(table)
        else:
            print("Report table:\n{}".format(table_df))


        # Generation automatic of a default plot_options json config file
        p_json_path = "./plotJsonFiles"
        if not os.path.exists(p_json_path):
            os.makedirs(p_json_path)

        dict_plot_options_path_name = "./plotJsonFiles/plot_options.json"

         # Generating the default_plot_options json config file
        p.gen_default_plot_options(dict_plot_options_path_name, plotType.upper())

        # Loading of the plot_options json config file
        plot_opts = p.load_plot_options(dict_plot_options_path_name)

        # Dumping DetMetrics objects
        if dump:
            for i,DM in enumerate(DM_List):
                DM.write(outRoot + '_query_' + str(i) + '.dm')

        # Creation of defaults plot curve options dictionnary (line style opts)
        Curve_opt = OrderedDict([('color', 'red'),
                                 ('linestyle', 'solid'),
                                 ('marker', '.'),
                                 ('markersize', 8),
                                 ('markerfacecolor', 'red'),
                                 ('antialiased', 'False')])

        # Creating the list of curves options dictionnaries (will be automatic)
        opts_list = list()
        colors = ['red','blue','green','cyan','magenta','yellow','black']
        linestyles = ['solid','dashed','dashdot','dotted']
        # Give a random rainbow color to each curve
        color = iter(cm.rainbow(np.linspace(0,1,len(DM_List))))
        lty = cycle(linestyles)
#        color = cycle(colors)
        for i in range(len(DM_List)):
            new_curve_option = OrderedDict(Curve_opt)
            col = next(color)
            new_curve_option['color'] = col
            new_curve_option['markerfacecolor'] = col
            new_curve_option['linestyle'] = next(lty)
            opts_list.append(new_curve_option)

        # Renaming the curves for the legend
        for curve_opts,query in zip(opts_list,selection.part_query_list):
            curve_opts["label"] = query

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
