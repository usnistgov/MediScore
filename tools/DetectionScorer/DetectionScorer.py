"""
* Date: 10/20/2016
* Authors: Yooyoung Lee and Timothee Kheyrkhah

* Description: this code calculates performance measures (for points, AUC, and EER)
on system outputs (confidence scores) and return report plot(s) and table(s).

* Disclaimer:
This software was developed at the National Institute of Standards
and Technology (NIST) by employees of the Federal Government in the
course of their official duties. Pursuant to Title 17 Section 105
of the United States Code, this software is not subject to copyright
protection and is in the public domain. NIST assumes no responsibility
whatsoever for use by other parties of its source code or open source
server, and makes no guarantees, expressed or implied, about its quality,
reliability, or any other characteristic."

"""

import argparse
import numpy as np
import pandas as pd
import os  # os.system("pause") for windows command line
import sys

#from matplotlib.pyplot import cm
from collections import OrderedDict
from itertools import cycle

#lib_path = "../../lib"
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib")
sys.path.append(lib_path)
import Render as p
import detMetrics as dm
import Partition as f
#import time


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

        def restricted_float(x):
            x = float(x)
            if x < 0.0 or x > 1.0:
                raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
            return x

        def restricted_ci_value(x):
            if x == '':
                raise argparse.ArgumentTypeError("{0} not provided".format(x))

            x = float(x)
            if x <= 0.8 or x >= 0.99:
                raise argparse.ArgumentTypeError("%r not in range [0.80, 0.99]" % (x,))

            return x

        def restricted_dprime_level(x):
            if x == '':
                raise argparse.ArgumentTypeError("{0} not provided".format(x))

            x = float(x)
            if x > 0.3 or x < 0:
                raise argparse.ArgumentTypeError("%r not in range [0.0, 0.3]" % (x,))

            return x

        parser = argparse.ArgumentParser(description='NIST detection scorer.')

        # Task Type Options
        parser.add_argument('-t', '--task', default='manipulation',
                            # add provenanceFiltering and provenance in future
                            choices=['manipulation', 'splice'],
                            help='Define the target manipulation task type for evaluation:[manipulation] and [splice] (default: %(default)s)', metavar='character')

        # Input Options
        parser.add_argument('--refDir', default='.',
                            help='Specify the reference and index data path: [e.g., ../NC2016_Test] (default: %(default)s)', metavar='character')
        # type=lambda x: is_dir(parser, x))#Optional

        parser.add_argument('-r', '--inRef', default='', type=is_file_specified,
                            help='Specify the reference CSV file (under the refDir folder) that contains the ground-truth and metadata information: [e.g., reference/manipulation/reference.csv]', metavar='character')
        # type=lambda x: is_file(parser, x))# Mandatory

        parser.add_argument('-x', '--inIndex', default='', type=is_file_specified,
                            help='Specify the index CSV file: [e.g., indexes/index.csv] (default: %(default)s)', metavar='character')

        parser.add_argument('--sysDir', default='.',
                            help='Specify the system output data path: [e.g., /mySysOutputs] (default: %(default)s)', metavar='character')  # Optional

        parser.add_argument('-s', '--inSys', default='', type=is_file_specified,
                            help='Specify the CSV file of the system performance result formatted according to the specification: [e.g., ~/expid/system_output.csv] (default: %(default)s)', metavar='character')

        # Metric Options
        parser.add_argument('--farStop', type=restricted_float, default=0.1,
                            help='Specify the stop point of FAR for calculating partial AUC, range [0,1] (default: %(default) FAR 10%)', metavar='float')

        # TODO: relation between ci and ciLevel
        parser.add_argument('--ci', action='store_true',
                            help="Calculate the lower and upper confidence interval for AUC if this option is specified. The option will slowdown the speed due to the bootstrapping method.")

        parser.add_argument('--ciLevel', type=restricted_ci_value, default=0.9,
                            help="Calculate the lower and upper confidence interval with the specified confidence level, The option will slowdown the speed due to the bootstrapping method.", metavar='float')

        parser.add_argument('--dLevel', type=restricted_dprime_level, default=0.0,
                            help="Define the lower and upper exclusions for d-prime calculation", metavar='float')

        # Output Options
        parser.add_argument('-o', '--outRoot', default='.',
                            help='Specify the report output path and the file name prefix for saving the plot(s) and table (s). For example, if you specify "--outRoot test/NIST_001", you will find the plot "NIST_001_det.png" and the table "NIST_001_report.csv" in the "test" folder: [e.g., temp/xx_sys] (default: %(default)s)', metavar='character')

        parser.add_argument('--outMeta', action='store_true',
                            help="Save the CSV file with the system scores with minimal metadata")

        parser.add_argument('--outAllmeta', action='store_true',
                            help="Save the CSV file with the system scores with all metadata")

        parser.add_argument('--dump', action='store_true',
                            help="Save the dump files (formatted as a binary) that contains a list of FAR, FPR, TPR, threshold, AUC, and EER values. The purpose of the dump files is to load the point values for further analysis without calculating the values again.")

        parser.add_argument('-v', '--verbose', action='store_true',
                            help="Print output with procedure messages on the command-line if this option is specified.")

        # Plot Options
        parser.add_argument('--plotTitle', default='Performance',
                            help="Define the plot title (default: %(default)s)", metavar='character')

        parser.add_argument('--plotSubtitle', default='',
                            help="Define the plot subtitle (default: %(default)s)", metavar='character')

        parser.add_argument('--plotType', default='', choices=['roc', 'det'],
                            help="Define the plot type:[roc] and [det] (default: %(default)s)", metavar='character')

        parser.add_argument('--display', action='store_true',
                            help="Display a window with the plot (s) on the command-line if this option is specified.")

        parser.add_argument('--multiFigs', action='store_true',
                            help="Generate plots (with only one curve) per a partition ")
        # Custom Plot Options
        parser.add_argument('--configPlot', action='store_true',
                            help="Open a JSON file that allows the user to customize the plot (e.g. change the title font size) by augmenting the json files located in the 'plotJsonFiles' folder.")

        # Performance Evaluation by Query Options
        factor_group = parser.add_mutually_exclusive_group()

        factor_group.add_argument('-q', '--query', nargs='*',
                                  help="Evaluate algorithm performance on a partitioned dataset (or subset) using multiple queries. Depending on the number (N) of queries, the option generates N report tables (CSV) and one plot (PDF) that contains N curves.", metavar='character')

        factor_group.add_argument('-qp', '--queryPartition',
                                  help="Evaluate algorithm performance on a partitioned dataset (or subset) using one query. Depending on the number (M) of partitions provided by the cartesian product on query conditions, this option generates a single report table (CSV) that contains M partition results and one plot that contains M curves. (syntax retriction: '==[]','<','<=')", metavar='character')

        factor_group.add_argument('-qm', '--queryManipulation', nargs='*',
                                  help="This option is similar to the '-q' option; however, the queries are only applied to the target trials (IsTarget == 'Y') and use all of non-target trials. Depending on the number (N) of queries, the option generates N report tables (CSV) and one plot (PDF) that contains N curves.", metavar='character')

        parser.add_argument('--optOut', action='store_true',
                            help="Evaluate algorithm performance on trials where the IsOptOut value is 'N' only.")

        parser.add_argument('--noNum', action='store_true',
                            help="Do not print the number of target trials and non-target trials on the legend of the plot")

        # Note that this requires different mutually exclusive gropu to use both -qm and -qn at the same time
#        parser.add_argument('-qn', '--queryNonManipulation',
# help="Provide a simple interface to evaluate algorithm performance by
# given query (for filtering non-target trials)", metavar='character')

        args = parser.parse_args()
        #print("Namespace :\n{}\n".format(args))

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

        if (not args.query) and (not args.queryPartition) and (not args.queryManipulation) and (args.multiFigs is True):
            print("ERROR: The multiFigs option is not available without query options.")
            exit(1)

        # Loading the reference file
        try:
            myRefFname = os.path.join(args.refDir, args.inRef)
            myRef = pd.read_csv(myRefFname, sep='|', low_memory=False)
            myRefDir = os.path.dirname(myRefFname)  # to use for loading JTJoin and JTMask files
        except IOError:
            print("ERROR: There was an error opening the reference csv file '" + myRefFname + "'")
            exit(1)

        # Loading the JTjoin and JTmask file
        myJTJoinFname = os.path.join(args.refDir, str(
            args.inRef.split('.')[:-1]).strip("['']") + '-probejournaljoin.csv')
        myJTMaskFname = os.path.join(args.refDir, str(
            args.inRef.split('.')[:-1]).strip("['']") + '-journalmask.csv')
        v_print("Ref file name {}".format(myRefFname))
        v_print("JTJoin file name {}".format(myJTJoinFname))
        v_print("JTMask file name {}".format(myJTMaskFname))

        # check existence of the JTjoin and JTmask csv files
        if os.path.isfile(myJTJoinFname) and os.path.isfile(myJTMaskFname):
            myJTJoin = pd.read_csv(myJTJoinFname, sep='|', low_memory=False)
            myJTMask = pd.read_csv(myJTMaskFname, sep='|', low_memory=False)
        else:
            v_print(
                "Either JTjoin or JTmask csv file do not exist and merging process with the reference file will be skipped")

        # Loading the index file
        try:
            myIndexFname = os.path.join(args.refDir, args.inIndex)
           # myIndex = pd.read_csv(myIndexFname, sep='|', dtype = index_dtype)
            myIndex = pd.read_csv(myIndexFname, sep='|', low_memory=False)

        except IOError:
            print("ERROR: There was an error opening the index csv file")
            exit(1)

        # Loading system output for SSD and DSD due to different columns between SSD and DSD
        try:

            if args.task in ['manipulation']:
                sys_dtype = {'ProbeFileID': str,
                             # this should be "string" due to the "nan" value, otherwise "nan"s will
                             # have different unique numbers
                             'ConfidenceScore': str,
                             'ProbeOutputMaskFileName': str}
            elif args.task in ['splice']:
                sys_dtype = {'ProbeFileID': str,
                             'DonorFileID': str,
                             # this should be "string" due to the "nan" value, otherwise "nan"s will
                             # have different unique numbers
                             'ConfidenceScore': str,
                             'ProbeOutputMaskFileName': str,
                             'DonorOutputMaskFileName': str}

            mySysFname = os.path.join(args.sysDir, args.inSys)
            v_print("Sys File Name {}".format(mySysFname))
            mySys = pd.read_csv(mySysFname, sep='|', dtype=sys_dtype, low_memory=False)
            #mySys['ConfidenceScore'] = mySys['ConfidenceScore'].astype(str)
        except IOError:
            print("ERROR: There was an error opening the system output csv file")
            exit(1)

        # merge the reference and system output for SSD/DSD reports
        if args.task in ['manipulation']:
            m_df = pd.merge(myRef, mySys, how='left', on='ProbeFileID')
        elif args.task in ['splice']:
            m_df = pd.merge(myRef, mySys, how='left', on=['ProbeFileID', 'DonorFileID'])

        # if the confidence scores are 'nan', replace the values with the mininum score
        m_df[pd.isnull(m_df['ConfidenceScore'])] = mySys['ConfidenceScore'].min()
        # convert to the str type to the float type for computations
        m_df['ConfidenceScore'] = m_df['ConfidenceScore'].astype(np.float)

        # to calculate TRR
        #total_num = m_df.shape[0]

        # if OptOut has chosen, all of queries should be applied
        # print(list(myIndex))

        # the performers' result directory
        if '/' not in args.outRoot:
            root_path = '.'
            file_suffix = args.outRoot
        else:
            root_path, file_suffix = args.outRoot.rsplit('/', 1)

        if root_path != '.' and not os.path.exists(root_path):
            os.makedirs(root_path)

        # grep system column names
        sys_cols_no_overlap = [c for c in mySys.columns if c not in myRef.columns]
        sys_cols_overlap = [c for c in mySys.columns if c in myRef.columns]
        # print(sys_cols_no_overlap)
        # print(sys_cols_overlap)
        index_cols_no_overlap = [c for c in myIndex.columns if c not in myRef.columns]
        index_cols_overlap = [c for c in myIndex.columns if c in myRef.columns]
        # print(index_cols_no_overlap)
        # print(index_cols_overlap)

        # merge the reference and index csv only
        index_m_df = pd.merge(m_df, myIndex, how='inner', on=index_cols_overlap)
        # print(index_m_df.columns)
        sub_pm_df = index_m_df[index_cols_overlap +
                               index_cols_no_overlap + ["IsTarget"] + sys_cols_no_overlap]
        # print(sub_pm_df.columns)

        if args.outAllmeta:  # save all metadata for analysis purpose
            index_m_df.to_csv(args.outRoot + '_allmeta.csv', index=False, sep='|')
        # save subset of metadata for analysis purpose
        if args.outMeta:
            sub_pm_df.to_csv(args.outRoot + '_subset_meta.csv', index=False, sep='|')

        #if args.outMeta and set(mani_meta_list).issubset(index_m_df.columns):
        #    sub_pm_df = index_m_df[index_cols_overlap +
        #                           index_cols_no_overlap + ["IsTarget"] + sys_cols_no_overlap]
            # print(sub_pm_df.columns)

        total_num = index_m_df.shape[0]
        v_print("Total data number: {}".format(total_num))

        sys_response = 'all'  # to distinguish use of the optout
        # Partition Mode
        if args.query or args.queryPartition or args.queryManipulation:  # add or targetManiTypeSet or nontargetManiTypeSet
            v_print("Query Mode ... \n")
            partition_mode = True
            # SSD
            if args.task in ['manipulation']:
                # if the files exist, merge the JTJoin and JTMask csv files with the
                # reference and index file
                if os.path.isfile(myJTJoinFname) and os.path.isfile(myJTMaskFname):
                    v_print("Merging the JournalJoin and JournalMask csv file with the reference files ...\n")
                    # merge the JournalJoinTable and the JournalMaskTable (this section should
                    # be inner join)
                    jt_meta = pd.merge(myJTJoin, myJTMask, how='left', on=[
                                       'JournalName', 'StartNodeID', 'EndNodeID'])  # JournalName instead of JournalID
                    # v_print("JT meta: {}".format(jt_meta.shape))
                    # merge the dataframes above
                    index_m_df = pd.merge(index_m_df, jt_meta, how='left', on='ProbeFileID')
                    # Removing duplicates in case the data were merged by the JTmask metadata, not for splice
                    # index_m_df = index_m_df.drop_duplicates('ProbeFileID') #only applied to manipulation
            # don't need JTJoin and JTMask for splice?

            if args.query:
                query_mode = 'q'
                query = args.query
            elif args.queryPartition:
                query_mode = 'qp'
                query = args.queryPartition
            elif args.queryManipulation:
                query_mode = 'qm'
                query = args.queryManipulation

            if args.optOut:
                sys_response = 'tr'
                if "IsOptOut" in index_m_df.columns:
                    index_m_df = index_m_df.query(" IsOptOut==['N', 'Localization'] ")
                elif "ProbeStatus" in index_m_df.columns:
                    index_m_df = index_m_df.query(
                        " ProbeStatus==['Processed', 'NonProcessed', 'OptOutAll', 'OptOutDetection', 'OptOutLocalization'] ")

            v_print("Query : {}\n".format(query))
            v_print("Creating partitions...\n")
            selection = f.Partition(index_m_df, query, query_mode, fpr_stop=args.farStop, isCI=args.ci,
                                    ciLevel=args.ciLevel, total_num=total_num, sys_res=sys_response, task=args.task)
            DM_List = selection.part_dm_list
            v_print("Number of partitions generated = {}\n".format(len(DM_List)))
            v_print("Rendering csv tables...\n")
            table_df = selection.render_table()
            if isinstance(table_df, list):
                v_print("Number of table DataFrame generated = {}\n".format(len(table_df)))
            if args.query:
                for i, df in enumerate(table_df):
                    df.to_csv(args.outRoot + '_q_query_' + str(i) +
                              '_report.csv', index=False, sep='|')
            elif args.queryPartition:
                table_df.to_csv(args.outRoot + '_qp_query_report.csv', sep='|')
            elif args.queryManipulation:
                for i, df in enumerate(table_df):
                    df.to_csv(args.outRoot + '_qm_query_' + str(i) +
                              '_report.csv', index=False, sep='|')

        # No partitions
        else:

            if args.optOut:
                sys_response = 'tr'
                if "IsOptOut" in index_m_df.columns:
                    index_m_df = index_m_df.query(" IsOptOut==['N', 'Localization'] ")
                elif "ProbeStatus" in index_m_df.columns:
                    index_m_df = index_m_df.query(
                        " ProbeStatus==['Processed', 'NonProcessed', 'OptOutAll', 'OptOutDetection', 'OptOutLocalization'] ")

            DM = dm.detMetrics(index_m_df['ConfidenceScore'], index_m_df['IsTarget'], fpr_stop=args.farStop,
                               isCI=args.ci, ciLevel=args.ciLevel, dLevel=args.dLevel, total_num=total_num, sys_res=sys_response)

            DM_List = [DM]
            table_df = DM.render_table()
            table_df.to_csv(args.outRoot + '_all_report.csv', index=False, sep='|')

        if isinstance(table_df, list):
            print("\nReport tables:\n")
            for i, table in enumerate(table_df):
                print("\nPartition {}:".format(i))
                print(table)
        else:
            print("Report table:\n{}".format(table_df))

        # Generating a default plot_options json config file
        p_json_path = "./plotJsonFiles"
        if not os.path.exists(p_json_path):
            os.makedirs(p_json_path)
        dict_plot_options_path_name = "./plotJsonFiles/plot_options.json"

        # Fixed: if plotType is indicated, then should be generated.
        if args.plotType == '' and os.path.isfile(dict_plot_options_path_name):
            # Loading of the plot_options json config file
            plot_opts = p.load_plot_options(dict_plot_options_path_name)
            args.plotType = plot_opts['plot_type']
            plot_opts['title'] = args.plotTitle
            plot_opts['subtitle'] = args.plotSubtitle
            plot_opts['subtitle_fontsize'] = 11
            #print("test plot title1 {}".format(plot_opts['title']))
        else:
            if args.plotType == '':
                args.plotType = 'roc'
            p.gen_default_plot_options(dict_plot_options_path_name, plot_title=args.plotTitle,
                                       plot_subtitle=args.plotSubtitle, plot_type=args.plotType.upper())
            plot_opts = p.load_plot_options(dict_plot_options_path_name)
            #print("test plot title2 {}".format(plot_opts['title']))

        # opening of the plot_options json config file from command-line
        if args.configPlot:
            p.open_plot_options(dict_plot_options_path_name)

        # Dumping DetMetrics objects
        if args.dump:
            for i, DM in enumerate(DM_List):
                DM.write(root_path + '/' + file_suffix + '_query_' + str(i) + '.dm')

        # Creation of defaults plot curve options dictionnary (line style opts)
        Curve_opt = OrderedDict([('color', 'red'),
                                 ('linestyle', 'solid'),
                                 ('marker', '.'),
                                 ('markersize', 6),
                                 ('markerfacecolor', 'red'),
                                 ('label', None),
                                 ('antialiased', 'False')])

        # Creating the list of curves options dictionnaries (will be automatic)
        opts_list = list()
        colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'sienna', 'navy', 'grey',
                  'darkorange', 'c', 'peru', 'y', 'pink', 'purple', 'lime', 'magenta', 'olive', 'firebrick']
        linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
        markerstyles = ['.', '+', 'x', 'd', '*', 's', 'p']
        # Give a random rainbow color to each curve
        # color = iter(cm.rainbow(np.linspace(0,1,len(DM_List)))) #YYL: error here
        color = cycle(colors)
        lty = cycle(linestyles)
        mkr = cycle(markerstyles)
        for i in range(len(DM_List)):
            new_curve_option = OrderedDict(Curve_opt)
            col = next(color)
            new_curve_option['color'] = col
            new_curve_option['marker'] = next(mkr)
            new_curve_option['markerfacecolor'] = col
            new_curve_option['linestyle'] = next(lty)
            opts_list.append(new_curve_option)

        if args.optOut:
            if plot_opts['plot_type'] == 'ROC':
                plot_opts['title'] = "tr" + args.plotTitle
            elif plot_opts['plot_type'] == 'DET':
                plot_opts['title'] = "tr" + args.plotTitle

        # Renaming the curves for the legend
        if args.query or args.queryPartition or args.queryManipulation:
            for curve_opts, query, dm_list in zip(opts_list, selection.part_query_list, DM_List):
                trr_str = ""
                #print("plottype {}".format(plot_opts['plot_type']))
                if plot_opts['plot_type'] == 'ROC':
                    met_str = " (AUC: " + str(round(dm_list.auc, 2))
                elif plot_opts['plot_type'] == 'DET':
                    met_str = " (EER: " + str(round(dm_list.eer, 2))

                if args.optOut:
                    trr_str = ", TRR: " + str(dm_list.trr)
                    if plot_opts['plot_type'] == 'ROC':
                        #plot_opts['title'] = "trROC"
                        met_str = " (trAUC: " + str(round(dm_list.auc, 2))
                    elif plot_opts['plot_type'] == 'DET':
                        #plot_opts['title'] = "trDET"
                        met_str = " (trEER: " + str(round(dm_list.eer, 2))
                if args.noNum:
                    curve_opts["label"] = query + met_str + trr_str + ")"
                else:
                    curve_opts["label"] = query + met_str + trr_str + ", T#: " + \
                        str(dm_list.t_num) + ", NT#: " + str(dm_list.nt_num) + ")"

        # Creation of the object setRender (~DetMetricSet)
        configRender = p.setRender(DM_List, opts_list, plot_opts)
        # Creation of the Renderer
        myRender = p.Render(configRender)
        # Plotting
        myfigure = myRender.plot_curve(
            args.display, multi_fig=args.multiFigs, isOptOut=args.optOut, isNoNumber=args.noNum)

        # save multiple figures if multi_fig == True
        if isinstance(myfigure, list):
            for i, fig in enumerate(myfigure):
                fig.savefig(args.outRoot + '_' + args.plotType +
                            '_' + str(i) + '.pdf', bbox_inches='tight')
        else:
            myfigure.savefig(args.outRoot + '_' + args.plotType + '_all.pdf', bbox_inches='tight')

    # Debugging mode
    else:
        # This section need to be reimplement later
        print('Starting debug mode ...\n')

        refDir = '/Users/yunglee/YYL/MEDIFOR/data'
        sysDir = '../../data/test_suite/detectionScorerTests'
        task = 'manipulation'
        outRoot = './testcases/sys_01'
        farStop = 1
        ci = True
        ciLevel = 0.9
        dLevel = 0
        plotType = 'roc'
        display = True
        multiFigs = False
        dump = False
        verbose = False
        args_optOut = True
        args_queryManipulation = None
        args_query = None
        args_queryPartition = None
        plotTitle = "Test"
        #args_queryManipulation = ["Purpose ==['add']", "Purpose ==['remove']"]
#       factor = ["Purpose ==['remove', 'splice', 'add']"]
#        queryManipulation = "Operation ==['PasteSplice', 'FillContentAwareFill']"
#        queryManipulation = "SemanticLevel ==['PasteSplice', 'FillContentAwareFill']"targetFilter
#        factor = "Purpose ==['remove']"
        #args_queryManipulation = ["Purpose ==['add']", "Purpose ==['remove']"]
#        factor = ["Purpose ==['remove', 'splice', 'add']","Operation ==['PasteSplice', 'FillContentAwareFill']"]
#        print("f query {}".format(factor))

#        queryPartition = "Purpose ==['remove', 'splice']"

        if (not args_query) and (not args_queryPartition) and (multiFigs is True):
            print("ERROR: The multiFigs option is not available without querys options.")
            exit(1)

        if task == 'manipulation':
            inRef = "NC2017-1215/NC2017-manipulation-ref.csv"
            inIndex = "NC2017-1215/NC2017-manipulation-index.csv"
            inJTJoin = "NC2017-manipulation-ref-probejournaljoin.csv"
            inJTMask = "NC2017-manipulation-ref-journalmask.csv"
            #inSys = "baseline/Base_NC2017_Manipulation_ImgOnly_p-copymove_01.csv"
            inSys = "baseline/Base_NC2017_Manipulation_ImgOnly_p-copymove_01_optout.csv"
            #inSys = "Base_NC2016_Manipulation_ImgOnly_p-dct_02_optout.csv"

        # Loading the reference file
        try:

            myRefFname = refDir + "/" + inRef
            #myRef = pd.read_csv(myRefFname, sep='|', dtype = ref_dtype)
            myRef = pd.read_csv(myRefFname, sep='|')
            myRefDir = os.path.dirname(myRefFname)
            #print ("Ref path {}".format(myRefDir))
        except IOError:
            print("ERROR: There was an error opening the reference csv file")
            exit(1)

#        try:
#
#            myJTJoinFname = myRefDir + "/" + inJTJoin
#            myJTJoin = pd.read_csv(myJTJoinFname, sep='|')
#        except IOError:
#            print("ERROR: There was an error opening the JournalJoin csv file")
#            exit(1)
#
#        try:
#
#            myJTMaskFname = myRefDir + "/" + inJTMask
#            myJTMask = pd.read_csv(myJTMaskFname, sep='|')
#        except IOError:
#            print("ERROR: There was an error opening the JournalMask csv file")
#            exit(1)
        # check existence of the JTjoin csv file and then load the file
        #inJTJoin = "NC2017-manipulation-ref-probejournaljoin.csv"
        #inJTMask = "NC2017-manipulation-ref-journalmask.csv"

        #myJTJoinFname = myRefDir + "/" + inJTJoin
        #myJTMaskFname = myRefDir + "/" + inJTMask

        myJTJoinFname = refDir + "/" + \
            str(inRef.split('.')[:-1]).strip("['']") + '-probejournaljoin.csv'
        myJTMaskFname = refDir + "/" + str(inRef.split('.')[:-1]).strip("['']") + '-journalmask.csv'

        if os.path.isfile(myJTJoinFname) and os.path.isfile(myJTMaskFname):
            myJTJoin = pd.read_csv(myJTJoinFname, sep='|')
            myJTMask = pd.read_csv(myJTMaskFname, sep='|')
        else:
            print("Note: either JTjoin or JTmask csv file do not exist and merging with the reference file will be skipped")

        try:

            myIndexFname = refDir + "/" + inIndex
           # myIndex = pd.read_csv(myIndexFname, sep='|', dtype = index_dtype)
            myIndex = pd.read_csv(myIndexFname, sep='|')
        except IOError:
            print("ERROR: There was an error opening the index csv file")
            exit(1)

        try:
            # Loading system output for SSD and DSD due to different columns between SSD and DSD
            if task in ['manipulation']:
                sys_dtype = {'ProbeFileID': str,
                             # this should be "string" due to the "nan" value, otherwise "nan"s will
                             # have different unique numbers
                             'ConfidenceScore': str,
                             'ProbeOutputMaskFileName': str}
            elif task in ['splice']:
                sys_dtype = {'ProbeFileID': str,
                             'DonorFileID': str,
                             # this should be "string" due to the "nan" value, otherwise "nan"s will
                             # have different unique numbers
                             'ConfidenceScore': str,
                             'ProbeOutputMaskFileName': str,
                             'DonorOutputMaskFileName': str}
            mySysFname = sysDir + "/" + inSys
            print("Sys File Name {}".format(mySysFname))
            mySys = pd.read_csv(mySysFname, sep='|', dtype=sys_dtype)
            #mySys['ConfidenceScore'] = mySys['ConfidenceScore'].astype(str)
        except IOError:
            print("ERROR: There was an error opening the system output csv file")
            exit(1)

        # merge the reference and system output for SSD/DSD reports
        if task in ['manipulation']:
            m_df = pd.merge(myRef, mySys, how='left', on='ProbeFileID')
        elif task in ['splice']:
            m_df = pd.merge(myRef, mySys, how='left', on=['ProbeFileID', 'DonorFileID'])

         # if the confidence scores are 'nan', replace the values with the mininum score
        m_df[pd.isnull(m_df['ConfidenceScore'])] = mySys['ConfidenceScore'].min()
        # convert to the str type to the float type for computations
        m_df['ConfidenceScore'] = m_df['ConfidenceScore'].astype(np.float)

        total_num = m_df.shape[0]
        # if OptOut has chosen, all of queries should be applied
        if args_optOut:
            m_df = m_df.query(" IsOptOut==['N', 'Localization'] ")
            optout_num = m_df.shape[0]

        # the performers' result directory
        if '/' not in outRoot:
            root_path = '.'
            file_suffix = outRoot
        else:
            root_path, file_suffix = outRoot.rsplit('/', 1)

        if root_path != '.' and not os.path.exists(root_path):
            os.makedirs(root_path)

        # Partition Mode
        if args_query or args_queryPartition or args_queryManipulation:  # add or targetManiTypeSet or nontargetManiTypeSet
            print("Partition Mode \n")
            partition_mode = True

            if task in ['manipulation']:
                # merge the reference and index csv only
                subIndex = myIndex[['ProbeFileID', 'ProbeWidth', 'ProbeHeight']]
                pm_df = pd.merge(m_df, subIndex, how='inner', on='ProbeFileID')

                # if the files exist, merge the JTJoin and JTMask csv files with the
                # reference and index file
                if os.path.isfile(myJTJoinFname) and os.path.isfile(myJTMaskFname):
                    print("Merging the JournalJoin and JournalMask csv file with the reference files ...\n")
                    # merge the reference and index csv
                    df_1 = pd.merge(m_df, subIndex, how='inner', on='ProbeFileID')
                    # merge the JournalJoinTable and the JournalMaskTable
                    df_2 = pd.merge(myJTJoin, myJTMask, how='inner', on='JournalName')
                    # merge the dataframes above
                    pm_df = pd.merge(df_1, df_2, how='inner', on='ProbeFileID')
                    #pm_df.to_csv(outRoot + 'test.csv', index = False)
# for queryManipulation, drop duplicates conditioning by the chosen columns (e.g., ProbeFileID and Purpose)
#                    if args.queryManipulation:
#                        print("Removing duplicates of the chosen column for filtering target trials ...\n")
#                        chosenField = [x.strip() for x in args.queryManipulation.split('==')]
#                        #fm_df.sort(['ProbeFileID', chosenField[0]], inplace=True) #TODO: not necesary, but for testing
# pm_df = pm_df.drop_duplicates(['ProbeFileID', chosenField[0]]) #remove
# duplicates for the chosen column

            elif task in ['splice']:  # TBD
                subIndex = myIndex[['ProbeFileID', 'DonorFileID', 'ProbeWidth', 'ProbeHeight',
                                    'DonorWidth', 'DonorHeight']]  # subset the columns due to duplications
                pm_df = pd.merge(m_df, subIndex, how='inner', on=['ProbeFileID', 'DonorFileID'])

            if args_query:
                query_mode = 'q'
                query = args_query  # TODO: double-check
            elif args_queryPartition:
                query_mode = 'qp'
                query = args_queryPartition
            elif args_queryManipulation:  # TODO: testcases
                query_mode = 'qm'
                query = args_queryManipulation
                # query = ["("+targetFilter+ " and IsTarget == ['Y']) or IsTarget == ['N']"] #TODO: double-check
                #print("targetQuery {}".format(query))

            print("Query : {}\n".format(query))
            print("Creating partitions...\n")
            selection = f.Partition(pm_df, query, query_mode, fpr_stop=farStop,
                                    isCI=ci, ciLevel=ciLevel, total_num=total_num)
            DM_List = selection.part_dm_list
            print("Number of partitions generated = {}\n".format(len(DM_List)))
            print("Rendering csv tables...\n")
            table_df = selection.render_table()
            if isinstance(table_df, list):
                print("Number of table DataFrame generated = {}\n".format(len(table_df)))
            if args_query:
                for i, df in enumerate(table_df):
                    df.to_csv(outRoot + '_q_query_' + str(i) + '_report.csv', index=False)
            elif args_queryPartition:
                table_df[0].to_csv(outRoot + '_qp_query_report.csv')  # table_df is List type
            elif args_queryManipulation:
                for i, df in enumerate(table_df):
                    df.to_csv(outRoot + '_qm_query_' + str(i) + '_report.csv', index=False)

        # No partitions
        else:
            DM = dm.detMetrics(m_df['ConfidenceScore'], m_df['IsTarget'], fpr_stop=farStop,
                               isCI=ci, ciLevel=ciLevel, dLevel=dLevel, total_num=total_num)
            #print("*****d-prime {} dpoint{}".format(DM.d, DM.dpoint))

            DM_List = [DM]
            table_df = DM.render_table()
            table_df.to_csv(outRoot + '_all_report.csv', index=False)

        if isinstance(table_df, list):
            print("\nReport tables...\n")
            for i, table in enumerate(table_df):
                print("\nPartition {}:".format(i))
                print(table)
        else:
            print("Report table:\n{}".format(table_df))

        # Generating a default plot_options json config file
        p_json_path = "./plotJsonFiles"
        if not os.path.exists(p_json_path):
            os.makedirs(p_json_path)
        dict_plot_options_path_name = "./plotJsonFiles/plot_options.json"

        # Fixed: if plotType is indicated, then should be generated.
        if plotType == '' and os.path.isfile(dict_plot_options_path_name):
            # Loading of the plot_options json config file
            plot_opts = p.load_plot_options(dict_plot_options_path_name)
            plotType = plot_opts['plot_type']
        else:
            if plotType == '':
                plotType = 'roc'
            p.gen_default_plot_options(dict_plot_options_path_name,
                                       plot_title=plotTitle, plot_type=plotType.upper())
            plot_opts = p.load_plot_options(dict_plot_options_path_name)

        # opening of the plot_options json config file from command-line
        configPlot = False
        if configPlot:
            p.open_plot_options(dict_plot_options_path_name)

        # Dumping DetMetrics objects
        if dump:
            for i, DM in enumerate(DM_List):
                DM.write(root_path + '/' + file_suffix + '_query_' + str(i) + '.dm')

        # Creation of defaults plot curve options dictionnary (line style opts)
        Curve_opt = OrderedDict([('color', 'red'),
                                 ('linestyle', 'solid'),
                                 ('marker', '.'),
                                 ('markersize', 8),
                                 ('markerfacecolor', 'red'),
                                 ('label', None),
                                 ('antialiased', 'False')])

        # Creating the list of curves options dictionnaries (will be automatic)
        opts_list = list()
        colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black']
        linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
        # Give a random rainbow color to each curve
        # color = iter(cm.rainbow(np.linspace(0,1,len(DM_List)))) #YYL: error here
        color = cycle(colors)
        lty = cycle(linestyles)
        for i in range(len(DM_List)):
            new_curve_option = OrderedDict(Curve_opt)
            col = next(color)
            new_curve_option['color'] = col
            new_curve_option['markerfacecolor'] = col
            new_curve_option['linestyle'] = next(lty)
            opts_list.append(new_curve_option)

#        # Renaming the curves for the legend
#        if args_query or args_queryPartition or args_queryManipulation:
#            for curve_opts,query in zip(opts_list,selection.part_query_list):
#                curve_opts["label"] = query

        # Renaming the curves for the legend
        if args_query or args_queryPartition or args_queryManipulation:
            for curve_opts, query, dm in zip(opts_list, selection.part_query_list, DM_List):
                curve_opts["label"] = query + " (AUC: " + str(round(dm.auc, 2)) + ")"
                print("DM {}".format(curve_opts["label"]))

        # Creation of the object setRender (~DetMetricSet)
        configRender = p.setRender(DM_List, opts_list, plot_opts)
        # Creation of the Renderer
        myRender = p.Render(configRender)
        # Plotting
        myfigure = myRender.plot_curve(display, multi_fig=multiFigs)

        # save multiple figures if multi_fig == True
        if isinstance(myfigure, list):
            for i, fig in enumerate(myfigure):
                fig.savefig(outRoot + '_' + plotType + '_' + str(i) + '.pdf')
        else:
            myfigure.savefig(outRoot + '_' + plotType + '_all.pdf')
