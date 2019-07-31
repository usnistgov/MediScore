"""
* Date: 5/25/2018
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

from collections import OrderedDict
from itertools import cycle

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib")
sys.path.append(lib_path)
import Render as p
import detMetrics as dm
import Partition as f

from medifor_datacontainer import MediForDataContainer
from metrics import Metrics

def v_print(*args):
    for arg in args:
        print(arg)
    print()

def load_csv(fname, mysep='|', mydtype=None):
    try:
        df = pd.read_csv(fname, sep=mysep, dtype=mydtype, low_memory=False)
        return df
    except IOError:
        print("ERROR: There was an error opening the file: {}".format(fname))
        exit(1)

def save_csv(df_list, outRoot, query_mode, report_tag):
    try:
        for i, df in enumerate(df_list):
            fname = outRoot + '_' + query_mode + '_query_' + str(i) + report_tag
            df.to_csv(fname, index=False, sep='|')
    except IOError:
        print("ERROR: There was an error saving the csv file: {}".format(fname))
        exit(1)

def is_optout(df, sys_response='tr'):
    v_print("Removing optout trials ...\n")
    if "IsOptOut" in df.columns:
        new_df = df.query(" IsOptOut==['N', 'Localization'] ")
    elif "ProbeStatus" in df.columns:
        new_df = index_m_df.query(
            " ProbeStatus==['Processed', 'NonProcessed', 'OptOutLocalization', 'FailedValidation', 'OptOutTemporal','OptOutSpatial']")
    return new_df

def overlap_cols(mySys, myRef):
    no_overlap = [c for c in mySys.columns if c not in myRef.columns]
    overlap = [c for c in mySys.columns if c in myRef.columns]
    return no_overlap, overlap

# Loading the specified file
def define_file_name(path, ref_fname, tag_name):
    my_fname = os.path.join(path, str(ref_fname.split('.')[:-1]).strip("['']") + tag_name)
    v_print("Specified JT file: {}".format(my_fname))
    return my_fname

def JT_merge(ref_dir, ref_fname, mainDF):
    join_fname = define_file_name(ref_dir, ref_fname, '-probejournaljoin.csv')
    mask_fname = define_file_name(ref_dir, ref_fname, '-journalmask.csv')
    if os.path.isfile(join_fname) and os.path.isfile(mask_fname):
        joinDF = pd.read_csv(join_fname, sep='|', low_memory=False)
        maskDF = pd.read_csv(mask_fname, sep='|', low_memory=False)
        jt_no_overlap, jt_overlap = overlap_cols(joinDF, maskDF)
        v_print("JT overlap columns: {}".format(jt_overlap))
        v_print("Merging (left join) the JournalJoin and JournalMask csv files with the reference file ...\n")
        jt_meta = pd.merge(joinDF, maskDF, how='left', on=jt_overlap)
        meta_no_overlap, meta_overlap = overlap_cols(mainDF, jt_meta)
        v_print("JT and main_df overlap columns: {}".format(meta_overlap))
        new_df = pd.merge(mainDF, jt_meta, how='left', on=meta_overlap)
        v_print("Main cols num: {}\n Meta cols num: {}\n, Merged cols num: {}".format(
            mainDF.shape, jt_meta.shape, new_df.shape))
        return new_df
    else:
        v_print("JT meta files do not exist, therefore, merging process will be skipped")
        return mainDF


def input_ref_idx_sys(refDir, inRef, inIndex, sysDir, inSys, outRoot, outSubMeta, sys_dtype):
    # Loading the reference file
    v_print("Ref file name: {}".format(os.path.join(refDir, inRef)))
    myRefDir = os.path.dirname(os.path.join(refDir, inRef))
    myRef = load_csv(os.path.join(refDir, inRef))
    # Loading the index file
    v_print("Index file name: {}".format(os.path.join(refDir, inIndex)))
    myIndex = load_csv(os.path.join(refDir, inIndex))
    # Loading system output
    v_print("Sys file name: {}".format(os.path.join(sysDir, inSys)))
    mySys = load_csv(os.path.join(sysDir, inSys), mydtype=sys_dtype)

    sys_ref_no_overlap, sys_ref_overlap = overlap_cols(mySys, myRef)
    v_print("sys_ref_no_overlap: {} \n, sys_ref_overlap: {}".format(
        sys_ref_no_overlap, sys_ref_overlap))
    index_ref_no_overlap, index_ref_overlap = overlap_cols(myIndex, myRef)
    v_print("index_ref_no_overlap: {}\n, index_ref_overlap: {}".format(
        index_ref_no_overlap, index_ref_overlap))

    # merge the reference and system output for SSD/DSD reports
    m_df = pd.merge(myRef, mySys, how='left', on=sys_ref_overlap)
    # if the confidence scores are 'nan', replace the values with the mininum score
    m_df[pd.isnull(m_df['ConfidenceScore'])] = mySys['ConfidenceScore'].min()
    # convert to the str type to the float type for computations
    m_df['ConfidenceScore'] = m_df['ConfidenceScore'].astype(np.float)

    # merge the reference and index csv (intersection only due to the partial index trials)
    index_m_df = pd.merge(m_df, myIndex, how='inner', on=index_ref_overlap)
    v_print("index_m_df_columns: {}".format(index_m_df.columns))

    # save subset of metadata for analysis purpose
    if outSubMeta:
        v_print("Saving the sub_meta csv file...")
        sub_pm_df = index_m_df[index_ref_overlap +
                               index_ref_no_overlap + ["IsTarget"] + sys_ref_no_overlap]
        v_print("sub_pm_df columns: {}".format(sub_pm_df.columns))
        sub_pm_df.to_csv(outRoot + '_subset_meta.csv', index=False, sep='|')

    return index_m_df, sys_ref_overlap


def yes_query_mode(df, task, refDir, inRef, outRoot, optOut, outMeta, farStop, ci, ciLevel, dLevel, total_num, sys_response, query_str, query_mode, sys_ref_overlap):

    m_df = df.copy()
    # if the files exist, merge the JTJoin and JTMask csv files with the reference and index file
    if task in ['manipulation', 'splice', 'camera', 'eventverification']:
        v_print("Merging the JournalJoin and JournalMask for the {} task\n".format(task))
        m_df = JT_merge(refDir, inRef, df)

    v_print("Creating partitions for queries ...\n")
    selection = f.Partition(m_df, query_str, query_mode, fpr_stop=farStop, isCI=ci,
                            ciLevel=ciLevel, total_num=total_num, sys_res=sys_response, overlap_cols=sys_ref_overlap)
    DM_List = selection.part_dm_list
    v_print("Number of partitions generated = {}\n".format(len(DM_List)))

    # Output the meta data as dataframe for queries
    DF_List = selection.part_df_list
    v_print("Number of CSV partitions generated = {}\n".format(len(DF_List)))

    if outMeta:  # save all metadata for analysis purpose
        v_print("Saving all the meta info csv file ...")
        save_csv(DF_List, outRoot, query_mode, '_allmeta.csv')

    table_df = selection.render_table()

    return DM_List, table_df, selection


def no_query_mode(df, task, refDir, inRef, outRoot, optOut, outMeta, farStop, ci, ciLevel, dLevel, total_num, sys_response):

    m_df = df.copy()

    if outMeta:  # save all metadata for analysis purpose
        v_print("Saving all the meta info csv file ...")
        v_print("Merging the JournalJoin and JournalMask for the {} task\n, But do not score with this data".format(task))
        meta_df = JT_merge(refDir, inRef, m_df)
        meta_df.to_csv(outRoot + '_allmeta.csv', index=False, sep='|')
        m_df.to_csv(outRoot + '_meta_scored.csv', index=False, sep='|')

    # DM = dm.detMetrics(m_df['ConfidenceScore'], m_df['IsTarget'], fpr_stop=farStop,
    #                    isCI=ci, ciLevel=ciLevel, dLevel=dLevel, total_num=total_num, sys_res=sys_response)
    target_label, non_target_label = "Y", "N"
    fpr, tpr, fnr, thres = Metrics.compute_rates(m_df['ConfidenceScore'], m_df['IsTarget'], target_label=target_label, non_target_label=non_target_label)
    DM = MediForDataContainer(fpr, fnr, thres, label=None, line_options=None)
    DM.setter_full(m_df['IsTarget'], m_df['ConfidenceScore'], total_num, farStop, ciLevel, dLevel, sys_response,                    
                   target_label=target_label, non_target_label=non_target_label, verbose=False)
    DM_List = [DM]
    # table_df = DM.render_table()
    table_df = DM.metrics_to_dataframe(orientation="vertical")

    return DM_List, table_df


def plot_options(DM_list, configPlot, plotType, plotTitle, plotSubtitle, optOut):
    # Generating a default plot_options json config file
    # p_json_path = "./plotJsonFiles"
    # if not os.path.exists(p_json_path):
    #     os.makedirs(p_json_path)
    

    # # opening of the plot_options json config file from command-line
    # if configPlot:
    #     p.open_plot_options(dict_plot_options_path_name)

    # # if plotType is indicated, then should be generated.
    # if plotType == '' and os.path.isfile(dict_plot_options_path_name):
    #     # Loading of the plot_options json config file
    #     plot_opts = p.load_plot_options(dict_plot_options_path_name)
    #     plotType = plot_opts['plot_type']
    #     plot_opts['title'] = plotTitle
    #     plot_opts['subtitle'] = plotSubtitle
    #     plot_opts['subtitle_fontsize'] = 11
    #     #print("test plot title1 {}".format(plot_opts['title']))
    # else:
    #     if plotType == '':
    #         plotType = 'roc'
    #     p.gen_default_plot_options(plot_title=plotTitle, plot_subtitle=plotSubtitle, plot_type=plotType.upper())
    #     plot_opts = p.load_plot_options(dict_plot_options_path_name)
    #     #print("test plot title2 {}".format(plot_opts['title']))
    plot_opts = OrderedDict([
            ('title', "Performance" if plotTitle is None else plotTitle),
            ('plot_type', plotType.upper()),
            ('subtitle', ''),
            ('figsize', (8, 6)),
            ('title_fontsize', 13), 
            ('subtitle_fontsize', 11), 
            ('xlim', [0,1]),
            ('ylim', [0,1]),
            ('xticks_size', 'medium'),
            ('yticks_size', 'medium'),
            ('xlabel', "False Alarm Rate [%]"),
            ('xlabel_fontsize', 11),
            ('ylabel_fontsize', 11)])

    if plotType.lower() == "det":
        plot_opts["xscale"] = "log"
        plot_opts["ylabel"] = "Miss Detection Rate [%]"
        # plot_opts["xticks"] = norm.ppf([.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, .01, .02, .05, .10, .20, .40, .60, .80, .90, .95, .98, .99, .995, .999])
        plot_opts["xticks"] = [0.01, 0.1, 1, 10]
        plot_opts["yticks"] = norm.ppf([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999])
        plot_opts["xlim"] = (plot_opts["xticks"][0], plot_opts["xticks"][-1])
        plot_opts["ylim"] = (plot_opts["yticks"][0], plot_opts["yticks"][-1])
        # plot_opts["xticks_labels"] = ['0.01', '0.02', '0.05', '0.1', '0.2', '0.5', '1', '2', '5', '10', '20', '40', '60', '80', '90', '95', '98', '99', '99.5', '99.9']
        plot_opts["xticks_labels"] = ["0.01", "0.1", "1", "10"]
        plot_opts["yticks_labels"] = ['5.0', '10.0', '20.0', '40.0', '60.0', '80.0', '90.0', '95.0', '98.0', '99.0', '99.5', '99.9']

    elif plotType.lower() == "roc":
        plot_opts["xscale"] = "linear"
        plot_opts["ylabel"] = "Correct Detection Rate [%]"
        plot_opts["xticks"] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        plot_opts["yticks"] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        plot_opts["yticks_labels"] = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']
        plot_opts["xticks_labels"] = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']

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
    # color = iter(cm.rainbow(np.linspace(0,1,len(DM_list)))) #YYL: error here
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

    if optOut:
        plot_opts['title'] = "tr" + plot_opts['title']

    return opts_list, plot_opts


def query_plot_options(DM_List, opts_list, plot_opts, selection, optOut, noNum):
    # Renaming the curves for the legend
    for curve_opts, query, dm_list in zip(opts_list, selection.part_query_list, DM_List):
        trr_str = ""
        #print("plottype {}".format(plot_opts['plot_type']))
        if plot_opts['plot_type'] == 'ROC':
            met_str = " (AUC: " + str(round(dm_list.auc, 2))
        elif plot_opts['plot_type'] == 'DET':
            met_str = " (EER: " + str(round(dm_list.eer, 2))

        if optOut:
            trr_str = ", TRR: " + str(dm_list.trr)
            if plot_opts['plot_type'] == 'ROC':
                met_str = " (trAUC: " + str(round(dm_list.auc, 2))
            elif plot_opts['plot_type'] == 'DET':
                met_str = " (trEER: " + str(round(dm_list.eer, 2))

        if noNum:
            curve_opts["label"] = query + met_str + trr_str + ")"
        else:
            curve_opts["label"] = query + met_str + trr_str + ", T#: " + \
                str(dm_list.t_num) + ", NT#: " + str(dm_list.nt_num) + ")"

    return opts_list, plot_opts

########### Command line interface ########################################################
def command_interface():
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
                        choices=['manipulation', 'splice', 'eventverification', 'camera'],
                        help='Define the target task for evaluation(default: %(default)s)', metavar='character')
    # Input Options
    parser.add_argument('--refDir', default='.',
                        help='Specify the reference and index data path: [e.g., ../NC2016_Test] (default: %(default)s)', metavar='character')
    parser.add_argument('-tv', '--tsv', default='',
                        help='Specify the reference TSV file that contains the ground-truth and metadata info [e.g., results.tsv]', metavar='character')
    parser.add_argument('-r', '--inRef', default='', type=is_file_specified,
                        help='Specify the reference CSV file that contains the ground-truth and metadata info [e.g., references/ref.csv]', metavar='character')
    parser.add_argument('-x', '--inIndex', default='', type=is_file_specified,
                        help='Specify the index CSV file: [e.g., indexes/index.csv] (default: %(default)s)', metavar='character')
    parser.add_argument('--sysDir', default='.',
                        help='Specify the system output data path: [e.g., /mySysOutputs] (default: %(default)s)', metavar='character')  # Optional
    parser.add_argument('-s', '--inSys', default='', type=is_file_specified,
                        help='Specify a CSV file of the system output formatted according to the specification: [e.g., expid/system_output.csv] (default: %(default)s)', metavar='character')
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
                        help='Specify the report path and the file name prefix for saving the plot(s) and table (s). For example, if you specify "--outRoot test/NIST_001", you will find the plot "NIST_001_det.png" and the table "NIST_001_report.csv" in the "test" folder: [e.g., temp/xx_sys] (default: %(default)s)', metavar='character')
    parser.add_argument('--outMeta', action='store_true',
                        help="Save a CSV file with the system output with metadata")
    parser.add_argument('--outSubMeta', action='store_true',
                        help="Save a CSV file with the system output with minimal metadata")
    parser.add_argument('--dump', action='store_true',
                        help="Save the dump files (formatted as a binary) that contains a list of FAR, FPR, TPR, threshold, AUC, and EER values. The purpose of the dump files is to load the point values for further analysis without calculating the values again.")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Print output with procedure messages on the command-line if this option is specified.")
    # Plot Options
    parser.add_argument('--plotTitle', default='Performance',
                        help="Define a plot title (default: %(default)s)", metavar='character')
    parser.add_argument('--plotSubtitle', default='',
                        help="Define a plot subtitle (default: %(default)s)", metavar='character')
    parser.add_argument('--plotType', default='roc', choices=['roc', 'det'],
                        help="Define a plot type:[roc] and [det] (default: %(default)s)", metavar='character')
    parser.add_argument('--display', action='store_true',
                        help="Display a window with the plot(s) on the command-line if this option is specified.")
    parser.add_argument('--multiFigs', action='store_true',
                        help="Generate plots(with only one curve) per a partition ")
    parser.add_argument('--noNum', action='store_true',
                        help="Do not print the number of target trials and non-target trials on the legend of the plot")
    # Custom Plot Options
    parser.add_argument('--configPlot', default='',
                        help="Load a JSON file that allows user to customize the plot (e.g. change the title font size) by augmenting the json files located in the 'plotJsonFiles' folder.")
    # Performance Evaluation by Query Options
    factor_group = parser.add_mutually_exclusive_group()

    factor_group.add_argument('-q', '--query', nargs='*',
                              help="Evaluate system performance on a partitioned dataset (or subset) using multiple queries. Depending on the number (N) of queries, the option generates N report tables (CSV) and one plot (PDF) that contains N curves.", metavar='character')
    factor_group.add_argument('-qp', '--queryPartition',
                              help="Evaluate system performance on a partitioned dataset (or subset) using one query. Depending on the number (M) of partitions provided by the cartesian product on query conditions, this option generates a single report table (CSV) that contains M partition results and one plot that contains M curves. (syntax retriction: '==[]','<','<=')", metavar='character')
    factor_group.add_argument('-qm', '--queryManipulation', nargs='*',
                              help="This option is similar to the '-q' option; however, the queries are only applied to the target trials (IsTarget == 'Y') and use all of non-target trials. Depending on the number (N) of queries, the option generates N report tables (CSV) and one plot (PDF) that contains N curves.", metavar='character')
    parser.add_argument('--optOut', action='store_true',
                        help="Evaluate system performance on trials where the IsOptOut value is 'N' only or the ProbeStatus values are ['Processed', 'NonProcessed', 'OptOutLocalization', 'FailedValidation','OptOutTemporal','OptOutSpatial']")

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

    return args


if __name__ == '__main__':

    if len(sys.argv) == 1:
        class ArgsList():
            def __init__(self):
                print("Debugging mode: initiating ...")
                # Inputs
                self.task = "manipulation"
                self.refDir = "../../data/test_suite/detectionScorerTests/reference"
                self.inRef = "NC2017-manipulation-ref.csv"
                self.inIndex = "NC2017-manipulation-index.csv"
                self.sysDir = "../../data/test_suite/detectionScorerTests/baseline"
                self.inSys = "Base_NC2017_Manipulation_ImgOnly_p-copymove_01.csv"
                # TSV
                #self.tsv = "tsv_example/q-query-example.tsv"
                self.tsv = ""
                # Metrics
                self.farStop = 0.05
                self.ci = False
                self.ciLevel = 0.9
                self.dLevel = 0.0
                # Outputs
                self.outRoot = "./testcase/test"
                self.outMeta = False
                self.outSubMeta = False
                self.dump = False
                self.verbose = False
                #  Plot options
                self.plotTitle = "Performance"
                self.plotSubtitle = "bla"
                self.plotType = "roc"
                self.display = True
                self.multiFigs = False
                self.configPlot = ""
                self.noNum = False
                # Query options
                self.query = ""
                self.queryPartition = ""
                self.queryManipulation = ["TaskID==['manipulation']"]
                #self.queryManipulation = ""
                self.optOut = False
                self.verbose = True

        args = ArgsList()
        # Verbosity option
        if args.verbose:
            def _v_print(*args):
                for arg in args:
                    print (arg),
                print
        else:
            _v_print = lambda *a: None      # do-nothing function

        v_print = _v_print

    else:
        args = command_interface()

    # the performers' result directory
    if '/' not in args.outRoot:
        root_path = '.'
        file_suffix = args.outRoot
    else:
        root_path, file_suffix = args.outRoot.rsplit('/', 1)

    if root_path != '.' and not os.path.exists(root_path):
        os.makedirs(root_path)

    if (not args.query) and (not args.queryPartition) and (not args.queryManipulation) and (args.multiFigs is True):
        print("ERROR: The multiFigs option is not available without query options.")
        exit(1)

    # this should be "string" due to the "nan" value, otherwise "nan"s will have different unique numbers
    sys_dtype = {'ConfidenceScore': str}

    if args.tsv:
        index_m_df = load_csv(os.path.join(args.refDir, args.inRef),
                              mysep='\t', mydtype=sys_dtype)
    else:
        index_m_df, sys_ref_overlap = input_ref_idx_sys(args.refDir, args.inRef, args.inIndex, args.sysDir,
                                                        args.inSys, args.outRoot, args.outSubMeta, sys_dtype)

    total_num = index_m_df.shape[0]
    v_print("Total data number: {}".format(total_num))

    sys_response = 'all'  # to distinguish use of the optout
    query_mode = ""
    tag_state = '_all'

    if args.optOut:
        sys_response = 'tr'
        index_m_df = is_optout(index_m_df, sys_response)

    # TSV input mode
    if args.tsv:
        print("Place TSV metrics here ...")
        DM_List, table_df = None, None

    # Query Mode
    elif args.query or args.queryPartition or args.queryManipulation:
        if args.query:
            query_mode = 'q'
            query_str = args.query
        elif args.queryPartition:
            query_mode = 'qp'
            query_str = args.queryPartition
        elif args.queryManipulation:
            query_mode = 'qm'
            query_str = args.queryManipulation

        tag_state = '_' + query_mode + '_query'

        v_print("Query_mode: {}, Query_str: {}".format(query_mode,query_str))
        DM_List, table_df, selection = yes_query_mode(index_m_df, args.task, args.refDir, args.inRef, args.outRoot,
                                                      args.optOut, args.outMeta, args.farStop, args.ci, args.ciLevel, args.dLevel, total_num, sys_response, query_str, query_mode, sys_ref_overlap)
        # Render plots with the options
        q_opts_list, q_plot_opts = plot_options(DM_List, args.configPlot, args.plotType,
                                                args.plotTitle, args.plotSubtitle, args.optOut)
        opts_list, plot_opts = query_plot_options(DM_List, q_opts_list, q_plot_opts, selection, args.optOut, args.noNum)

    # No Query mode
    else:
        #print(index_m_df.columns)
        DM_List, table_df = no_query_mode(index_m_df, args.task, args.refDir, args.inRef, args.outRoot,
                                          args.optOut, args.outMeta, args.farStop, args.ci, args.ciLevel, args.dLevel, total_num, sys_response)
        # Render plots with the options
        opts_list, plot_opts = plot_options(DM_List, args.configPlot, args.plotType,
                                            args.plotTitle, args.plotSubtitle, args.optOut)

    v_print("Rendering/saving csv tables...\n")
    if isinstance(table_df, list):
        print("\nReport tables:\n")
        for i, table in enumerate(table_df):
            print("\nPartition {}:".format(i))
            print(table)
            table.to_csv(args.outRoot + tag_state + '_' + str(i) + '_report.csv', index=False, sep='|')
    else:
        print("Report table:\n{}".format(table_df))
        table_df.to_csv(args.outRoot + tag_state + '_report.csv', index=False, sep='|')

    if args.dump:
        v_print("Dumping metric objects ...\n")
        for i, DM in enumerate(DM_List):
            DM.write(root_path + '/' + file_suffix + '_query_' + str(i) + '.dm')

    v_print("Rendering/saving plots...\n")
    # Creation of the object setRender (~DetMetricSet)
    configRender = p.setRender(DM_List, opts_list, plot_opts)
    # Creation of the Renderer
    myRender = p.Render(configRender)
    # Plotting
    myfigure = myRender.plot_curve(
        args.display, multi_fig=args.multiFigs, isOptOut=args.optOut, isNoNumber=args.noNum, isCI=args.ci)

    # save multiple figures if multi_fig == True
    if isinstance(myfigure, list):
        for i, fig in enumerate(myfigure):
            fig.savefig(args.outRoot + tag_state + '_' + str(i)  + '_' + plot_opts['plot_type'] + '.pdf', bbox_inches='tight')
    else:
        myfigure.savefig(args.outRoot + tag_state + '_' + plot_opts['plot_type'] +'.pdf', bbox_inches='tight')
