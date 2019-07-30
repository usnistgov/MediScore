import os
import sys
import shlex
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from DetectionScorer import *

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

commands_folder_path = "/Users/tnk12/Documents/MediScoreV2/tools/DetectionScorer/DSWrapper/commands_local/"
os.chdir(commands_folder_path)
commands = [os.path.abspath(p) for p in os.listdir()]
command_arg_0 = "python2 /mnt/Backend1/Server/MediScore/tools/DetectionScorer/DetectionScorer.py "

with open(commands[0]) as f:
    x = f.read()

args = parser.parse_args(shlex.split(x[len(command_arg_0):]))

for key, value in vars(args).items():
    print("{}: {}".format(key, value))

def v_print(*args):
    for arg in args:
        print(arg)
    print()


# the performers' result directory
if '/' not in args.outRoot:
    root_path = '.'
    file_suffix = args.outRoot
else:
    root_path, file_suffix = args.outRoot.rsplit('/', 1)

if root_path != '.' and not os.path.exists(root_path):
    print("Trying to create a directory at {}".format(root_path))
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
    q_opts_list, q_plot_opts = plot_options(DM_List, args.configPlot, args.plotType, args.plotTitle, args.plotSubtitle, args.optOut)
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

