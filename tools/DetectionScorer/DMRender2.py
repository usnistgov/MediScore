# -*- coding: utf-8 -*-
"""
Date: 03/07/2017
Authors: Yooyoung Lee and Timothee Kheyrkhah

Description: this script loads DM files and renders plots.
In addition, the user can customize the plots through the command line interface or via 
json files.

"""

import os 
import sys
import json
import logging
import argparse
from ast import literal_eval

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib")
sys.path.append(lib_path)

from render import Render, Annotation
from detMetrics import detMetrics
from data_container import DataContainer
from medifor_data_container import MediForDataContainer

def create_parser():
    """Command line interface creation with arguments definitions.

    Returns:
        argparse.ArgumentParser

    """
    parser = argparse.ArgumentParser(description='NIST detection scorer.', formatter_class=argparse.RawTextHelpFormatter)

    input_help = ("Supports the following inputs:\n- .txt file containing one file path per line\n- .mdc file\n",
                  "- a list of pair [{'path':'path/to/mdc_file','label':str,'show_label':bool}, **{any matplotlib.lines.Line2D properties}].\n",
                  "Example:\n  [[{'path':'path/to/file_1.mdc','label':'sys_1','show_label':True}, {'color':'red','linestyle':'solid'}],\n",
                  "             [{'path':'path/to/file_2.mdc','label':'sys_2','show_label':False}, {}]",
                  "Note: Use an empty dict for default behavior.")

    parser.add_argument('-i', '--input', required=True,metavar = "str",
                        help=''.join(input_help))

    parser.add_argument("--outputFolder", default='.',
                        help="Path to the output folder. (default: %(default)s)",metavar='')

    parser.add_argument("--outputFileNameSuffix", default='plot',
                        help="Output file name suffix. (default: '%(default)s')",metavar='')

    # Plot Options
    parser.add_argument("--plotOptionJsonFile", help="Path to a json file containing plot options", metavar='path')

    parser.add_argument("--lineOptionJsonFile", help="Path to a json file containing a list of matplotlib.lines.Line2D dictionnaries properties (One per line)", metavar='path')

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

    parser.add_argument('--dumpPlotParams', action="store_true",
                        help="Dump the parameters used for the plot and the lines as Jsons in the output directory")

    parser.add_argument("--logtype", type=int, default=0, const=0, nargs='?',
                        choices=[0, 1, 2, 3],
                        help="Set the logging type")

    parser.add_argument("--console_log_level", dest="consoleLogLevel", default="INFO", const="INFO", nargs='?',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Set the console logging level")

    parser.add_argument("--file_log_level", dest="fileLogLevel", default="DEBUG", const="DEBUG", nargs='?',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        help="Set the file logging level")

    return parser

def create_logger(logger_type=1, filename="./DMRender.log", console_loglevel="INFO", file_loglevel="DEBUG"):
    """Create a logger with the provided log level

    Args:
        logger_type (int): type of logging (0: no logging, 1: console only, 2: file only, 3: both)
        filename (str): filename or path of the log file
        console_loglevel (str): loglevel string for the console -> 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        file_loglevel (str): loglevel string for the file -> :'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

    """
    if logger_type == 0:
        logger = logging.getLogger('DMlog')
        NullHandler = logging.NullHandler()
        logger.addHandler(NullHandler)

    else:
        try:
            numeric_file_loglevel= getattr(logging, file_loglevel.upper())
            numeric_console_loglevel = getattr(logging, console_loglevel.upper())
        except AttributeError as e:
            print("LoggingError: Invalid logLevel -> {}".format(e))
            sys.exit(1)

        logger = logging.getLogger('DMlog')
        logger.setLevel(logging.DEBUG)

        # create console handler which logs to stdout
        if logger_type in [1,3]:
            consoleLogger = logging.StreamHandler(stream=sys.stdout)
            consoleLogger.setLevel(numeric_console_loglevel)
            if sys.version_info[0] >= 3:
                consoleFormatter = logging.Formatter("{name:<5} - {levelname} - {message}", style='{')
            else:
                consoleFormatter = logging.Formatter("%(name)-5s - %(levelname)s - %(message)s")
            consoleLogger.setFormatter(consoleFormatter)
            logger.addHandler(consoleLogger)

        # create file handler which logs to a file
        if logger_type in [2,3]:
            fileLogger = logging.FileHandler(filename,mode='w')
            fileLogger.setLevel(numeric_file_loglevel)
            if sys.version_info[0] >= 3:
                fileFormatter = logging.Formatter("{asctime}|{name:<5}|{levelname:^9} - {message}", datefmt='%H:%M:%S', style='{')
            else:
                fileFormatter = logging.Formatter("%(asctime)s|%(name)-5s|%(levelname)-9s - %(message)s", datefmt='%H:%M:%S')
            fileLogger.setFormatter(fileFormatter)
            logger.addHandler(fileLogger)

        # Silence the matplotlib logger
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)

    return logger

def close_file_logger(logger):
    for handler in logger.handlers:
        if handler.__class__.__name__ == "FileHandler":
            handler.close()

def DMRenderExit(logger):
    close_file_logger(logger)
    sys.exit(1)

def validate_plot_options(plot_options):
    """Validation of the custom dictionnary of general options for matplotlib's plot.
    This function raises a custom exception in case of invalid or missing plot options
    and catches in order to quit with a specific error message.

    Args:
        plot_options (dict): The dictionnary containing the general plot options
    # TODO: Update the following fields (add the new ones)
    Note: The dictionnary should contain at most the following keys
            'title', 'subtitle', 'plot_type', 
            'title_fontsize', 'subtitle_fontsize', 
            'xticks_size', 'yticks_size', 
            'xlabel', 'xlabel_fontsize', 
            'ylabel', 'ylabel_fontsize'
        See the matplotlib documentation for a description of those parameters 
        (except for the plot_type (choose from 'ROC', 'DET'))
    """

    class PlotOptionValidationError(Exception):
        """Custom Exception raised for errors in the global plot option json file
        Attributes:
            msg (str): explanation message of the error
        """
        def __init__(self,msg):
            self.msg = msg

    logger = logging.getLogger("DMlog")
    logger.info("Validating global plot options...")
    try:
        #1 check plot type
        plot_type = plot_options["plot_type"]
        if plot_type not in ["ROC", "DET"]:
            raise PlotOptionValidationError("invalid plot type '{}' (choose from 'ROC', 'DET')".format(plot_type))

    except PlotOptionValidationError as e:
        logging.error("PlotOptionValidationError: {}".format(e.msg))
        DMRenderExit(logger)

    except KeyError as e:
        logging.error("PlotOptionValidationError: no '{}' provided".format(e.args[0]))
        DMRenderExit(logger)

def evaluate_input(args):
    """This function parse and evaluate the argument from command line interface,
    it returns the list of DM files loaded with also potential custom plot and lines options provided.
    The functions parse the input argument and the potential custom options arguments (plot and lines).

    It first infers the type of input provided. The following 3 input type are supported:
        - type 1: A .txt file containing a pass of .mdc file per lines
        - type 2: A single .mdc path
        - type 3: A custom list of pairs of dictionnaries (see the input help from the command line parser)

    Then it loads custom (or defaults if not provided) plot and lines options per DM file.

    Args:
        args (argparse.Namespace): the result of the call of parse_args() on the ArgumentParser object

    Returns:
        Result (tuple): A tuple containing
            - MDC_list (list): list of DM objects
            - opts_list (list): list of dictionnaries for the lines options
            - plot_opts (dict): dictionnary of plot options  
    """

    def call_loader(path, logger):
        try:
            if os.path.isfile(path): # Use this instead of catching  FileNotFoundError for Python2 support
                return DataContainer.load(path)
            else:
                logger.error("FileNotFoundError: No such file or directory: '{}'".format(path))
                DMRenderExit(logger)
        except IOError as e:
            logger.error("IOError: {}".format(str(e)))
            DMRenderExit(logger)

        except UnicodeDecodeError as e:
            logger.error("UnicodeDecodeError: {}\n".format(str(e)))
            DMRenderExit(logger)

    logger = logging.getLogger('DMlog')
    MDC_list = list()
    # Case 1: text file containing one path per line
    if args.input.endswith('.txt'):
        logger.debug("Input of type 1 detected")
        input_type = 1
        if os.path.isfile(args.input):
            with open(args.input) as f:
                fp_list = f.read().splitlines()
        else:
            logger.error("FileNotFoundError: No such file or directory: '{}'".format(args.input))
            DMRenderExit(logger)

        for mdc_file_path in fp_list:
            label = mdc_file_path
            # We handle a potential label provided
            if ':' in mdc_file_path:
                mdc_file_path, label = mdc_file_path.rsplit(':', 1)

            mdc_obj = call_loader(mdc_file_path, logger)

            mdc_obj.path = mdc_file_path
            mdc_obj.label = label
            mdc_obj.show_label = True
            MDC_list.append(mdc_obj)

    # Case 2: One mdc pickled file
    elif args.input.endswith('.mdc'):
        logger.debug("Input of type 2 detected")
        input_type = 2
        mdc_obj = call_loader(args.input, logger)
        mdc_obj.path = args.input
        mdc_obj.label = args.input
        mdc_obj.show_label = None
        MDC_list = [mdc_obj]

    # Case 3: String containing a list of input with their metadata
    elif args.input.startswith('[[') and args.input.endswith(']]'):
        logger.debug("Input of type 3 detected")
        input_type = 3
        try:
            input_list = literal_eval(args.input)
            for mdc_data, mdc_opts in input_list:
                mdc_file_path = mdc_data['path']
                mdc_obj = call_loader(mdc_file_path, logger)
                mdc_obj.path = mdc_file_path
                mdc_obj.label = mdc_data['label']
                mdc_obj.show_label = mdc_data['show_label']
                mdc_obj.line_options = mdc_opts
                MDC_list.append(mdc_obj)

        except ValueError as e:
            if not all([len(x) == 2 for x in input_list]):
                logger.error("ValueError: Invalid input format. All sub-lists must be a pair of two dictionnaries.\n-> {}".format(str(e)))
            else:
                logger.error("ValueError: {}".format(str(e)))
            DMRenderExit(logger)

        except SyntaxError as e:
            logger.error("SyntaxError: The input provided is invalid.\n-> {}".format(str(e)))
            DMRenderExit(logger)

    else:
        logger.error("The input type does not match any of the following inputs:\n- .txt file containing one file path per line\n- .mdc file\n- a list of pair [{'path':'path/to/mdc_file','label':str,'show_label':bool}, **{any matplotlib.lines.Line2D properties}].\n")
        DMRenderExit(logger)

    #*-* Options Processing *-*

    # General plot options
    if not args.plotOptionJsonFile:
        logger.info("Generating the default plot options...")
        plot_opts = render.gen_default_plot_options(plot_title = args.plotTitle, plot_subtitle = args.plotSubtitle, plot_type = args.plotType)
        
    else:
        logger.info("Loading of the plot options from the json config file...")
        if os.path.isfile(args.plotOptionJsonFile):
            with open(args.plotOptionJsonFile, 'r') as f:
                plot_opts = json.load(f)
            validate_plot_options(plot_opts)
        else:
            logger.error("FileNotFoundError: No such file or directory: '{}'".format(args.plotOptionJsonFile))
            DMRenderExit(logger)
    
    # line options
    if args.lineOptionJsonFile and input_type != 3:
        logger.info("Loading of the lines options from the json config file and overriding data container line settings...")
        if os.path.isfile(args.lineOptionJsonFile):

            with open(args.lineOptionJsonFile, 'r') as f:
                opts_list = json.load(f)

            if len(opts_list) != len(MDC_list):
                print("ERROR: the number of the line options is different with the number of the DM objects: ({} < {})".format(len(opts_list), len(MDC_list)))
                DMRenderExit(logger)
            else:
                for mdc, line_options in zip(MDC_list, opts_list):
                    mdc.line_options = line_options
        else:
            logger.error("FileNotFoundError: No such file or directory: '{}'".format(args.lineOptionJsonFile))
            DMRenderExit(logger)

    return MDC_list, plot_opts

def outputFigure(figure, outputFolder, outputFileNameSuffix, plotType):
    """Generate the plot file(s) as pdf at the provided destination
    The filename created as the following format:
        * for a single figure: {file_suffix}_{plot_type}_all.pdf
        * for a list of figures: {file_suffix}_{plot_type}_{figure_number}.pdf

    Args:
        figure (matplotlib.pyplot.figure or a list of matplotlib.pyplot.figure): the figure to plot
        outputFolder (str): path to the destination folder 
        outputFileNameSuffix (str): string suffix that will be inserted at the beginning of the filename
        plotType (str): the type of plot (ROC or DET)

    """
    logger = logging.getLogger("DMlog")
    logger.info("Figure output generation...")
    if outputFolder != '.' and not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    # Figure Output
    fig_filename_tmplt = "{file_suffix}_{plot_type}_{plot_id}.pdf".format(file_suffix=outputFileNameSuffix,
                                                                          plot_type=plotType,
                                                                          plot_id="{plot_id}")
    
    fig_path = os.path.normpath(os.path.join(outputFolder, fig_filename_tmplt))

    # This will save multiple figures if multi_fig == True
    if isinstance(figure,list):
        for i,fig in enumerate(figure):
            fig.savefig(fig_path.format(plot_id=str(i)), bbox_inches='tight')
    else:
        figure.savefig(fig_path.format(plot_id='all'), bbox_inches='tight')

    logger.info("Figure output generation... Done.")

def dumpPlotOptions(outputFolder, mdc_list, plot_opts):
    """This function dumps the options used for the plot and lines as json files.
    at the provided outputFolder. The two file have following names:
        - Global options plot: "plot_options.json"
        - lines options:  "line_options.json"

    Args: 
        outputFolder (str): path to the output folder
        opts_list (list): list of dictionnaries for the lines options
        plot_opts (dict): dictionnary of plot options  

    """

    output_json_path = os.path.normpath(os.path.join(outputFolder, "plotJsonFiles"))
    if not os.path.exists(output_json_path):
        os.makedirs(output_json_path)

    all_line_options = [mdc.line_options for mdc in mdc_list]
    for json_data, json_filename in zip([all_line_options, plot_opts], ["line_options.json", "plot_options.json"]):
        with open(os.path.join(output_json_path, json_filename), 'w') as f:
            f.write(json.dumps(json_data, indent=2, separators=(',', ':')))


if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()
    logger = create_logger(logger_type=args.logtype, 
                           filename="./DMRender.log", 
                           console_loglevel=args.consoleLogLevel, 
                           file_loglevel=args.fileLogLevel)

    logger.info("Starting DMRender...")

    isCI = False

    # Backend option
    if not args.display: # If no plot displayed, we set the matplotlib backend
        import matplotlib
        matplotlib.use('Agg')

    logger.debug("Evaluating parameters...")
    MDC_list, opts_list, plot_opts = evaluate_input(args)
    logger.debug("Processing {} files".format(len(MDC_list)))
    
    isOptOut = False

    # Updating the mdc label with metadata
    for mdc in MDC_List:
        mdc_label = mdc.label if mdc.label is not None else mdc.path

        metric_list, metric_tmplt = [], "{tr}{metric}: {val}"
        if plot_opts['plot_type'] == 'ROC':
            metric_list.append(metric_tmplt.format(tr='{tr}', metric="AUC", val=round(mdc.auc,2)))
        elif plot_opts['plot_type'] == 'DET':
            metric_list.append(metric_tmplt.format(tr='{tr}', metric="EER", val=round(mdc.eer,2)))

        tr_label, isOptOut = '', False
        if mdc.sys_res == 'tr':
            tr_label, isOptOut = "tr", True
            metric_list.append("TRR: {}".format(mdc.trr))
        metric_list[0].format(tr=tr_label)
        
        if not args.noNum:
            metric_list.extend(["T#: {}".format(mdc.t_num)], ["NT#: {}".format(mdc.nt_num)])

        mdc.line_options["label"] = "{} ({})".format(mdc_label, ', '.join(metric_list))

    if len(MDC_list) == 1:
        annotation_list = []
        mdc = MDC_list[0]
        # DET Annotation
        if args.plotType == "DET"
            det_annotation_parameters = {"xy":(norm.ppf(mdc.eer), norm.ppf(mdc.eer)), "xycoords":'data',
                                         "xytext":(norm.ppf(mdc.eer + 0.05) + 0.5, norm.ppf(mdc.eer + 0.05) + 0.5), "textcoords":'data',
                                         "arrowprops":{"arrowstyle":"-|>", "connectionstyle":"arc3, rad=+0.2", "fc":"w"},
                                         "size":10, "va":'center', "ha":'center', "bbox":{"boxstyle":"round4", "fc":"w"}}
            if args.noNum:
                if isOptOut:
                    annotation_text = "trEER = {:.2f} (TRR: {:.2f})".format(mdc.eer * 100, mdc.trr)
                else:
                    annotation_text = "EER = {.2f}%".format(mdc.eer * 100)
            else:
                if isOptOut:
                    annotation_text = "trEER = {:.2f} \n(TRR: {:.2f}, T#: {}, NT#: {})".format(mdc.eer * 100, mdc.trr, mdc.t_num, mdc.nt_num)
                else:
                    annotation_text = "EER = {:.2f} \n(T#: {}, NT#: {})".format(mdc.eer * 100, mdc.t_num, mdc.nt_num)
            
            annotation_list.append(Annotation(annotation_text, det_annotation_parameters))

        # ROC Annotation
        elif args.plotType == "ROC"
            roc_annotation_parameters = {"xy":(0.7, 0.2), "xycoords":'data', "xytext":(0.7, 0.2), "textcoords":'data',
                                         "size":10, "va":'center', "ha":'center', "bbox":{"boxstyle":"round4", "fc":"w"}}
            if isCI:
                if args.noNum:
                    if isOptOut:
                        annotation_text = "trAUC={:.2f}\n(TRR: {:.2f}, CI_L: {:.2f}, CI_U: {:.2f})".format(mdc.auc, mdc.trr, mdc.auc_ci_lower,mdc.auc_ci_upper)
                    else:
                        annotation_text = "AUC={:.2f} (CI_L: {:.2f}, CI_U: {:.2f})".format(mdc.auc, mdc.auc_ci_lower, mdc.auc_ci_upper)
                else:
                    if isOptOut:
                        annotation_text = "trAUC={:.2f}\n(TRR: {:.2f}, CI_L: {:.2f}, CI_U: {:.2f}, T#: {}, NT#: {})".format(mdc.auc, mdc.trr, mdc.auc_ci_lower,mdc.auc_ci_upper,mdc.t_num, mdc.nt_num)
                    else:
                        annotation_text = "AUC={:.2f}\n(CI_L: {:.2f}, CI_U: {:.2f}, T#: {}, NT#: {})".format(mdc.auc, mdc.auc_ci_lower,mdc.auc_ci_upper, mdc.t_num, mdc.nt_num)
            else:
                if args.noNum:
                    if isOptOut:
                        annotation_text = "trAUC={:.2f}\n(TRR: {:.2f})".format(mdc.auc, mdc.trr)
                    else:
                        annotation_text = "AUC={:.2f}".format(mdc.auc)
                else:
                    if isOptOut:
                        annotation_text = "trAUC={:.2f}\n(TRR: {:.2f}, T#: {}, NT#: {})".format(mdc.auc, mdc.trr, mdc.t_num, mdc.nt_num)
                    else:
                        annotation_text = "AUC={:.2f}\n(T#: {}, NT#: {})".format(mdc.auc, mdc.t_num, mdc.nt_num)

            annotation_list.append(Annotation(annotation_text, roc_annotation_parameters))

            if mdc.d is not None:
                x = mdc.dpoint[0]
                y = mdc.dpoint[1]
                if (y <= .5):
                    x += .1
                elif (.5 < y and y < .9):
                    x -= .1
                elif (y >= .9):
                    y -= .1
                d_annotation_paramaters = {"xy":(mdc.dpoint[0], mdc.dpoint[1]), "xycoords":'data',
                                           "xytext":(x, y), "textcoords":'data',
                                           "arrowprops":{"arrowstyle":"->", "connectionstyle":"arc3,rad=0"},
                                           "size":8, "va":'center', "ha":'center', "bbox":{"boxstyle":"round4", "fc":"w"}}

                annotation_list.append(Annotation("d' = {:.2f}".format(mdc.d), d_annotation_paramaters))
        
    #*-* Plotting *-*
    logger.debug("Plotting...")
    # Creation of the Renderer
    myRender = Render(plot_type=args.plotType, plot_options=plot_opts)
    # Plotting
    annotations=[], plot_type=None, plot_options=None, display=True, multi_fig=False
    myfigure = myRender.plot(MDC_list, annotations=annotation_list, display=args.display, multi_fig=args.multiFigs)

    # Output process
    outputFigure(myfigure, args.outputFolder, args.outputFileNameSuffix, args.plotType)
    
    # If we need to dump the used plotting options
    if args.dumpPlotParams:
        dumpPlotOptions(args.outputFolder, opts_list, plot_opts)





