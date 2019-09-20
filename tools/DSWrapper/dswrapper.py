# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import shlex
import argparse
import configparser

if sys.version_info[:2] < (3, 4):
    from platform import python_version
    print("The lowest supported python version for this script is python 3.4.\nYour current python version is {}".format(python_version()))
    sys.exit()

from pathlib import Path
from jinja2 import Environment, FileSystemLoader

def args_parser(command_line, test_name=None, test_config_file=Path("./test/tests.ini")):
    """This function returns and object with command lines arguments as attributes.
    Either stored in an argparse.Namespace object or a custom ArgsNameSpace class for testing.

    Args:
        command_line (boolean):
            - True: The command line is parsed with and argparser and used.
            - False: The parameters are loaded from a configuration file located at test_config_file
        test_name (string): if command_line is False, it defines the config section's name from where 
                            parameters will be loaded
        test_config_file (pathlib.Path): Path to the config file

    Returns:
        args (object): Either an argparse.Namespace object or a custom ArgsNameSpace object

    """
    if command_line:
        parser = argparse.ArgumentParser(description=None)
        parser.add_argument("-I", "--scoring-dict", help="path to the json file describing each sub scoring parameters", type=Path)
        parser.add_argument("-g", "--plotgroup-dict", help="path to the json file describing each plot group", type=Path)
        parser.add_argument("-d", "--datasetDir", help="path to the dataset directory", type=Path)
        parser.add_argument("-S", "--sysDir", help="path to the system directory (not used)", type=Path, default=Path())
        parser.add_argument("-s", "--system", help="path to the system output", type=Path)      
        parser.add_argument("-i", "--index", help="path to the index file", type=Path)
        parser.add_argument("-r", "--ref", help="path to the ref folder", type=Path)
        parser.add_argument("-o", "--output-dir", help="path to the output folder", type=Path)
        parser.add_argument("-O", "--output-prefix", help="prefix to the filenames", default="")
        parser.add_argument("-n", "--summary-filename", help="name of the html generated summary ", default="generated_summary.html")
        parser.add_argument("-p", "--only-group-plots", help="Flag to display only the groups plots. The individuals sub-scorings plots are still computed.", action="store_true")
        parser.add_argument("-m", "--mediscore-path", help="path to the root of mediscore folder",type=Path)
        args = parser.parse_args()
        return args

    else: # Test Mode
        if test_config_file.is_file():
            config = configparser.ConfigParser()
            config.read(test_config_file)

            if test_name:
                if test_name in config:
                    test_config = config[test_name]

                    class ArgsNameSpace():
                        def __init__(self, config):
                            self.scoring_dict = Path(config["scoring_dict"])
                            self.plotgroup_dict = Path(config["plotgroup_dict"])
                            self.datasetDir = Path(config["datasetDir"])
                            self.sysDir = Path(config["sysDir"])
                            self.system = Path(config["system"])
                            self.output_dir = Path(config["output_dir"])
                            self.output_prefix = config["output_prefix"]
                            self.index = Path(config["index"])
                            self.ref = Path(config["ref"])
                            self.only_group_plots = config.getboolean("only-group-plots")
                            self.summary_filename = config["summary_filename"]
                            self.mediscore_path = Path(config["mediscore_path"])
                        
                        def __repr__(self):
                            return "Args list:\n - {}".format("\n - ".join(["{:>6}: {}".format(a,v) for a,v in self.__dict__.items()]))
                            
                    args = ArgsNameSpace(test_config)
                    return args

                else:
                    print("Error (Test Mode): No test name provided")
                    sys.exit()
            else:
                print("Error (Test Mode): No test name provided")
                sys.exit()
        else:
            print("Error (Test Mode): No test config file found ({})".format(str(test_config_file)))

def process_args_paths(directory_abspaths, file_abspaths, path_make_dir):
    """This function processes some of the paths provided in the parameters
    It checks:
        - If the folders exist
        - If the file exists
    it creates if needed empty directories   

    Args:
        directory_abspaths (list): list of directories pathlib.Path to check
        file_abspaths (list): list of file pathlib.Path to check
        path_make_dir (list): list of directories pathlib.Path to create

    """
    for directory in path_make_dir:
        directory.mkdir(parents=True, exist_ok=True)
    
    dir_paths_validation = [p.is_dir() for p in directory_abspaths]
    file_paths_validation = [p.is_file() for p in file_abspaths]
    all_paths = directory_abspaths + file_abspaths
    all_paths_valid = dir_paths_validation + file_paths_validation
    if not all(all_paths_valid):
        invalid_paths = [path for path, valid in zip(all_paths, all_paths_valid) if not valid]
        print("Error: The following paths are invalid..\n{}".format('\n'.join(map(str, invalid_paths))))
        sys.exit(1)
    else:
        print("All paths are valid")

def remove_multiple_spaces(string):
    return ' '.join(string.split())

def create_html(output_path, group_plots, ss_dicts, template_path, only_group_plots,
                group_plot_name="plot_ROC_all.pdf", sub_plot_name = "nist_001_qm_query_ROC.pdf"):
    """This function creates the html string representing the summary.
    It uses a set of jinja templates interpolated with the provided parameters.

    Args:
        output_path (pathlib.Path): Path to the output folder
        group_plots (dict): Plots dictionnary loaded from the plots json file
        ss_dicts (dict): Scorings dictionnary loaded from the scorings json file
        template_path (pathlib.Path): Path to the templates folder
        only_group_plots (boolean): boolean to display only group plots in the html summary
        group_plot_name (string): Generic name of the group plot to display
        sub_plot_name (string): Generic name of the scoring plot to display

    Returns:
        html (string): the summary html output

    """
    file_loader = FileSystemLoader(str(template_path))
    env = Environment(loader=file_loader)
    template = env.get_template("base.html")
    template_variables = {"page_title": "Scoring Summary",
                          "container_id": "#container",
                          "only_group_plots": only_group_plots,
                          "group_plots":group_plots,
                          "ss_dicts":ss_dicts,
                          "gplot_filename":group_plot_name,
                          "splot_filename":sub_plot_name,
                          "output_path":str(output_path),
                          "pdf_reader_width":"1000px",
                          "pdf_reader_height":"800px"
                         }    
    html = template.render(template_variables)
    return html


# *---------------------------------- Main ----------------------------------*

args = args_parser(len(sys.argv) > 1, test_name="test_2", test_config_file=Path("./test/tests.ini"))

# *---------- Paths and files processing ----------*
detection_scorer_path = args.mediscore_path / "tools/DetectionScorer/DetectionScorer.py"
dm_render_path = args.mediscore_path / "tools/DetectionScorer/DMRender.py"
templates_path = args.mediscore_path / "tools/DSWrapper/templates"

directory_abspaths = [args.mediscore_path.resolve(), 
                      args.datasetDir.resolve(), 
                      args.sysDir.resolve(),
                      templates_path.resolve()]

file_abspaths = [args.system.resolve(), 
                 args.datasetDir.resolve() / args.ref, 
                 args.datasetDir.resolve() / args.index, 
                 detection_scorer_path.resolve(),
                 dm_render_path.resolve()]

process_args_paths(directory_abspaths, file_abspaths, [args.output_dir])

with open(args.scoring_dict, 'r') as f:
    ss_dicts = json.load(f)

with open(args.plotgroup_dict, 'r') as f:
    group_plots = json.load(f)

# *--------------------------------------*

detection_scorer_command_template = "python {script_path} --sysDir {sysDir} --refDir {refDir} -s {system} -x {index} -r {ref} -o {output} {verbose} {options}"
dm_render_command_template = "python {script_path} -i {input} --plotType ROC {display} --outputFolder {output} --outputFileNameSuffix {output_fprefix} --logtype {logtype} --console_log_level {console_log_level}"
cmd_output_string = " 1> {stdout} 2> {stderr}"

process_start = time.time()

# *======================== Scoring runs ========================*

for ss_key, ss in ss_dicts.items():
    start = time.time()
    print("Processing query '{}'... ".format(ss["name"]), end='', flush=True)

    # Output folder organisation
    sub_output_path = args.output_dir / ss["sub_output_folder"]
    sub_output_path.mkdir(parents=True, exist_ok=True)

    ds_stdout_filepath = sub_output_path / "detection_scorer.stdout"
    ds_stderr_filepath = sub_output_path / "detection_scorer.stderr"

    # Command creation
    cmd_name = "{}.command.sh".format(ss_key)
    cmd = detection_scorer_command_template.format(script_path=detection_scorer_path, sysDir=args.sysDir, refDir=args.datasetDir, 
                                  system=args.system, index=args.index, ref=args.ref, output=sub_output_path / args.output_prefix, 
                                  verbose='-v', options=ss["options"])
    cmd = remove_multiple_spaces(cmd)

    # Command storage
    with open(sub_output_path / cmd_name,'w') as f:
        f.write(cmd)

    cmd += cmd_output_string.format(stdout=ds_stdout_filepath, stderr=ds_stderr_filepath)
    # Command call
    exit_code = os.system(cmd)
    print("Done. ({:.2f}s)".format(time.time() - start))
    if exit_code != 0:
        print("Error: Something went wrong during the execution of the Detection Scorer.\nPlease check the error ouptut at {}".format(ds_stderr_filepath))
        sys.exit(1)

# *==================== Plot output handling ====================*

for gplot_key, gplot_data in group_plots.items():
    start = time.time()
    print("Plotting '{}'... ".format(gplot_data["name"]), end='', flush=True)

    sub_output_plot_path = args.output_dir / gplot_key
    sub_output_plot_path.mkdir(parents=True, exist_ok=True)

    dmr_stdout_filepath = sub_output_plot_path / "dm_render.stdout"
    dmr_stderr_filepath = sub_output_plot_path / "dm_render.stderr"

    input_list = []
    gplot_ss_dicts = gplot_data["ss_list"]
    for gplot_ss_dict in gplot_ss_dicts:
        sub_output_path = args.output_dir / ss_dicts[gplot_ss_dict["s_name"]]["sub_output_folder"]
        data_dict = {"path": str(sub_output_path / "{}_query_0.dm".format(args.output_prefix)),
                     "label": ss_dicts[gplot_ss_dict["s_name"]]["name"],
                     "show_label":True,
                     "gplot_ss_dict": True}
        line_options = gplot_ss_dict["s_line_options"]
        input_list.append([data_dict, line_options])

    # Command creation
    cmd_name = "dm_render.command.sh"
    cmd = dm_render_command_template.format(script_path=dm_render_path, input='"{}"'.format(input_list), output=sub_output_plot_path, 
                                            output_fprefix="plot", display='', logtype=1, console_log_level="INFO", 
                                            stdout=dmr_stdout_filepath, stderr=dmr_stderr_filepath)
    cmd = remove_multiple_spaces(cmd)

    # Command storage
    with open(sub_output_plot_path / cmd_name,'w') as f:
        f.write(cmd)

    cmd += cmd_output_string.format(stdout=dmr_stdout_filepath, stderr=dmr_stderr_filepath)

    # Command call
    exit_code = os.system(cmd)
    print("Done. ({:.2f}s)".format(time.time() - start))
    if exit_code != 0:
        print("Error: Something went wrong during the execution of DMRender.\nPlease check the error ouptut at {}".format(dmr_stderr_filepath))
        sys.exit(1)

# *=================== Html summary generation ===================*
html_file_output = args.output_dir / args.summary_filename
html_summary = create_html(args.output_dir, group_plots, ss_dicts, templates_path, args.only_group_plots, sub_plot_name= args.output_prefix+"_qm_query_ROC.pdf")
with open(html_file_output,"w") as f:
    f.write(html_summary)
    f.write("\n")
print("Html summary created and located at:\n{}".format(html_file_output))

print("Total runtime: {:.2f}s".format(time.time() - process_start))
