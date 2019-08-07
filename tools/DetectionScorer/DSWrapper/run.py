# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import shlex
import argparse
# import subprocess
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

def args_parser(command_line=True):
    if command_line:
        parser = argparse.ArgumentParser(description=None)
        parser.add_argument("-I", "--scoring-dict", help="path to the json file describing each sub scoring parameters", type=Path)
        parser.add_argument("-g", "--plotgroup-dict", help="path to the json file describing each plot group", type=Path)
        parser.add_argument("-d", "--datasetDir", help="path to the dataset directory", type=Path)
        parser.add_argument("-S", "--sysDir", help="path to the dataset directory", type=Path)
        parser.add_argument("-s", "--system", help="path to the system output", type=Path)      
        parser.add_argument("-i", "--index", help="path to the index file", type=Path)
        parser.add_argument("-r", "--ref", help="path to the ref folder", type=Path)
        parser.add_argument("-o", "--output", help="path to the output folder", type=Path)
        parser.add_argument("--detection-scorer-path", default="./DetectionScorer.py",help="path to the detection scorer script",type=Path)
        parser.add_argument("--dm-render-path", default="./DMRender.py",help="path to the DMRender script",type=Path)
        args = parser.parse_args()
    else:
        class ArgsNameSpace():
            def __init__(self):
                self.scoring_dict = Path("/Users/tnk12/Documents/MediScoreV2/tools/DetectionScorer/DSWrapper/subscorings.json")
                self.plotgroup_dict = Path("/Users/tnk12/Documents/MediScoreV2/tools/DetectionScorer/DSWrapper/plot_groups.json")
                self.datasetDir = Path("/Users/tnk12/Documents/MediScoreV2/tools/DetectionScorer/DSWrapper/")
                self.sysDir = Path("/Users/tnk12/Documents/MediScoreV2/tools/DetectionScorer/DSWrapper/system/")
                self.system = Path("/Users/tnk12/Documents/MediScoreV2/tools/DetectionScorer/DSWrapper/system/kitware-holistic-image-v18_20190327-120000.csv")
                self.output = Path("/Users/tnk12/Documents/MediScoreV2/tools/DetectionScorer/DSWrapper/output/nist_001")
                self.index = Path("indexes/MFC19_EvalPart1-manipulation-image-index.csv")
                self.ref = Path("reference/manipulation-image/MFC19_EvalPart1-manipulation-image-ref.csv")
                self.detection_scorer_path = "/Users/tnk12/Documents/MediScoreV2/tools/DetectionScorer/DetectionScorer.py"
                self.dm_render_path = "/Users/tnk12/Documents/MediScoreV2/tools/DetectionScorer/DMRender.py"
            def __repr__(self):
                return "Args list:\n - {}".format("\n - ".join(["{:>6}: {}".format(a,v) for a,v in self.__dict__.items()]))
        args = ArgsNameSpace()
        
    return args

def process_args_paths(directory_abspaths, file_abspaths, path_make_dir):
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

def create_html(output_path, group_plots, ss_dicts, template_path, 
                group_plot_name="plot_ROC_all.pdf",sub_plot_name = "nist_001_qm_query_ROC.pdf"):
    file_loader = FileSystemLoader(str(template_path))
    env = Environment(loader=file_loader)
    template = env.get_template("base.html")
    template_variables = {"page_title": "Scoring Summary",
                          "container_id": "#container",
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

args = args_parser(command_line=False)

# *---------- Paths processing ----------*
output_folder = args.output.parent
templates_path = Path(os.getcwd()) / "templates" 

directory_abspaths = [args.datasetDir.resolve(), 
                      args.sysDir.resolve()]

file_abspaths = [args.system.resolve(), 
                 args.datasetDir.resolve() / args.ref, 
                 args.datasetDir.resolve() / args.index]

process_args_paths(directory_abspaths, file_abspaths, [output_folder])

with open(args.scoring_dict, 'r') as f:
    ss_dicts = json.load(f)

with open(args.plotgroup_dict, 'r') as f:
    group_plots = json.load(f)

# *-------- Hard coded variables --------*

output_file_prefix = args.output.name
display = '' # --display

# *--------------------------------------*

detection_scorer_command_template = "python {script_path} --sysDir {sysDir} --refDir {refDir} -s {system} -x {index} -r {ref} -o {output} {verbose} {options} 1> {stdout} 2> {stderr}"
dm_render_command_template = "python {script_path} -i {input} --plotType ROC {display} --outputFolder {output} --outputFileNameSuffix {output_fsuffix} --logtype {logtype} --console_log_level {console_log_level} 1> {stdout} 2> {stderr}"

# *======================== Scoring runs ========================*

for ss_key, ss in ss_dicts.items():
    start = time.time()
    print("Processing query '{}'... ".format(ss["name"]), end='', flush=True)

    # Output folder organisation
    sub_output_path = output_folder / ss["sub_output_folder"]
    sub_output_path.mkdir(parents=True, exist_ok=True)

    ds_stdout_filepath = sub_output_path / "detection_scorer.stdout"
    ds_stderr_filepath = sub_output_path / "detection_scorer.stderr"

    # Command creation
    cmd_name = "{}.command.sh".format(ss_key)
    cmd = detection_scorer_command_template.format(script_path=args.detection_scorer_path, sysDir=args.sysDir, refDir=args.datasetDir, 
                                  system=args.system, index=args.index, ref=args.ref, output=sub_output_path / output_file_prefix, 
                                  verbose='-v', stdout=ds_stdout_filepath, stderr=ds_stderr_filepath, options=ss["options"])
    cmd = remove_multiple_spaces(cmd)

    # Command storage
    with open(sub_output_path / cmd_name,'w') as f:
        f.write(cmd)

    # Command call
    exit_code = os.system(cmd)
    print("Done. ({:.2f}s), exit code = {}".format(time.time() - start, exit_code))

# *==================== Plot output handling ====================*

for gplot_key, gplot_data in group_plots.items():
    start = time.time()
    print("Plotting '{}'... ".format(gplot_data["name"]), end='', flush=True)

    sub_output_plot_path = output_folder / gplot_key
    sub_output_plot_path.mkdir(parents=True, exist_ok=True)

    dmr_stdout_filepath = sub_output_plot_path / "dm_render.stdout"
    dmr_stderr_filepath = sub_output_plot_path / "dm_render.stderr"

    input_list = []
    gplot_ss_dicts = gplot_data["ss_list"]
    for gplot_ss_dict in gplot_ss_dicts:
        sub_output_path = output_folder / ss_dicts[gplot_ss_dict["s_name"]]["sub_output_folder"]
        data_dict = {"path": str(sub_output_path / "{}_query_0.dm".format(output_file_prefix)),
                     "label": ss_dicts[gplot_ss_dict["s_name"]]["name"],
                     "gplot_ss_dict": True}
        line_options = gplot_ss_dict["s_line_options"]
        input_list.append([data_dict, line_options])

    # Command creation
    cmd_name = "dm_render.command.sh"
    cmd = dm_render_command_template.format(script_path=args.dm_render_path, input='"{}"'.format(input_list), output=sub_output_plot_path, 
                                            output_fsuffix="plot", display=display, logtype=1, console_log_level="INFO", 
                                            stdout=dmr_stdout_filepath, stderr=dmr_stderr_filepath)
    cmd = remove_multiple_spaces(cmd)

    # Command storage
    with open(sub_output_plot_path / cmd_name,'w') as f:
        f.write(cmd)

    # Command call
    exit_code = os.system(cmd)
    print("Done. ({:.2f}s), exit code = {}".format(time.time() - start, exit_code))

# *=================== Html summary generation ===================*

html_summary = create_html(output_folder, group_plots, ss_dicts, templates_path)
with open(output_folder / "generated_summary.html","w") as f:
    f.write(html_summary)
    f.write("\n")

