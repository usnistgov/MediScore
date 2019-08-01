import os
import sys
import time
import shlex
import argparse
# import subprocess
from pathlib import Path

def args_parser(command_line=True):
    if command_line:
        parser = argparse.ArgumentParser(description=None)
        parser.add_argument()
        parser.add_argument("-d", "--datasetDir", help="path to the dataset directory", type=Path)
        parser.add_argument("-d", "--sysDir", help="path to the dataset directory", type=Path)
        parser.add_argument("-s", "--system", help="path to the system output", type=Path)      
        parser.add_argument("-i", "--index", help="path to the index file", type=Path)
        parser.add_argument("-r", "--ref", help="path to the ref folder", type=Path)
        parser.add_argument("-o", "--output", help="path to the output folder", type=Path)
        args = parser.parse_args()
    else:
        class ArgsNameSpace():
            def __init__(self):
                self.datasetDir = Path("/Users/tnk12/Documents/MediScoreV2/tools/DetectionScorer/DSWrapper/")
                self.sysDir = Path("/Users/tnk12/Documents/MediScoreV2/tools/DetectionScorer/DSWrapper/system/")
                self.system = Path("/Users/tnk12/Documents/MediScoreV2/tools/DetectionScorer/DSWrapper/system/kitware-holistic-image-v18_20190327-120000.csv")
                self.output = Path("/Users/tnk12/Documents/MediScoreV2/tools/DetectionScorer/DSWrapper/output/nist_001")
                self.index = Path("indexes/MFC19_EvalPart1-manipulation-image-index.csv")
                self.ref = Path("reference/manipulation-image/MFC19_EvalPart1-manipulation-image-ref.csv")
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

args = args_parser(command_line=False)

# *---------- Paths processing ----------*
output_folder = args.output.parent
directory_abspaths = [args.datasetDir.resolve(), args.sysDir.resolve()]
file_abspaths = [args.system.resolve(), 
                 args.datasetDir.resolve() / args.ref, 
                 args.datasetDir.resolve() / args.index]
process_args_paths(directory_abspaths, file_abspaths, [output_folder])

# *-------- Hard coded variables --------*

ss_dicts = {"s1":{"name":"Full_False_NA_Crop",
                  "sub_output_folder":"sub_output_1",
                  "description":"",
                  "options":("""--outMeta --outSubMeta --dump --farStop 0.05 --ciLevel 0.90 --ci """
                             """-qm "Operation==['TransformCrop', 'TransformCropResize'] or PlugInName==['CropByPercentage','FaceCrop']" """
                              """-t manipulation --plotTitle kitware-holistic-image-v18_20190327-120000""")},
            "s2":{"name":"Full_False_NA_None",
                  "sub_output_folder":"sub_output_2",
                  "description":"",
                  "options":("""--outMeta --outSubMeta --dump --farStop 0.05 --ciLevel 0.90 --ci """
                            """-qm "TaskID==['manipulation']" """
                            """-t manipulation --plotTitle kitware-holistic-image-v18_20190327-120000""")},
            "s3":{"name":"Full_True_NA_Crop",
                  "sub_output_folder":"sub_output_3",
                  "description":"",
                  "options":("""--outMeta --outSubMeta --dump --farStop 0.05 --ciLevel 0.90 --ci """
                            """-qm "Operation==['TransformCrop', 'TransformCropResize'] or PlugInName==['CropByPercentage','FaceCrop']" """
                            """-t manipulation --plotTitle kitware-holistic-image-v18_20190327-120000""")},
            "s4":{"name":"Full_True_NA_None",
                  "sub_output_folder":"sub_output_4",
                  "description":"",
                  "options":("""--outMeta --outSubMeta --dump --farStop 0.05 --ciLevel 0.90 --ci """
                            """-qm "TaskID==['manipulation']"  """
                            """-t manipulation --plotTitle kitware-holistic-image-v18_20190327-120000""")}}

group_plots = {"gplot_1":["s1", "s2"],
              "gplot_2":["s2", "s3", "s4"]}

detection_scorer_path = "/Users/tnk12/Documents/MediScoreV2/tools/DetectionScorer/DetectionScorer.py"
dm_render_path = "/Users/tnk12/Documents/MediScoreV2/tools/DetectionScorer/DMRender.py"

output_file_suffix = args.output.name

# *---------------------------------------*

detection_scorer_command_template = "python {script_path} --sysDir {sysDir} --refDir {refDir} -s {system} -x {index} -r {ref} -o {output} {verbose} {options} 1> {stdout} 2> {stderr}"
dm_render_command_template = "python {script_path} -i {input} --plotType ROC {display} --outputFolder {output} --outputFileNameSuffix {output_fsuffix} --console_log_level {console_log_level} 1> {stdout} 2> {stderr}"

# *======================== Scoring runs ========================*

sub_output_folder_paths = []
for ss_key, ss in ss_dicts.items():
    start = time.time()
    print("Processing query '{}'... ".format(ss["name"]), end='', flush=True)

    # Output folder organisation
    sub_output_path = output_folder / ss["sub_output_folder"]
    sub_output_path.mkdir(parents=True, exist_ok=True)
    sub_output_folder_paths.append(sub_output_path)

    ds_stdout_filepath = sub_output_path / "detection_scorer.stdout"
    ds_stderr_filepath = sub_output_path / "detection_scorer.stderr"

    # Command creation
    cmd_name = "{}.command.sh".format(ss["name"])
    cmd = detection_scorer_command_template.format(script_path=detection_scorer_path, sysDir=args.sysDir, refDir=args.datasetDir, 
                                  system=args.system, index=args.index, ref=args.ref, output=sub_output_path / output_file_suffix, 
                                  verbose='-v', stdout=ds_stdout_filepath, stderr=ds_stderr_filepath, options=ss["options"])
    cmd = remove_multiple_spaces(cmd)

    # Command storage
    with open(sub_output_path / cmd_name,'w') as f:
        f.write(cmd)

    # Command call
    # os.system(cmd)
    print("Done. ({:.2f}s)".format(time.time() - start))

# *==================== Plot output handling ====================*

for gplot_name, ss_list in group_plots.items():
    start = time.time()
    print("Plotting '{}'... ".format(gplot_name), end='', flush=True)
    
    sub_output_plot_path = output_folder / gplot_name
    sub_output_plot_path.mkdir(parents=True, exist_ok=True)

    dmr_stdout_filepath = sub_output_plot_path / "dm_render.stdout"
    dmr_stderr_filepath = sub_output_plot_path / "dm_render.stderr"

    input_list = []
    for ss_key in ss_list:
        sub_output_path = output_folder / ss_dicts[ss_key]["sub_output_folder"]
        data_dict = {"path": str(sub_output_path / "{}_query_0.dm".format(output_file_suffix)),
                     "label": ss_dicts[ss_key]["name"],
                     "show_label": True}
        line_options = {}
        input_list.append([data_dict, line_options])

    # Command creation
    cmd_name = "dm_render.command.sh"
    cmd = dm_render_command_template.format(script_path=dm_render_path, input='"{}"'.format(input_list), output=sub_output_plot_path, 
                                            output_fsuffix="plot", display="--display", console_log_level="INFO", 
                                            stdout=dmr_stdout_filepath, stderr=dmr_stderr_filepath)
    cmd = remove_multiple_spaces(cmd)

    # Command storage
    with open(sub_output_plot_path / cmd_name,'w') as f:
        f.write(cmd)

    # Command call
    os.system(cmd)
    print("Done. ({:.2f}s)".format(time.time() - start))
