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


args = args_parser(command_line=False)

# *---------- Paths processing ----------*
output_folder = args.output.parent
directory_abspaths = [args.datasetDir.resolve(), args.sysDir.resolve()]
file_abspaths = [args.system.resolve(), 
                 args.datasetDir.resolve() / args.ref, 
                 args.datasetDir.resolve() / args.index]
process_args_paths(directory_abspaths, file_abspaths, [output_folder])

# *-------- Hard coded variables --------*

ss_dicts = [{"name":"Full_False_NA_Crop",
             "sub_output_folder":"sub_output_1",
             "description":"",
             "options":("""--outMeta --outSubMeta --dump --farStop 0.05 --ciLevel 0.90 --ci """
                       """-qm "Operation==['TransformCrop', 'TransformCropResize'] or PlugInName==['CropByPercentage','FaceCrop']" """
                       """-t manipulation --plotTitle kitware-holistic-image-v18_20190327-120000""")},
            {"name":"Full_False_NA_None",
             "sub_output_folder":"sub_output_2",
             "description":"",
             "options":("""--outMeta --outSubMeta --dump --farStop 0.05 --ciLevel 0.90 --ci """
                       """-qm "TaskID==['manipulation']" """
                       """-t manipulation --plotTitle kitware-holistic-image-v18_20190327-120000""")},
            {"name":"Full_True_NA_Crop",
             "sub_output_folder":"sub_output_3",
             "description":"",
             "options":("""--outMeta --outSubMeta --dump --farStop 0.05 --ciLevel 0.90 --ci """
                       """-qm "Operation==['TransformCrop', 'TransformCropResize'] or PlugInName==['CropByPercentage','FaceCrop']" """
                       """-t manipulation --plotTitle kitware-holistic-image-v18_20190327-120000""")},
            {"name":"Full_True_NA_None",
             "sub_output_folder":"sub_output_4",
             "description":"",
             "options":("""--outMeta --outSubMeta --dump --farStop 0.05 --ciLevel 0.90 --ci """
                       """-qm "TaskID==['manipulation']"  """
                       """-t manipulation --plotTitle kitware-holistic-image-v18_20190327-120000""")}]

script_path = "../DetectionScorer.py"
output_file_suffix = args.output.name

# *---------------------------------------*

command_template = "python {script_path} --sysDir {sysDir} --refDir {refDir} -s {system} -x {index} -r {ref} -o {output} {verbose} {options} 1> {stdout} 2> {stderr}"

for ss in ss_dicts:
    start = time.time()
    print("Processing query '{}'... ".format(ss["name"]), end='', flush=True)

    # Output folder organisation
    sub_output_path = output_folder / ss["sub_output_folder"]
    sub_output_path.mkdir(parents=True, exist_ok=True)

    stdout_filepath = sub_output_path / "detection_scorer.stdout"
    stderr_filepath = sub_output_path / "detection_scorer.stderr"

    # Command creation
    cmd_name = "{}.command.sh".format(ss["name"])
    cmd = command_template.format(script_path=script_path, sysDir=args.sysDir, refDir=args.datasetDir, 
                                  system=args.system, index=args.index, ref=args.ref, output=sub_output_path / output_file_suffix, 
                                  verbose='-v', stdout=stdout_filepath, stderr=stderr_filepath, options=ss["options"])

    # Command storage
    with open(sub_output_path / cmd_name,'w') as f:
        f.write(cmd)

    # Command call
    os.system(cmd)
    print("Done. ({:.2f}s)".format(time.time() - start))


