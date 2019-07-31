import os
import sys
import shlex
import subprocess

command_template = "python {script_path} --sysDir {sysDir} --refDir {refDir} -s {system} -x {index} -r {ref} -o {output} {verbose} {options}"

script_path = "../DetectionScorer.py"
sysDir = "/Users/tnk12/Documents/MediScoreV2/tools/DetectionScorer/DSWrapper/system/"
refDir = "/Users/tnk12/Documents/MediScoreV2/tools/DetectionScorer/DSWrapper/"
system = "/Users/tnk12/Documents/MediScoreV2/tools/DetectionScorer/DSWrapper/system/kitware-holistic-image-v18_20190327-120000.csv"
output = "/Users/tnk12/Documents/MediScoreV2/tools/DetectionScorer/DSWrapper/output/nist_001"
index = "indexes/MFC19_EvalPart1-manipulation-image-index.csv"
ref = "reference/manipulation-image/MFC19_EvalPart1-manipulation-image-ref.csv"


ss_dicts = [{"name":"x",
            "description":"y",
            "options":"""--outMeta --outSubMeta --dump --farStop 0.05 --ciLevel 0.90 --ci -qm "Operation==['TransformCrop', 'TransformCropResize'] or PlugInName==['CropByPercentage','FaceCrop']" -t manipulation --plotTitle kitware-holistic-image-v18_20190327-120000"""}]

for ss in ss_dicts:
    cmd = command_template.format(script_path=script_path, sysDir=sysDir, refDir=refDir, system=system, index=index, ref=ref, output=output, verbose='-v', options=ss["options"])
    with open("command_x.sh",'w') as f:
        f.write(cmd)
    print(shlex.split(cmd))
    p = subprocess.run(shlex.split(cmd), capture_output=True)
    print(p.stdout.decode())
    print(p.stderr.decode())


