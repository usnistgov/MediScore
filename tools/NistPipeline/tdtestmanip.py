from TaskData import Data
from TaskData import WHPData
from TaskData import TaskDefinition
import pandas as pd
import numpy
import os
from TaskDefAbs import TaskDefManipulation


HPdir="/Volumes/medifor/HPFiles/images/hp/"
HPj="/Volumes/medifor/NC2017/00_SourceData/old/HP_SequesteredPool_SubSet_100_20161019.json"  #HP_SequesteredPool_SubSet_1_20161030.json" #HP_SequesteredPool_SubSet_100_20161019.json"
JTj="/Volumes/medifor/NC2017/00_SourceData/JT_SubSet_56_20161103.txt" #JT_SubSet_1_20161031.txt" #JT_SubSet_20_20161021.txt"
task="manipulation"
NCdir="/Volumes/medifor/NC2017/30_NC2017_Beta/NC2017_Data_Beta_test1109"


filedf=pd.DataFrame(columns=('RealfileName', 'ProbeFileID', 'ProbeFileName', 'IsTarget', 'ProbeWidth', 'ProbeHeight', 'TaskID', 'RealmaskName', 'ProbeMaskName', 'Type'))

indexheader=['TaskID','ProbeFileID', 'ProbeFileName', 'ProbeWidth', 'ProbeHeight']
refheader=['TaskID','ProbeFileID','ProbeFileName','IsTarget', 'ProbeMaskName']



TaskData=Data(JTj, HPj, HPdir, NCdir)
TaskDef=TaskDefManipulation(task, TaskData)

probe= TaskDef.getTrials()

for i in range(0,len(probe)):
    md5=TaskData.getMD5(probe[i])
    dim=TaskData.getDim(probe[i])
    fName=TaskData.getName(probe[i],md5,"probe")
    masktup=TaskDef.getMask(probe[i])    
    print masktup
    if masktup[1]!='':
        maskmd5=TaskData.getMD5(masktup)
        maskgname=TaskData.getName(masktup,maskmd5,"reference/manipulation/mask")
    else:
        maskmd5=''
        maskgname=''

    filedf.loc[len(filedf)]=[probe[i][1], md5, fName, TaskDef.isTarg(probe[i]), str(dim[0]), str(dim[1]), task, masktup[1], maskgname, probe[i][0] ]
    


TaskData.createDirs()

TaskData.linkProbes([tuple(x) for x in filedf[['Type','RealfileName','ProbeFileName']].values])
TaskData.linkMasks([tuple(x) for x in filedf[['Type', 'RealmaskName', 'ProbeMaskName']].values])

filedf.to_csv(path_or_buf=NCdir+"/indexes/index.csv",index=False,columns=indexheader,sep='|')
filedf.to_csv(path_or_buf=NCdir+"/reference/manipulation/reference.csv",index=False,columns=refheader,sep='|')






#print test
#print test[120][1]
#print test.hpjson.files[99]
#print len(test.wjson.files)
#print test.wjson.jl
