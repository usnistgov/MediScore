from TaskData import Data
from TaskData import WHPData
from TaskData import TaskDefinition
import pandas as pd
import numpy
import os
from TaskDefAbs import TaskDefSplice


HPdir="/Volumes/medifor/HPFiles/images/hp/"
HPj="/Volumes/medifor/NC2017/00_SourceData/old/HP_SequesteredPool_SubSet_100_20161019.json"
JTj="/Volumes/medifor/NC2017/00_SourceData/JT_SubSet_56_20161103.txt"  #JT_SubSet_56_20161103.txt #JT_SubSet_1_20161031.txt" #JT_SubSet_20_20161021.txt"    JT_SubSet_4_20161103.txt
task="splice"
NCdir="/Volumes/medifor/NC2017/32_NC2017_Beta2_Splice/NC2017_Beta_TestSet2"
Worldj="/Volumes/medifor/NC2017/00_SourceData/World_SequesteredPool_SubSet_100_20161019.json"
Worlddir="/Volumes/medifor/WorldFiles/images/"

filedf=pd.DataFrame(columns=('Type', 'RealfileName', 'ProbeFileID', 'ProbeFileName', 'IsTarget', 'ProbeWidth', 'ProbeHeight', 'TaskID', 'RealmaskName', 'ProbeMaskName', 'DonorType', 'RealdonorName', 'DonorFileID', 'DonorFileName', 'DonorWidth', 'DonorHeight', 'DonorMaskName', 'RealDonorMaskName'))

indexheader=['TaskID','ProbeFileID', 'ProbeFileName', 'ProbeWidth', 'ProbeHeight', 'DonorFileID', 'DonorFileName', 'DonorWidth', 'DonorHeight']
refheader=['TaskID','ProbeFileID','ProbeFileName','DonorFileID','DonorFileName','IsTarget' ]

TaskData=Data(JTj, HPj, HPdir, NCdir, Worldj, Worlddir)
TaskDef=TaskDefSplice(task, TaskData)

#probe= TaskDef.getProbes(TaskData)
#alldonors=TaskDef.getDonors(TaskData)
trials=TaskDef.getTrials()
#exit()
#print trials

for t in trials:
    probe=t[0]
    donor=t[1]
    pdons=[]
    md5=TaskData.getMD5(probe)
    dim=TaskData.getDim(probe)
    fName=TaskData.getName(probe,md5,"probe")
#    masktup=TaskDef.getMask(probe[i],TaskData)    
#    print masktup
#    if masktup[1]!='':
#        maskmd5=TaskData.getMD5(masktup)
#        maskgname=TaskData.getName(masktup,maskmd5,"reference/manipulation/mask")
#    else:
#        maskmd5=''
#        maskgname=''


    dondim=TaskData.getDim(donor)
    donmd5=TaskData.getMD5(donor)
    donname=TaskData.getName(donor,donmd5,"world")
    masktup=["0","1"]
    maskgname=""
#    print probe
#    print donor
    targ=TaskDef.isTarg(t)
    
#    if targ=="Y":
#        donmasktup=TaskDef.getDonorMask(t)
#        donmaskmd5=TaskData.getMD5(donmasktup)
#        donmasknm=TaskData.getName(donmasktup,donmaskmd5,"reference/splice/mask")
#    else:
    donmasktup=["",""]
    donmasknm=""
    filedf.loc[len(filedf)]=[probe[0], probe[1], md5, fName, TaskDef.isTarg(t), str(dim[0]), str(dim[1]), task, masktup[1], maskgname, donor[0], donor[1], donmd5, donname, str(dondim[0]), str(dondim[1]), donmasknm, donmasktup[1] ]
    
    
#print filedf

#print filedf
TaskData.createDirs()

TaskDef.linkProbes([tuple(x) for x in filedf[['Type','RealfileName','ProbeFileName']].values])
#TaskData.linkMasks([tuple(x) for x in filedf[['Type', 'RealmaskName', 'ProbeMaskName']].values])
TaskDef.linkDonors([tuple(x) for x in filedf[['DonorType', 'RealdonorName', 'DonorFileName']].values])
# ' DonorType', 'RealdonorName', 'DonorFileID', 'DonorFileName', 'DonorWidth', 'DonorHeight'
filedf.to_csv(path_or_buf=NCdir+"/indexes/NC2017_Splice_Index.csv",index=False,columns=indexheader,sep='|')
filedf.to_csv(path_or_buf=NCdir+"/reference/splice/NC2017_Splice_Reference.csv",index=False,columns=refheader,sep='|')

print len(trials)
print len(filedf)

#print test
#print test[120][1]
#print test.hpjson.files[99]
#print len(test.wjson.files)
#print test.wjson.jl
