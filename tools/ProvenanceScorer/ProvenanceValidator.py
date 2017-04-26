#!/usr/bin/env python2
import sys
import os
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Validator")
sys.path.append(lib_path)
from validator import validator,printq

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib")
sys.path.append(lib_path)
import json
from jsonschema import validate,exceptions
import argparse
import pandas as pd
import numpy as np

class ProvenanceValidator(validator):
    def __init__(self,sysfname,idxfname):
        self.sysname=sysfname
        self.idxname=idxfname

    def nameCheck(self,NCID):
        printq('Validating the name of the system file...')

        sys_pieces = self.sysname.rsplit('.',1)
        sys_ext = sys_pieces[1]
        if sys_ext != 'csv':
            printq('ERROR: Your system output does not appear to be a csv.',True)
            return 1
    
        fileExpid = sys_pieces[0].split('/')
        dirExpid = fileExpid[-2]
        fileExpid = fileExpid[-1]
        if fileExpid != dirExpid:
            printq("ERROR: Please follow the naming convention. The system output should follow the naming <EXPID>/<EXPID>.csv.",True)
            return 1
    
        taskFlag = 0
        teamFlag = 0
        sysPath = os.path.dirname(self.sysname)
        sysfName = os.path.basename(self.sysname)

        arrSplit = sysfName.split('_')
        if len(arrSplit) < 7:
            printq("ERROR: There are not enough arguments to verify in the name.")
            return 1
        elif len(arrSplit) > 7:
            printq("ERROR: The team name must not include underscores '_'.",True)
            teamFlag = 1

        team = arrSplit[0]
        ncid = arrSplit[1]
        data = arrSplit[2]
        task = arrSplit[3]
        condition = arrSplit[4]
        sys = arrSplit[5]
        version = arrSplit[6]
    
        if team == '':
            printq("ERROR: The team name must not include underscores '_'.",True)
            teamFlag = 1
        if ncid != NCID:
            printq("ERROR: The NCID must be {}.".format(NCID),True)
            ncidFlag = 1
        task = task.lower()
        self.task = task
        if (task != 'provenance') and (task != 'provenancefiltering'):
            printq("ERROR: " + task + " unrecognized. Only provenance or provenancefiltering are recognized. Make sure your team name does not include underscores '_'.",True)
            taskFlag = 1
    
        if (taskFlag == 0) and (teamFlag == 0):
            printq('The name of this file is valid!')
        else:
            printq('The name of the file is not valid. Please review the requirements in the eval plan.',True)
            return 1 
        
    def contentCheck(self,identify=False,neglectMask=False,reffname=0):
        printq('Validating the syntactic content of the system output.')
        index_dtype = {'TaskID':str,
                 'ProvenanceProbeFileID':str,
                 'ProvenanceProbeFileName':str,
                 'ProvenanceProbeWidth':np.int64,
                 'ProvenanceProbeHeight':np.int64}
        idxfile = pd.read_csv(self.idxname,sep='|',dtype=index_dtype,na_filter=False)
        sysfile = pd.read_csv(self.sysname,sep='|',na_filter=False)
    
        dupFlag = 0
        xrowFlag = 0
        jsonFlag = 0
        matchFlag = 0
        
        if sysfile.shape[1] < 2:
            printq("ERROR: The number of columns of the system output file must be at least 2. Make sure you are using '|' to separate your columns.",True)
            return 1

        sysHeads = list(sysfile.columns)
        allClear = True
#        truelist = ["ProbeFileID","ConfidenceScore","OutputProbeMaskFileName","OptOut"]
        truelist = ["ProvenanceProbeFileID","ProvenanceOutputFileName"]

        for i in range(0,len(truelist)):
            allClear = allClear and (truelist[i] in sysHeads)
            if not (truelist[i] in sysHeads):
#                headlist = []
#                properhl = []
#                for i in range(0,len(truelist)):
#                    if sysHeads[i] != truelist[i]:
#                        headlist.append(sysHeads[i])
#                        properhl.append(truelist[i]) 
#                printq("ERROR: Your header(s) " + ', '.join(headlist) + " should be " + ', '.join(properhl) + " respectively.",True)
                printq("ERROR: The required column {} is absent.".format(truelist[i]),True)

        if not allClear:
            return 1

        testMask = True #False
#        if "OutputProbeMaskFileName" in sysHeads:
#            testMask = True
        
        if sysfile.shape[0] != sysfile.drop_duplicates().shape[0]:
            rowlist = range(0,sysfile.shape[0])
            printq("ERROR: Your system output contains duplicate rows for ProvenanceProbeFileID's: "
                    + ' ,'.join(list(map(str,sysfile['ProvenanceProbeFileID'][sysfile.duplicated()])))    + " at row(s): "
                    + ' ,'.join(list(map(str,[i for i in rowlist if sysfile.duplicated()[i]]))) + " after the header. Please delete these row(s).",True)
            dupFlag = 1
        
        if sysfile.shape[0] != idxfile.shape[0]:
            printq("ERROR: The number of rows in your system output ({}) does not match the number of rows in the index file ({}).".format(sysfile.shape[0],idxfile.shape[0]),True)
            xrowFlag = 1
        
        if not ((dupFlag == 0) and (xrowFlag == 0)):
            printq("The contents of your file are not valid!",True)
            return 1
        
        sysfile['ProvenanceProbeFileID'] = sysfile['ProvenanceProbeFileID'].astype(str)
        sysfile['ProvenanceOutputFileName'] = sysfile['ProvenanceOutputFileName'].astype(str) 
#        sysfile['ConfidenceScore'] = sysfile['ConfidenceScore'].astype(np.float64)

#        idxfile['ProbeFileID'] = idxfile['ProbeFileID'].astype(str) 
#        idxfile['ProbeHeight'] = idxfile['ProbeHeight'].astype(np.float64) 
#        idxfile['ProbeWidth'] = idxfile['ProbeWidth'].astype(np.float64) 

        sysPath = os.path.dirname(self.sysname)
    
        for i in range(0,sysfile.shape[0]):
            if not (sysfile['ProvenanceProbeFileID'][i] in idxfile['ProvenanceProbeFileID'].unique()):
                printq("ERROR: " + sysfile['ProvenanceProbeFileID'][i] + " does not exist in the index file.",True)
                matchFlag = 1
                continue
#                printq("The contents of your file are not valid!",True)
#                return 1
            if not (idxfile['ProvenanceProbeFileID'][i] in sysfile['ProvenanceProbeFileID'].unique()):
                printq("ERROR: " + idxfile['ProvenanceProbeFileID'][i] + " seems to have been missed by the system file.",True)
                matchFlag = 1
                continue

            #check mask validation
            if testMask and not neglectMask:
                outputJsonName = sysfile['ProvenanceOutputFileName'][i]
                if outputJsonName in [None,'',np.nan,'nan']:
                    printq("The json for file " + sysfile['ProvenanceProbeFileID'][i] + " appears to be absent. Skipping it.")
                    continue
                jsonFlag = jsonFlag | jsonCheck(os.path.join(sysPath,sysfile['ProvenanceOutputFileName'][i]),sysfile['ProvenanceProbeFileID'][i],self.task)
        
        #final validation
        if (jsonFlag == 0) and (dupFlag == 0) and (xrowFlag == 0) and (matchFlag == 0):
            printq("The contents of your file are valid!")
        else:
            printq("The contents of your file are not valid!",True)
            return 1

def jsonCheck(provfile,provID,task):
    #toggle with task
    refschema = os.path.join(os.path.dirname(os.path.abspath(__file__)),'ProvenanceFilteringSchema-1.2.json')
    if task == 'provenance':
        refschema = os.path.join(os.path.dirname(os.path.abspath(__file__)),'ProvenanceGraphBuildingSchema-1.2.json')

    #read the schema
    schema = 0
    with open(refschema) as schema_data:
        schema = json.load(schema_data)
    
    #read the submission json
    prov = 0
    with open(provfile) as prov_data:
        prov = json.load(prov_data)
    try:
        validate(prov,schema)
        return 0
    except exceptions.ValidationError:
        provclause=''
        if task == 'provenance':
            provclause=' Make sure to include links!'

        print("{} did not pass the {} schema check!{}".format(provfile,task,provclause))
        return 1
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Score Medifor ProvenanceFiltering task output")
    parser.add_argument("-x", "--index_file", help="Task Index file", type=str, required=True)
    parser.add_argument("-s", "--system_output_file", help="System output file (i.e. <EXPID>.csv)", type=str, required=True)
    parser.add_argument("-t", "--task", help="Evaluation task. Required only if name checking is not done.", type=str)
    parser.add_argument('-nc','--nameCheck',action="store_true",\
    help='Check the format of the name of the file in question to make sure it matches up with the evaluation plan.')
    parser.add_argument('-nm','--neglectJSON',action="store_true",\
    help="Neglect JSON validation.")
    parser.add_argument('--ncid',type=str,default="NC17",\
    help="the NCID to validate against.")
    parser.add_argument('-v','--verbose',type=int,default=None,\
    help='Control print output. Select 1 to print all non-error print output and 0 to suppress all printed output (bar argument-parsing errors).',metavar='0 or 1')
    args = parser.parse_args()

    verbose = args.verbose
    if verbose==1:
        def printq(mystring,iserr=False):
            print(mystring)
    elif verbose==0:
        printq = lambda *x : None
    else:
        def printq(mystring,iserr=False):
            if iserr:
                print(mystring)

    validation = ProvenanceValidator(args.system_output_file,args.index_file)
    if args.task:
        validation.task = args.task
    exit(validation.fullCheck(args.nameCheck,False,args.ncid,args.neglectJSON))
