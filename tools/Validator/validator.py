"""
* File: validator.py
* Date: 09/23/2016
* Translation by: Daniel Zhou
* Original Author: Yooyoung Lee
* Status: Complete 

* Description: This object validates the format of the input of the 
  system output along with the index file and some basic mask content.

* Requirements: This code requires the following packages:
    - cv2
    - numpy
    - pandas
  
  The rest should be available on your system.

* Inputs
    * -x, inIndex: index file name
    * -s, inSys: system output file name
    * -vt, valType: validator type: SSD or DSD
    * -v, verbose: Control printed output. -v 0 suppresss all printed output except argument parsng errors, -v 1 to print all output.

* Outputs
    * 0 if the files are validly formatted, 1 if the files are not.

* Disclaimer: 
This software was developed at the National Institute of Standards 
and Technology (NIST) by employees of the Federal Government in the
course of their official duties. Pursuant to Title 17 Section 105 
of the United States Code, this software is not subject to copyright 
protection and is in the public domain. NIST assumes no responsibility 
whatsoever for use by other parties of its source code or open source 
server, and makes no guarantees, expressed or implied, about its quality, 
reliability, or any other characteristic."
"""

import sys
import os
import cv2
import numpy as np
import pandas as pd
import contextlib
import StringIO
import subprocess
import multiprocessing
import ast
from abc import ABCMeta, abstractmethod
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../lib')
sys.path.append(lib_path)
#from printbuffer import printbuffer

#print_lock = multiprocessing.Lock() #for printout

@contextlib.contextmanager
def stdout_redirect(where):
    sys.stdout = where
    try:
        yield where
    finally:
        sys.stdout = sys.__stdout__

def printq(mystring,iserr=False):
    if iserr:
        print(mystring)

def is_finite_number(s):
    try:
        s = float(s)
        return np.isfinite(s)
    except ValueError:
        return False

def is_integer(s):
    try:
        s = int(s)
        return True
    except ValueError:
        return False

def checkProbe(args):
    return SSD_Validator.checkMoreProbes(*args)

class validator:
    __metaclass__ = ABCMeta

    def __init__(self,sysfname,idxfname,verbose):
        self.sysname=sysfname
        self.idxname=idxfname
        self.condition=''
        self.verbose=verbose
    @abstractmethod
    def nameCheck(self,NCID): pass
    @abstractmethod
    def contentCheck(self,identify=False,neglectMask=False,reffname=0,indexFilter=False): pass
    def checkMoreProbes(self,maskData): pass
#    def fullCheck(self,nc,identify,NCID,neglectMask,reffname=0,processors=1):
    def fullCheck(self,params):
        #check for existence of files
        eflag = False
        self.procs = params.processors
        manager = multiprocessing.Manager()
        msg_queue = manager.Queue()
#        self.printbuffer = printbuffer(self.verbose)
        self.printbuffer = msg_queue
        if not os.path.isfile(self.sysname):
#            printq("ERROR: I can't find your system output " + self.sysname + "! Where is it?",True)
            self.printbuffer.put("ERROR: Expected system output {}. Please check file path to make sure the system output file exists.".format(self.sysname))
            eflag = True
        if not os.path.isfile(self.idxname):
#            printq("ERROR: I can't find your index file " + self.idxname + "! Where is it?",True)
            self.printbuffer.put("ERROR: Expected index file {}. Please check file path to make sure the index file exists.".format(self.idxname))
            eflag = True
        if eflag:
            if self.verbose:
                while not self.printbuffer.empty():
                    msg = self.printbuffer.get()
                    print("="*30)
                    print(msg)
#            self.printbuffer.atomprint(print_lock)
            return 1
        
        self.init_other_variables(params)

        #option to do a namecheck
        self.doNameCheck = params.doNameCheck
        if self.doNameCheck:
            if self.nameCheck(params.ncid) == 1:
                if self.verbose:
                    while not self.printbuffer.empty():
                        msg = self.printbuffer.get()
                        print("="*30)
                        print(msg)
#                self.printbuffer.atomprint(print_lock)
                return 1

#        printq("Checking if index file is a pipe-separated csv...")
        self.printbuffer.put("Checking if index file is a pipe-separated csv...")
        idx_pieces = self.idxname.split('.')
        idx_ext = idx_pieces[-1]

        if idx_ext != 'csv':
#            printq("ERROR: Your index file should have csv as an extension! (It is separated by '|', I know...)",True)
            self.printbuffer.put("ERROR: Index file should have csv as an extension.")
            if self.verbose:
                while not self.printbuffer.empty():
                    msg = self.printbuffer.get()
                    print("="*30)
                    print(msg)
#            self.printbuffer.atomprint(print_lock)
            return 1

#        printq("Your index file appears to be a pipe-separated csv, for now. Hope it isn't separated by commas.")
        self.printbuffer.put("Index file is a pipe-separated csv.")

        self.optOut = params.optOut
        if self.contentCheck(params.identify,params.neglectMask,params.ref,params.indexFilter) == 1:
            if self.verbose:
                while not self.printbuffer.empty():
                    msg = self.printbuffer.get()
                    print("="*30)
                    print(msg)
#            self.printbuffer.atomprint(print_lock)
            return 1
        if self.verbose:
            while not self.printbuffer.empty():
                msg = self.printbuffer.get()
                print("="*30)
                print(msg)
#        self.printbuffer.atomprint(print_lock)
        return 0

    def init_other_variables(self,params):
        var_params = vars(params)
        for v in ['outputRewrite','task','ignore_eof','ignore_overlap']:
            setattr(self,v,getattr(params,v,False))

class SSD_Validator(validator):
    def nameCheck(self,NCID):
#        printq('Validating the name of the system file...')
        self.printbuffer.put('Validating the name of the system file...')

        sys_pieces = self.sysname.rsplit('.',1)
        sys_ext = sys_pieces[1]
        if sys_ext != 'csv':
            self.printbuffer.put('ERROR: System output is not a csv.')
            return 1
    
        fileExpid = sys_pieces[0].split('/')
        dirExpid = fileExpid[-2]
        fileExpid = fileExpid[-1]
        if fileExpid != dirExpid:
            self.printbuffer.put("ERROR: Please follow the naming convention. The system output should follow the naming <EXPID>/<EXPID>.csv.")
            return 1

        taskFlag = 0
        ncidFlag = 0
        teamFlag = 0
        sysPath = os.path.dirname(self.sysname)
        sysfName = os.path.basename(self.sysname)

        arrSplit = sysfName.split('_')
        if len(arrSplit) < 7:
            self.printbuffer.put("ERROR: There are not enough arguments to verify in the name.")
            return 1
        elif len(arrSplit) > 7:
            self.printbuffer.put("ERROR: The team name must not include underscores.")
            teamFlag = 1

        team = arrSplit[0]
        ncid = arrSplit[1]
        data = arrSplit[2]
        task = arrSplit[3]
        self.condition = arrSplit[4]
        sys = arrSplit[5]
        version = arrSplit[6]

        if ncid != NCID:
            self.printbuffer.put("ERROR: The NCID must be {}.".format(NCID))
            ncidFlag = 1
        if team == '':
            self.printbuffer.put("ERROR: The team name must not include underscores.")
            teamFlag = 1
        task = task.lower()
        if (task != 'manipulation'): # and (task != 'provenance') and (task != 'provenancefiltering'):
            self.printbuffer.put('ERROR: Task {} is unrecognized. The task must be \'manipulation\'.'.format(task)) #, provenance, or provenancefiltering!',True)
            taskFlag = 1
    
        if (taskFlag == 0) and (ncidFlag == 0) and (teamFlag == 0):
            self.printbuffer.put('The name of this file is valid.')
            return 0
        else:
            self.printbuffer.put('The name of the file is not valid. Please review the requirements.')
            return 1 

    def contentCheck(self,identify,neglectMask,reffname,indexFilter):
        printq('Validating the syntactic content of the system output.')
        index_dtype = {'TaskID':str,
                 'ProbeFileID':str,
                 'ProbeFileName':str,
                 'ProbeWidth':np.int64,
                 'ProbeHeight':np.int64}
        
        idxfile = pd.read_csv(self.idxname,sep='|',dtype=index_dtype,na_filter=False)
        sysfile = pd.read_csv(self.sysname,sep='|',na_filter=False)
        idxmini = 0
        self.identify = identify
        self.indexFilter = indexFilter

        pk_field = 'ProbeFileID'
        pk_field_list = ['ProbeFileID']
        min_sys_cols = 3
        targetquery = "IsTarget=='Y'"
        if self.task == 'camera':
            min_sys_cols = 6
        elif self.task == 'eventverification':
            min_sys_cols = 4
    
        scoreFlag = 0
        maskFlag = 0
        matchFlag = 0
        
        if sysfile.shape[1] < min_sys_cols:
            print("ERROR: The number of columns of the system output file must be at least {}. Please check to see if '|' is used to separate your columns.".format(min_sys_cols))
            return 1

        sysHeads = sysfile.columns.values.tolist()
        #either IsOptOut or ProbeStatus must be in the file header
        optOut = 0

        allClear = True
        if not (("IsOptOut" in sysHeads) or ("ProbeStatus" in sysHeads)):
            print("ERROR: Either 'IsOptOut' or 'ProbeStatus' must be in the column headers.")
            allClear = False
        else:
            if ("IsOptOut" in sysHeads) and ("ProbeStatus" in sysHeads):
                print("The system output has both 'IsOptOut' and 'ProbeStatus' in the column headers. It is advised for the performer not to confuse him or herself.")
            if "IsOptOut" in sysHeads:
                optOut = 1
            elif "ProbeStatus" in sysHeads:
                if not (("ProbeOptOutPixelValue" in sysHeads) or (self.task in ['manipulation-video','eventverification'])):
                    print("ERROR: The required column ProbeOptOutPixelValue is absent.")
                    return 1
                optOut = 2
        self.optOutNum = optOut

        self.testMask = self.task != 'manipulation-video'
        if not (self.checkHeads(sysHeads,optOut) and allClear):
            return 1

        #define primatives here according to task, in particular, camera and eventverification.
        if self.task == 'camera':
            targetquery = "(IsTarget=='Y') & (IsManipulation=='Y')"
            sysfile['TrainCamID'] = sysfile['TrainCamID'].astype(str)
            sysfile['ProbeCamID'] = sysfile['ProbeFileID'] + ":" + sysfile['TrainCamID']
            idxfile['ProbeCamID'] = idxfile['ProbeFileID'] + ":" + idxfile['TrainCamID']
            pk_field = 'ProbeCamID'
            pk_field_list = ['ProbeFileID','TrainCamID']
        elif self.task == 'eventverification':
            sysfile['EventName'] = sysfile['EventName'].astype(str)
            sysfile['ProbeEventID'] = sysfile['ProbeFileID'] + ":" + sysfile['EventName']
            idxfile['ProbeEventID'] = idxfile['ProbeFileID'] + ":" + idxfile['EventName']
            pk_field = 'ProbeEventID'
            pk_field_list = ['ProbeFileID','EventName']

        if reffname is not 0:
            #filter idxfile based on ProbeFileID's in reffile
            reffile = pd.read_csv(reffname,sep='|',na_filter=False)
            #set up dual-ID's for the relevant task
            if self.task == 'camera':
                reffile['ProbeCamID'] = reffile['ProbeFileID'] + ":" + reffile['TrainCamID']
            elif self.task == 'eventverification':
                reffile['ProbeEventID'] = reffile['ProbeFileID'] + ":" + reffile['EventName']
            gt_ids = reffile.query(targetquery)[pk_field].tolist()
            idxmini = idxfile.query("{}=={}".format(pk_field,gt_ids))
        #switch to video validation
        if self.condition in ["VidOnly","VidMeta"] or (self.task in ['manipulation-video','eventverification']):
            neglectMask = True
            #TODO: if OutputProbeMaskFileName in sysHeads and nonempty, throw an error?
        self.neglectMask = neglectMask

        if not (self.rowCheck(sysfile,idxfile,pk_field,pk_field_list) == 0):
            self.printbuffer.put("The contents of your file are not valid.")
            return 1
        
        #NOTE: typesetting
        sysfile['ProbeFileID'] = sysfile['ProbeFileID'].astype(str)

        try:
            sysfile['ConfidenceScore'] = sysfile['ConfidenceScore'].astype(np.float64)
        except ValueError:
            confprobes = sysfile[~sysfile['ConfidenceScore'].map(is_finite_number)][pk_field_list]
            identifier = 'probes'
            if len(pk_field_list) == 2:
                identifier = 'pairs'
            print("ERROR: The Confidence Scores for {} {} are not numeric.".format(identifier,confprobes))
            scoreFlag = 1

        if self.testMask and (self.task not in ['manipulation-video','eventverification']):
            sysfile['OutputProbeMaskFileName'] = sysfile['OutputProbeMaskFileName'].astype(str) 
        #NOTE: typesetting ends here

        #setting variables based on primary ID.
        sysProbes = sysfile[pk_field].unique()
        idxProbes = idxfile[pk_field].unique()
        iminiProbes = 0
        if idxmini is not 0:
            iminiProbes = idxmini[pk_field].unique()

        self.sysProbes = sysProbes
        self.idxProbes = idxProbes
        self.iminiProbes = iminiProbes

        for probeID in idxProbes:
            if not (probeID in sysProbes):
                self.printbuffer.put("ERROR: Expected probe {} in system file. Only found in index file.".format(probeID))
                matchFlag = 1

        self.sysfile = sysfile
        self.idxfile = idxfile
        self.sysPath = os.path.dirname(self.sysname)

        self.sysfile['maskFlag'] = 0
        self.sysfile['matchFlag'] = 0
        self.sysfile['Message'] = '' #for return print messages

        #parallelize with multiprocessing a'la localization scorer
        sysfile=self.checkProbes(sysfile,self.procs)
        maskFlag = np.sum(sysfile['maskFlag'])
        matchFlag = np.sum(sysfile['matchFlag'])

        if self.outputRewrite is None:
            output_rewrite_name = '_'.join([self.sysname[:-4],'rewrite.csv'])
        else:
            output_rewrite_name = self.outputRewrite
        if self.outputRewrite:
            sysfile = sysfile[sysHeads]
            sysfile.to_csv(os.path.join(self.sysPath,output_rewrite_name),sep="|",index=False)
        
        #final validation
        flagSum = maskFlag + scoreFlag + matchFlag
        if flagSum == 0:
            self.printbuffer.put("The contents of your file have passed validation.")
            return 0
        else:
            self.printbuffer.put("The contents of your file are not valid.")
            return 1

    def checkHeads(self,sysHeads,optOutMode):
#        truelist = ["ProbeFileID","ConfidenceScore","OutputProbeMaskFileName","IsOptOut"]
        truelist = ["ProbeFileID","ConfidenceScore","OutputProbeMaskFileName"]
        if self.task == 'manipulation-video':
            if optOutMode == 1:
                truelist = ['ProbeFileID','ConfidenceScore','IsOptOut']
            elif optOutMode == 2:
                truelist = ['ProbeFileID','ConfidenceScore','ProbeStatus','VideoFrameSegments','AudioSampleSegments','VideoFrameOptOutSegments']
        elif self.task == 'camera':
            truelist = ["ProbeFileID","TrainCamID","OutputProbeMaskFileName","ConfidenceScore","ProbeStatus","ProbeOptOutPixelValue"]
        elif self.task == 'eventverification':
            truelist = ["ProbeFileID","EventName","ConfidenceScore","ProbeStatus"]
        #check for ProbeOptOutPixelValue
        self.pixOptOut = 'ProbeOptOutPixelValue' in sysHeads

        headflag = True
        for i in xrange(len(truelist)):
            headcheck = truelist[i] in sysHeads
            if not headcheck:
                print("ERROR: The required column {} is absent.".format(truelist[i]))
                headflag = False
        return headflag

    def rowCheck(self,sysfile,idxfile,pk_field,pk_field_list):
        rowFlag = 0
        if sysfile.shape[0] != sysfile.drop_duplicates().shape[0]:
            rowlist = xrange(sysfile.shape[0])
            print(" ".join(["ERROR: Your system output contains duplicate rows for {}:".format(pk_field_list),
                  ' ,'.join(list(map(str,sysfile[pk_field][sysfile.duplicated()]))),"at row(s):",
                  ' ,'.join(list(map(str,[i for i in rowlist if sysfile.duplicated()[i]]))),"after the header. I recommended you delete these row(s)."]))
            rowFlag = 1
        
        if sysfile.shape[0] != idxfile.shape[0]:
            # +1 for header row
            print("ERROR: The number of rows in your system output ({}) does not match the number of rows in the index file ({}).".format(sysfile.shape[0]+1,idxfile.shape[0]+1))
            rowFlag = 1
        return rowFlag

    def checkProbes(self,maskData,processors):
        maxprocs = multiprocessing.cpu_count() - 2
        nrow = len(maskData)
        if (processors > nrow) and (nrow > 0):
            print("Warning: too many processors for rows in the data. Defaulting to rows in data ({}).".format(nrow))
            processors = nrow
        if processors > maxprocs:
            print("Warning: the machine does not have that many processors available. Defaulting to max ({}).".format(max(maxprocs,1)))
            processors = max(maxprocs,1)

        if processors > 1:
            chunksize = nrow//processors
            maskDataS = [[self,maskData[i:(i+chunksize)]] for i in range(0,nrow,chunksize)]
            p = multiprocessing.Pool(processes=processors)
            maskDataS = p.map(checkProbe,maskDataS)
            p.close()
            p.join()
            maskData = pd.concat(maskDataS)
        else:
            maskData = self.checkMoreProbes(maskData)
        #add all mask 'Message' entries to printBuffer
        maskData.apply(lambda x: self.printbuffer.put(x['Message']),axis=1,reduce=False)

        return maskData

    def checkMoreProbes(self,maskData):
        return maskData.apply(self.checkOneProbe,axis=1,reduce=False)

    #attach the flag to each row and send the row back
    def checkOneProbe(self,sysrow):
        pk_field = 'ProbeFileID'
        if self.task == 'camera':
            pk_field = 'ProbeCamID'
        elif self.task == 'eventverification':
            pk_field = 'ProbeEventID'
        probeFileID = sysrow[pk_field]
        sysrow['maskFlag'] = 0
        sysrow['matchFlag'] = 0
        if not ((probeFileID in self.idxProbes) or self.indexFilter):
            sysrow['Message']="ERROR: {} does not exist in the index file.".format(probeFileID)
            sysrow['matchFlag'] = 1
            return sysrow
        elif self.iminiProbes is not 0:
            if not (probeFileID in self.iminiProbes):
#                sysrow['Message']="Neglecting mask validation for probe {}.".format(probeFileID)
                return sysrow

#                printq("The contents of your file are not valid!",True)
#                return 1

        if self.pixOptOut and (self.task != 'manipulation-video'):
            oopixval = str(sysrow['ProbeOptOutPixelValue'])
            #check if ProbeOptOutPixelValue is blank or an integer if it exists in the header.
            isProbeOOdigit = True
            if oopixval != '':
                isProbeOOdigit = is_integer(oopixval)
            if not ((oopixval == '') or isProbeOOdigit):
                sysrow['Message']="ERROR: ProbeOptOutPixelValue for probe {} is {}. Please check if it is blank ('') or an integer.".format(probeFileID,oopixval)
                sysrow['matchFlag'] = 1

        #check mask validation
        if self.testMask or (self.task == 'manipulation-video'):
            #if IsOptOut or ProbeStatus is present
            #check if IsOptOut is 'Y' or 'Detection'. Likewise for ProbeStatus as relevant
            if self.optOutNum == 1:
                #throw error if not in set of allowed values
                all_statuses = ['Y','Detection','Localization','N','FailedValidation']
                if not sysrow['IsOptOut'] in all_statuses:
                    sysrow['Message'] = " ".join([sysrow['Message'],"ERROR: Probe status {} for probe {} is not recognized.".format(sysrow['IsOptOut'],sysrow['ProbeFileID'])])
                    sysrow['matchFlag'] = 1
                if sysrow['IsOptOut'] == 'FailedValidation':
                    return sysrow
            elif self.optOutNum == 2:
                all_statuses = ['Processed','NonProcessed','OptOutAll','OptOutDetection','OptOutLocalization','FailedValidation']
                if self.task == 'eventverification':
                    all_statuses = ['Processed','NonProcessed','OptOut','FailedValidation']
                if self.task == 'manipulation-video':
                    all_statuses.extend(["OptOutTemporal","OptOutSpatial"])
                if not sysrow['ProbeStatus'] in all_statuses:
                    sysrow['Message'] = " ".join([sysrow['Message'],"ERROR: Probe status {} for probe {} is not recognized.".format(sysrow['ProbeStatus'],sysrow['ProbeFileID'])])
                    sysrow['matchFlag'] = 1
                if sysrow['ProbeStatus'] == 'FailedValidation':
                    sysrow['Message'] = " ".join([sysrow['Message'],"Warning: Probe {} has failed validation due to incorrect mask dimensions, but is excused in this system output."])
                    return sysrow

            if self.task == 'manipulation-video':
                if self.optOutNum == 1:
                    return sysrow
                msgs = []
                for col in ['VideoFrameSegments','AudioSampleSegments','VideoFrameOptOutSegments']:
                    if 'FrameCount' not in list(self.idxfile):
                        probeFileName=self.idxfile[self.idxfile.ProbeFileID.isin([probeFileID])].ProbeFileName.iloc[0]
                        refroot = os.path.abspath(os.path.join(os.path.dirname(self.idxname),'..'))
                        cap = cv2.VideoCapture(os.path.join(refroot,probeFileName))
                        #NOTE: OpenCV 3+ versions will not have cv2.cv.
                        try:
                            vid_frame_feature = cv2.cv.CV_CAP_PROP_FRAME_COUNT
                        except AttributeError:
                            try:
                                vid_frame_feature = cv2.CAP_PROP_FRAME_COUNT
                            except:
                                vid_frame_feature = 7
                        maxFrame = int(cap.get(vid_frame_feature))
                    else:
                        maxFrame = self.idxfile[self.idxfile.ProbeFileID.isin([probeFileID])].FrameCount.iloc[0]
                    mymskflag,mymsg = self.vidIntervalsCheck(sysrow[col],'Frame',maxFrame,col,probeFileID) #NOTE: 'Frame' evaluation until we have time evaluations
                    sysrow['maskFlag'] = sysrow['maskFlag'] | mymskflag
                    msgs.append(mymsg)
                sysrow['Message'] = "\n".join([sysrow['Message']] + msgs)
                return sysrow
            if self.neglectMask:
                return sysrow

            probeOutputMaskFileName = sysrow['OutputProbeMaskFileName']
            if probeOutputMaskFileName in [None,'',np.nan,'nan']:
                sysrow['Message']=" ".join([sysrow['Message'],"The mask for file {} appears to be absent. Skipping it.".format(probeFileID)])
                return sysrow
            mskflag = 0
            msg = ""
            mskflag,msg = self.maskCheck(os.path.join(self.sysPath,probeOutputMaskFileName),probeFileID,self.idxfile,self.identify,pk_field)
            sysrow['Message'] = "\n".join([sysrow['Message'],msg])
            if self.outputRewrite:
                if mskflag >= 2:
                    if self.optOutNum == 1:
                        oo_col = 'IsOptOut'
                    elif self.optOutNum == 2:
                        oo_col = 'ProbeStatus'
                    sysrow[oo_col] = 'FailedValidation'
            
            sysrow['maskFlag'] = sysrow['maskFlag'] | mskflag 
        return sysrow

    def maskCheck(self,maskname,fileid,indexfile,identify,pk_field):
        #check to see if index file input image files are consistent with system output
        flag = 0
        msg=["Validating {} for file {}...".format(maskname,fileid)]
        mask_pieces = maskname.split('.')
        mask_ext = mask_pieces[-1]
        if mask_ext.lower() != 'png':
            msg.append('ERROR: Mask image {} for FileID {} is not a png. It is {}.'.format(maskname,fileid,mask_ext))
            return 1,"\n".join(msg)
        if not os.path.isfile(maskname):
            msg.append("ERROR: Expected mask image {}. Please check the name of the mask image.".format(maskname))
            return 1,"\n".join(msg)
        baseHeight = list(map(int,indexfile['ProbeHeight'][indexfile[pk_field] == fileid]))[0] 
        baseWidth = list(map(int,indexfile['ProbeWidth'][indexfile[pk_field] == fileid]))[0]
    
        #subprocess with imagemagick identify for speed
        if identify:
            dimoutput = subprocess.check_output(["identify","-format","'%f|%w|%h'",maskname]).rstrip().replace("'","").split('|')
            dims = (int(dimoutput[2]),int(dimoutput[1]))
        else:
            try:
                dims = cv2.imread(maskname,cv2.IMREAD_UNCHANGED).shape
            except:
                msg.append("ERROR: system probe mask {} cannot be read as a png.".format(maskname))
                return 1,"\n".join(msg)
    
        if identify:
            channel = subprocess.check_output(["identify","-format","%[channels]",maskname]).rstrip()
            if channel != "gray":
                msg.append("ERROR: {} is not single-channel. It is {}. Make it single-channel.".format(maskname,channel))
                flag = 1
        elif len(dims)>2:
            msg.append("ERROR: {} is not single-channel. It has {} channels. Make it single-channel.".format(maskname,dims[2]))
            flag = 1
    
        if (baseHeight != dims[0]) or (baseWidth != dims[1]):
            msg.append("ERROR: Expected dimensions {},{} for output mask {} for {} {}. Got {},{}.".format(baseHeight,baseWidth,maskname,pk_field,fileid,dims[0],dims[1]))
            #Add 2 to flag to let scorer know to reset the row
            if self.outputRewrite:
                msg.append("Rewriting probe status for {} {} to 'FailedValidation'...".format(pk_field,fileid))
                flag = flag + 2
            else: 
                flag = 1
            
        #maskImg <- readPNG(maskname) #EDIT: expensive for only getting the number of channels. Find cheaper option
        #note: No need to check for third channel. The imread option automatically reads as grayscale.
    
        if flag == 0:
            msg.append(maskname + " is valid.")
    
        return flag,"\n".join(msg)

    def vidIntervalsCheck(self,intvl,mode,max_frame_number,column_name,probeFileID):
        msg = ["Evaluationg column {} for probe {}.".format(column_name,probeFileID)]
        intervalflag = 0
        cleared_interval_list = []
        try:
            interval_list = ast.literal_eval(intvl)
            for interval in interval_list:
                if len(interval) != 2:
                    errmsg = "ERROR: The list {} is not a valid interval.".format(interval)
                    msg.append(errmsg)
                    intervalflag = 1
                    continue
                #TODO: check frame datatype for int and float respectively.
                if interval[0] > interval[1]:
                    errmsg = "ERROR: Interval {} must be formatted as a valid interval [a,b], where a <= b".foramt(interval)
                    msg.append(errmsg)
                    intervalflag = 1
                if mode=='Frame':
                    min_frame_number=1
                elif mode=='Time':
                    min_frame_number=0
                if ((interval[0] < min_frame_number) or (interval[1] > max_frame_number)) and not self.ignore_eof:
                    errmsg = "ERROR: Interval {} is out of bounds. The max interval for this video is {}.".format(interval,[min_frame_number,max_frame_number])
                    msg.append(errmsg)
                    intervalflag = 1
                elif ((interval[0] < min_frame_number) or (interval[1] > max_frame_number)) and self.ignore_eof:
                    errmsg = "Warning: Interval {} is out of bounds. The max interval for this video is {}.".format(interval,[min_frame_number,max_frame_number])
                    msg.append(errmsg)
        
                if intervalflag == 0:
                    #each of the intervals in each list of intervals must be disjoint (except for endpoints)
                    if len(cleared_interval_list) == 0:
                        cleared_interval_list.append(interval)
                        continue

                    for intvl in cleared_interval_list:
                        #ensure if a singularity that it only coincides with an endpoint
                        if interval[0] == interval[1]:
                            if (interval[0] > intvl[0]) and (interval[0] < intvl[1]):
                                errmsg = "ERROR: Singular frame {} is contained in interval {} and is not one of its endpoints.".format(interval,intvl)
                                msg.append(errmsg)
                                intervalflag = 1
                        else:
                            if ((interval[0] >= intvl[0]) and (interval[0] <= intvl[1])) or ((interval[1] <= intvl[1]) and (interval[1] >= intvl[0])):
                                #coincides with at least an endpoint
                                msgtype = "Warning" if self.ignore_overlap else "ERROR"
                                if not self.ignore_overlap:
                                    intervalflag = 1
                                errmsg = "{msgtype}: Interval {intvl1} intersects with interval {intvl2}.".format(msgtype=msgtype,intvl1=interval,intvl2=intvl)
                                msg.append(errmsg)
                    cleared_interval_list.append(interval)
        except Exception,e:
#            exc_type,exc_obj,exc_tb = sys.exc_info()
#            print("Exception {} encountered at line {}.".format(exc_type,exc_tb.tb_lineno))
            msg.append("ERROR: Interval list '{}' cannot be read as intervals for column {} for probe {}.".format(intvl,column_name,probeFileID))
            return 1,'\n'.join(msg)
        return intervalflag,'\n'.join(msg)


class DSD_Validator(validator):
    def nameCheck(self,NCID):
        self.printbuffer.put('Validating the name of the system file...')

        sys_pieces = self.sysname.rsplit('.',1)
        sys_ext = sys_pieces[1]
        if sys_ext != 'csv':
            self.printbuffer.put('ERROR: System output is not a csv.')
            return 1
    
        fileExpid = sys_pieces[0].split('/')
        dirExpid = fileExpid[-2]
        fileExpid = fileExpid[-1]
        if fileExpid != dirExpid:
            self.printbuffer.put("ERROR: Please follow the naming convention. The system output should follow the naming <EXPID>/<EXPID>.csv.")
            return 1
    
        taskFlag = 0
        ncidFlag = 0
        teamFlag = 0
        sysPath = os.path.dirname(self.sysname)
        sysfName = os.path.basename(self.sysname)

        arrSplit = sysfName.split('_')
        if len(arrSplit) < 7:
            self.printbuffer.put("ERROR: There are not enough arguments to verify in the name.")
            return 1
        elif len(arrSplit) > 7:
            self.printbuffer.put("ERROR: The team name must not include underscores.")
            teamFlag = 1

        team = arrSplit[0]
        ncid = arrSplit[1]
        data = arrSplit[2]
        task = arrSplit[3]
        self.condition = arrSplit[4]
        sys = arrSplit[5]
        version = arrSplit[6]
    
        if team == '':
            self.printbuffer.put("ERROR: The team name must not include underscores.")
            teamFlag = 1
    
        if ncid != NCID:
            self.printbuffer.put("ERROR: The NCID must be {}.".format(NCID))
            ncidFlag = 1

        task = task.lower()
        if task != 'splice':
            self.printbuffer.put('ERROR: Task {} is unrecognized. The task must be \'splice\'.'.format(task)) #, provenance, or provenancefiltering!',True)
            taskFlag = 1
    
        if (taskFlag == 0) and (ncidFlag == 0) and (teamFlag == 0):
            self.printbuffer.put('The name of this file is valid.')
            return 0
        else:
            self.printbuffer.put('The name of the file is not valid. Please review the requirements.')
            return 1 

    #redesigned pipeline
    def contentCheck(self,identify,neglectMask,reffname,indexFilter):
        self.printbuffer.put('Validating the syntactic content of the system output.')
        #read csv line by line
        dupFlag = 0
        xrowFlag = 0
        scoreFlag = 0
        colFlag = 0
        maskFlag = 0
        keyFlag = 0

        self.indexFilter = indexFilter
        i_len = 0
        s_len = 0 #number of elements in iterator
        s_headnames = ''
        s_heads = {}
        i_heads = {}
        s_lines = []
        with open(self.idxname) as idxfile:
            i_len = sum(1 for l in idxfile)

        with open(self.sysname) as sysfile:
            s_len = sum(1 for l in sysfile)
            #check for row number matchup with index file.

            if i_len != s_len:
#                printq("ERROR: The number of rows in your system output does not match the number of rows in the index file.",True)
                self.printbuffer.put("ERROR: The number of rows in your system output ({}) does not match the number of rows in the index file ({}).".format(s_len,i_len))
                xrowFlag = 1

        r_files = {}
        if reffname is not 0:
            #filter rows based on ProbeFileID's in reffile
            with open(reffname) as reffile:
                r_heads = {}
                for idx,l in enumerate(reffile):
                    #print "Process index " + str(idx) + " " + l
                    if idx == 0:
                        r_headnames = l.rstrip().split('|')
                        for i,h in enumerate(r_headnames):
                            r_heads[h] = i
                    else:
                        r_data = l.rstrip().replace("\"","").split('|')
                        r_files[":".join([r_data[r_heads['ProbeFileID']],r_data[r_heads['DonorFileID']]])] = r_data[r_heads['IsTarget']]
            
#        with open(self.idxname) as idxfile:
#            for i,l in enumerate(idxfile):
#                if i==0:
#                    i_headnames = l.split('|')
#                    for i,h in enumerate(i_headnames):
#                        i_heads[h.replace('\n','')] = i
#                else: break

        idxPath = os.path.dirname(self.idxname)
        ind = {}
        with open(self.idxname) as idxfile:
            for idx,l in enumerate(idxfile):
                #print "Process index " + str(idx) + " " + l
                if idx == 0:
                    i_headnames = l.rstrip().split('|')
                    for i,h in enumerate(i_headnames):
                        i_heads[h] = i
                else:
                    i_data = l.rstrip().replace("\"","").split('|')
                    #print i_data[i_heads['ProbeFileID']] + ":" + i_data[i_heads['DonorFileID']]
                    ind[":".join([i_data[i_heads['ProbeFileID']],i_data[i_heads['DonorFileID']]])] = i_data

        self.printbuffer.put("Index read")

        #output rewrite file
        if self.outputRewrite:
            if self.outputRewrite is None:
                output_rewrite_name = '_'.join([self.sysname[:-4],'rewrite.csv'])
            else:
                output_rewrite_name = self.outputRewrite
            sysPath = os.path.dirname(self.sysname)
            rewrite_file = open(os.path.join(sysPath,output_rewrite_name),'w+')

        sysPath = os.path.dirname(self.sysname)
        testMask = True
        optOut = 0
        with open(self.sysname) as sysfile:
            for idx,l in enumerate(sysfile):
                self.printbuffer.put("Process {} ".format(idx) + l)

                if idx == 0:
                    #parse headers
                    s_headnames = l.rstrip().split('|')
                    #header checking
                    if len(s_headnames) < 5:
                        #check number of headers
                        self.printbuffer.put("ERROR: The number of columns of the system output file must be at least 6. Please check to see if '|' is used to separate your columns.")
                        return 1
                    allClear = True
                    truelist = ["ProbeFileID","DonorFileID","ConfidenceScore","OutputProbeMaskFileName","OutputDonorMaskFileName"]
#                    if self.condition in ["ImgOnly","ImgMeta"]:
#                        truelist = ["ProbeFileID","DonorFileID","ConfidenceScore"]
#                        testMask = True
                    
                    if not (("IsOptOut" in s_headnames) or (("ProbeStatus" in s_headnames) and ("DonorStatus" in s_headnames))):
                        self.printbuffer.put("ERROR: The column 'IsOptOut', or 'ProbeStatus' and 'DonorStatus', must be in the column headers.")
                    else:
                        if ("IsOptOut" in s_headnames) and ("ProbeStatus" in s_headnames) and ("DonorStatus" in s_headnames):
                            self.printbuffer.put("The system output has both 'IsOptOut' and 'ProbeStatus' in the column headers. It is advised for the performer not to confuse him or herself.")

#                        if self.optOut:
                        if "IsOptOut" in s_headnames:
                            optOut = 1
                        elif ("ProbeStatus" in s_headnames) and ("DonorStatus" in s_headnames):
                            optOut = 2

                    self.pixOptOut = False
                    if ("ProbeOptOutPixelValue" in s_headnames) and ("DonorOptOutPixelValue" in s_headnames):
                        self.pixOptOut = True
                    elif (("ProbeOptOutPixelValue" in s_headnames) and not ("DonorOptOutPixelValue" in s_headnames)) or ((not ("ProbeOptOutPixelValue" in s_headnames)) and ("DonorOptOutPixelValue" in s_headnames)):
                        self.printbuffer.put("ERROR: Both ProbeOptOutPixelValue and DonorOptOutPixelValue are required for the splice task.")
                        allClear = False
                             
                    for th in truelist:
                        headcheck = th in s_headnames
                        allClear = allClear and headcheck
                        if not headcheck:
                            self.printbuffer.put("ERROR: The required column {} is absent.".format(th))
                    if not allClear:
                        return 1
            
#                    if ("OutputProbeMaskFileName" in s_headnames) and ("OutputDonorMaskFileName" in s_headnames):
#                        testMask = True

                    for i,h in enumerate(s_headnames):
                        #drop into dictionary for indexing
                        s_heads[h] = i

                    if self.outputRewrite:
                        rewrite_file.write(l)
                    continue

                #for non-headers
                l_content = l.rstrip().replace("\"","").split('|')
                probeID = l_content[s_heads['ProbeFileID']]
                donorID = l_content[s_heads['DonorFileID']]
                key = ":".join([probeID,donorID])

                #try catch the key lookup
                keyPresent=True
                try:
                    indRec = ind[key]
                except KeyError:
                    keyPresent=False
                    if not indexFilter:
                        self.printbuffer.put("ERROR: The pair ({},{}) does not exist in the index file.".format(probeID,donorID))
                        keyFlag = 1
                    continue

                if reffname is not 0:
                    if (r_files[key] == 'N') or not keyPresent:
#                        self.printbuffer.put("Skipping the pair ({},{}).".format(probeID,donorID))
                        continue

                #confidence score checking
                if not is_finite_number(l_content[s_heads['ConfidenceScore']]):
                    self.printbuffer.put("ERROR: The Confidence Score for probe-donor pair ({}) is not a real number.".format(key.replace(":",",")))
                    scoreFlag = 1

                if self.pixOptOut:
                    oopixvalp = str(l_content[s_heads['ProbeOptOutPixelValue']])
                    oopixvald = str(l_content[s_heads['DonorOptOutPixelValue']])
                    isProbeOOdigit = True
                    isDonorOOdigit = True
                    if (oopixvalp != ''):
                        isProbeOOdigit = is_integer(oopixvalp)
                    if (oopixvald != ''):
                        isDonorOOdigit = is_integer(oopixvald)
                    
                    if (not ((oopixvalp == '') or isProbeOOdigit)) or (not ((oopixvald == '') or isDonorOOdigit)):
                        self.printbuffer.put("ERROR: ProbeOptOutPixelValue for probe-donor pair ({}) is {} and DonorOptOutPixelValue is {}. Please check if either is not blank ('') or an integer.".format(key.replace(":",","),oopixvalp,oopixvald))
                        scoreFlag = 1

                if testMask:
                    if neglectMask:
                        continue
                    probeOutputMaskFileName = l_content[s_heads['OutputProbeMaskFileName']]
                    donorOutputMaskFileName = l_content[s_heads['OutputDonorMaskFileName']]
                    probeOutputMaskFileName = os.path.join(sysPath,probeOutputMaskFileName) if probeOutputMaskFileName not in ['',None,np.nan] else ''
                    donorOutputMaskFileName = os.path.join(sysPath,donorOutputMaskFileName) if donorOutputMaskFileName not in ['',None,np.nan] else ''

                    optOutOption = 0
                    if optOut == 1:
                        all_statuses = ['Y','Detection','Localization','N','FailedValidation']
                        if not l_content[s_heads['IsOptOut']] in all_statuses:
                            self.printbuffer.put("ERROR: Probe status {} for probe {} is not recognized.".format(l_content[s_heads['IsOptOut']],probeID))
                            colFlag = 1
                        if l_content[s_heads['IsOptOut']] == 'FailedValidation':
                            continue
#                        if l_content[s_heads['IsOptOut']] in ['Y','Localization']:
#                            continue
                    elif optOut == 2:
                        #if ProbeStatus is opting out, opt out of Probe; likewise for Donor.
                        #split into donor and probe checking with optOutOption
                        all_statuses = ['Processed','NonProcessed','OptOutAll','OptOutDetection','OptOutLocalization','FailedValidation']
                        if not l_content[s_heads['ProbeStatus']] in all_statuses:
                            self.printbuffer.put("ERROR: Probe status {} for probe {} is not recognized.".format(l_content[s_heads['ProbeStatus']],probeID))
                            colFlag = 1
                        if not l_content[s_heads['DonorStatus']] in ['Processed','NonProcessed','OptOutLocalization','FailedValidation']:
                            self.printbuffer.put("ERROR: Donor status {} for donor {} is not recognized.".format(l_content[s_heads['DonorStatus']],donorID))
                            colFlag = 1

                        if l_content[s_heads['ProbeStatus']] == 'FailedValidation':
                            optOutOption = optOutOption + 1
                        if l_content[s_heads['DonorStatus']] == 'FailedValidation':
                            optOutOption = optOutOption + 2
#                        if l_content[s_heads['ProbeStatus']] in ["OptOutAll","OptOutLocalization"]:
#                            optOutOption = optOutOption + 1
#                        if l_content[s_heads['DonorStatus']] == "OptOutLocalization":
#                            optOutOption = optOutOption + 2

                    #skip in the maskCheck operation if neither is present
#                    if (probeOutputMaskFileName == '') and (donorOutputMaskFileName == ''):
#                        self.printbuffer.put("Both masks for the pair ({},{}) appear to be absent. Skipping this pair.".format(probeID,donorID))
#                        continue

                    probeWidth = int(indRec[i_heads['ProbeWidth']])
                    probeHeight = int(indRec[i_heads['ProbeHeight']])
                    donorWidth = int(indRec[i_heads['DonorWidth']])
                    donorHeight = int(indRec[i_heads['DonorHeight']])

                    
                    maskCheckFlag = self.maskCheckPD(probeOutputMaskFileName,donorOutputMaskFileName,probeID,donorID,probeWidth,probeHeight,donorWidth,donorHeight,idx,identify,optOutOption)
                    maskFlag = maskFlag | maskCheckFlag

                    if self.outputRewrite:
                        if optOut == 1:
                            if maskCheckFlag >= 2:
                                l_content[s_heads['IsOptOut']] = 'FailedValidation'
                        if optOut == 2:
                            if maskCheckFlag >= 4:
                                l_content[s_heads['DonorStatus']] = 'FailedValidation'
                                maskCheckFlag = maskCheckFlag - 4
                            if maskCheckFlag >= 2:
                                l_content[s_heads['ProbeStatus']] = 'FailedValidation'

                        newline = []
                        for h in s_headnames:
                            newline.append(l_content[s_heads[h]])
                        
                        rewrite_file.write("{}\n".format('|'.join(newline)))

#                     if l not in s_lines:
#                         #check for duplicate rows. Append to empty list at every opportunity for now
#                         s_lines.append(l)
#                     else:
#                         printq("ERROR: Row {} with ProbeFileID and DonorFileID pair ({},{}) is a duplicate. Please delete it.".format(idx,probeID,donorID),True)
#                         dupFlag = 1

#                     if dupFlag == 0:
#                         #only point in checking masks is if index file doesn't have duplicates in the first place

#                         os.system("grep -i {} {} > tmp.txt".format(probeID,self.idxname)) #grep to temporary file
#                         os.system("grep -i {} tmp.txt > row.txt".format(donorID))
#                         qlen = 0

#                         with open("row.txt") as row:
#                             qlen = sum(1 for m in row)
#                         if qlen > 0:
#                             #if it yields at least one row, look for masks.
#                             m_content = ''
#                             probeOutputMaskFileName = ''
#                             donorOutputMaskFileName = ''
#                             probeWidth = 0
#                             probeHeight = 0
#                             donorWidth = 0
#                             donorHeight = 0
#                             with open("row.txt") as row:
#                                 for m in row:
#                                     m_content = m.split('|')
#                                     probeOutputMaskFileName = l_content[s_heads['OutputProbeMaskFileName']].replace("\"","").replace("\n","").replace("\r","")
#                                     donorOutputMaskFileName = l_content[s_heads['OutputDonorMaskFileName']].replace("\"","").replace("\n","").replace("\r","")
#                                     probeWidth = int(m_content[i_heads['ProbeWidth']].replace("\"",""))
#                                     probeHeight = int(m_content[i_heads['ProbeHeight']].replace("\"",""))
#                                     donorWidth = int(m_content[i_heads['DonorWidth']].replace("\"",""))
#                                     donorHeight = int(m_content[i_heads['DonorHeight']].replace("\"",""))
#                             if (probeOutputMaskFileName in [None,'',np.nan,'nan']) or (donorOutputMaskFileName in [None,'',np.nan,'nan']):
#                                 printq("At least one mask for the pair ({},{}) appears to be absent. Skipping this pair.".format(probeOutputMaskFileName,donorOutputMaskFileName))
#                                 continue
#                             maskFlag = maskFlag | maskCheck2(os.path.join(sysPath,probeOutputMaskFileName),os.path.join(sysPath,donorOutputMaskFileName),probeID,donorID,probeWidth,probeHeight,donorWidth,donorHeight,idx)
#                         else:
#                             #if no row, no match and throw error
#                             printq("ERROR: The pair ({},{}) does not exist in the index file.",format(probeID,donorID),True)
#                             printq("The contents of your file are not valid!",True)
#                             return 1

        #final validation
        flagSum = maskFlag + dupFlag + xrowFlag + scoreFlag + colFlag + keyFlag
        if self.outputRewrite:
            rewrite_file.close()

        if flagSum == 0:
            self.printbuffer.put("The contents of your file have passed validation.")
        else:
            self.printbuffer.put("The contents of your file are not valid!")
            return 1

    def maskCheckPD(self,pmaskname,dmaskname,probeid,donorid,pbaseWidth,pbaseHeight,dbaseWidth,dbaseHeight,rownum,identify,optOutOption=0):
        #check to see if index file input image files are consistent with system output
        self.printbuffer.put("Validating probe and donor mask pair({},{}) for ({},{}) pair at row {}...".format(pmaskname,dmaskname,probeid,donorid,rownum))
        checkThisProbe = True
        checkThisDonor = True
        if (optOutOption % 2 == 1) or (pmaskname == ''):
            checkThisProbe = False
        if (optOutOption // 2 == 1) or (dmaskname == ''):
            checkThisDonor = False
  
        if not (checkThisProbe or checkThisDonor):
            self.printbuffer.put("Both system masks for the pair ({},{}) appear to be absent. Skipping this pair.".format(probeid,donorid))
            return 0
 
        flag = 0
        #check to see if index file input image files are consistent with system output
        if checkThisProbe:
            flag = flag | self.maskCheck('Probe',pmaskname,probeid,donorid,pbaseWidth,pbaseHeight,rownum,identify)

        if checkThisDonor:
            flag = flag | self.maskCheck('Donor',dmaskname,probeid,donorid,dbaseWidth,dbaseHeight,rownum,identify)
        
        if flag == 0:
            msgpfx = "masks"
            maskmsg = "{} and {}".format(pmaskname,dmaskname)
            msgsfx = "are"
            if not (checkThisProbe and checkThisDonor):
                msgpfx = "mask"
                if checkThisProbe:
                    msgpfx = "probe %s" % msgpfx
                    maskmsg = pmaskname
                elif checkThisDonor:
                    msgpfx = "donor %s" % msgpfx
                    maskmsg = dmaskname
                msgsfx = "is"
                
            self.printbuffer.put("Your {} {} {} valid.".format(msgpfx,maskmsg,msgsfx))
        return flag

    def maskCheck(self,mode,maskname,probeid,donorid,probeWidth,probeHeight,rownum,identify):
        mask_pieces,mask_ext = os.path.splitext(maskname)
        pngflag = False 
        eflag = False
        if mask_ext not in ['.png','.PNG']:
            self.printbuffer.put('ERROR: {} mask image {} for pair ({},{}) at row {} is not a png. It is a {}.'.format(mode,maskname,probeid,donorid,rownum,mask_ext))
            pngflag = True
    
        #check to see if png files exist before doing anything with them.
        if not os.path.isfile(maskname):
            self.printbuffer.put("ERROR: Expected mask image {}, but did not find it. Please check the name of the mask image.".format(maskname))
            eflag = True

        if eflag or pngflag:
            return 1

        flag = 0
        #subprocess with imagemagick identify for speed
        if identify:
            dimoutput = subprocess.check_output(["identify","-format","'%f|%w|%h'",maskname]).rstrip().replace("'","").split('|')
            dims = (int(dimoutput[2]),int(dimoutput[1]))
        else:
            try:
                dims = cv2.imread(maskname,cv2.IMREAD_UNCHANGED).shape
            except:
                self.printbuffer.put("ERROR: system probe mask {} cannot be read as a png.".format(maskname))
                return 1
    
        if (probeHeight != dims[0]) or (probeWidth != dims[1]):
            self.printbuffer.put("ERROR: Expected dimensions {},{} for output donor mask {} for probe-donor pair ({}.{}). Got {},{}.".format(probeHeight,probeWidth,maskname,probeid,donorid,dims[0],dims[1]))
            #Add 2 to flag to let scorer know to reset the row
            if self.outputRewrite:
                self.printbuffer.put("Rewriting probe status for {}FileID {} for pair ({},{}) to 'FailedValidation'...".format(mode,probeid,probeid,donorid))
                flag |= 2 if mode.lower() == 'probe' else 4
            else:
                flag |= 1
    
        if identify:
            channel = subprocess.check_output(["identify","-format","%[channels]",maskname]).rstrip()
            if channel != "gray":
                self.printbuffer.put("ERROR: {} is not single-channel. It is {}. Make it single-channel.".format(maskname,channel))
                flag |= 1
        elif len(dims)>2:
            self.printbuffer.put("ERROR: {} is not single-channel. It has {} channels. Make it single-channel.".format(maskname,dims[2]))
            flag |= 1
        return flag

    def maskCheck_0(self,pmaskname,dmaskname,probeid,donorid,indexfile,rownum):
        #check to see if index file input image files are consistent with system output
        flag = 0
        self.printbuffer.put("Validating probe and donor mask pair({},{}) for ({},{}) pair at row {}...".format(pmaskname,dmaskname,probeid,donorid,rownum))
    
        #check to see if index file input image files are consistent with system output
        pmask_pieces = pmaskname.split('.')
        pmask_ext = pmask_pieces[-1]
        if pmask_ext != 'png':
            self.printbuffer.put('ERROR: Probe mask image {} for pair ({},{}) at row {} is not a png. Make it into a png!'.format(pmaskname,probeid,donorid,rownum))
            return 1
    
        dmask_pieces = dmaskname.split('.')
        dmask_ext = dmask_pieces[-1]
        if dmask_ext != 'png':
            self.printbuffer.put('ERROR: Donor mask image {} for pair ({},{}) at row {} is not a png. Make it into a png!'.format(dmaskname,probeid,donorid,rownum))
            return 1
    
        #check to see if png files exist before doing anything with them.
        eflag = False
        if not os.path.isfile(pmaskname):
            self.printbuffer.put("ERROR: {} does not exist! Did you name it wrong?".format(pmaskname))
            eflag = True
        if not os.path.isfile(dmaskname):
            self.printbuffer.put("ERROR: {} does not exist! Did you name it wrong?".format(dmaskname))
            eflag = True
        if eflag:
            return 1
    
        pbaseHeight = list(map(int,indexfile['ProbeHeight'][(indexfile['ProbeFileID'] == probeid) & (indexfile['DonorFileID'] == donorid)]))[0] 
        pbaseWidth = list(map(int,indexfile['ProbeWidth'][(indexfile['ProbeFileID'] == probeid) & (indexfile['DonorFileID'] == donorid)]))[0]
        pdims = cv2.imread(pmaskname,cv2.IMREAD_UNCHANGED).shape
    
        if len(pdims)>2:
            self.printbuffer.put("ERROR: {} is not single-channel. It has {} channels. Make it single-channel.".format(pmaskname,pdims[2]))
            flag = 1
    
        if (pbaseHeight != pdims[0]) or (pbaseWidth != pdims[1]):
            self.printbuffer.put("ERROR: Expected dimensions {},{} for output donor mask {} for probe-donor pair ({}.{}). Got {},{}.".format(pbaseHeight,pbaseWidth,pmaskname,probeid,donorid,pdims[0],pdims[1]))
            flag = 1
         
        dbaseHeight = list(map(int,indexfile['DonorHeight'][(indexfile['ProbeFileID'] == probeid) & (indexfile['DonorFileID'] == donorid)]))[0] 
        dbaseWidth = list(indexfile['DonorWidth'][(indexfile['ProbeFileID'] == probeid) & (indexfile['DonorFileID'] == donorid)])[0]
        ddims = cv2.imread(dmaskname,cv2.IMREAD_UNCHANGED).shape
    
        if len(ddims)>2:
            self.printbuffer.put("ERROR: {} is not single-channel. It has {} channels. Make it single-channel.".format(dmaskname,ddims[2]))
            flag = 1
    
        if (dbaseHeight != ddims[0]) or (dbaseWidth != ddims[1]):
            self.printbuffer.put("ERROR: Expected dimensions {},{} for output donor mask {} for probe-donor pair ({}.{}). Got {},{}.".format(dbaseHeight,dbaseWidth,dmaskname,probeid,donorid,ddims[0],ddims[1]))
            flag = 1
     
        if flag == 0:
            self.printbuffer.put("Your masks {} and {} are valid.".format(pmaskname,dmaskname))
        return flag

class validation_params:
    """
    Description: Stores list of parameters for validation.
    """
    def __init__(self,**kwds):
        self.__dict__.update(kwds)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Validate the file and data format of the Single-Source Detection (SSD) or Double-Source Detection (DSD) files.')
    parser.add_argument('-x','--inIndex',type=str,default=None,\
        help='required index file',metavar='character')
    parser.add_argument('-s','--inSys',type=str,default=None,\
        help='required system output file',metavar='character')
    parser.add_argument('-r','--inRef',type=str,default=0,\
        help='optional reference file for filtration',metavar='character')
    #TODO: deprecate valtype in the future
    parser.add_argument('-vt','--valtype',type=str,default=None,\
        help='required validator type. Pick one of SSD, DSD, or SSD-video.',metavar='character')
    parser.add_argument('-nc','--nameCheck',action="store_true",\
        help='Check the format of the name of the file in question to make sure it matches up with the evaluation plan.')
    parser.add_argument('-id','--identify',action="store_true",\
        help='use ImageMagick\'s identify to get dimensions of mask. OpenCV reading is used by default.')
    parser.add_argument('--optOut',action='store_true',    help="Evaluate algorithm performance on a select number of trials determined by the performer via values in the ProbeStatus column.")
    parser.add_argument('-v','--verbose',type=int,default=None,\
        help='Control print output. Select 1 to print all non-error print output and 0 to suppress all printed output (bar argument-parsing errors).',metavar='0 or 1')
    parser.add_argument('-p','--processors',type=int,default=1,\
        help='The number of processors to use for validation. Choosing too many processors will cause the program to forcibly default to a smaller number. [default=1].',metavar='positive integer')
    parser.add_argument('-xF','--indexFilter',action='store_true',
        help="Filter validation only to files that are present in the index file. This option permits validation on a subset of the dataset by modifying the index file.")
    parser.add_argument('-nm','--neglectMask',action="store_true",\
        help="neglect mask dimensionality validation.")
    parser.add_argument('--ncid',type=str,default="NC17",\
        help="the NCID to validate against.")
    parser.add_argument('--ignore_eof',action='store_true',
        help="Ignore EOF of video if performer's frames go out of bounds. Has no effect on image validation.")
    parser.add_argument('--ignore_overlap',action='store_true',
        help="Ignore overlap of video frames if performer's frames go out of bounds.")
    parser.add_argument('--output_revised_system',type=str,default=None,
        help="Set probe status for images that fail dimensionality validation to 'FailedValidation' and output the new CSV to a specified file [e.g. 'my_revised_system.csv']. Submissions that only have 'FailedValidation' will be skipped in image localization scoring. [default=None]")

    if len(sys.argv) < 2:
        parser.print_help()
        exit(0)

    args = parser.parse_args()
    verbose = args.verbose
    #TODO: remove later for printout that is more amenable to parallelization
    if verbose==1:
        def printq(mystring,iserr=False):
            print(mystring)
    elif verbose==0:
        printq = lambda *x : None
    else:
        verbose = 1
#        else:
#            def printq(mystring,iserr=False):
#                if iserr:
#                    print(mystring)

    if args.identify:
        try:
            subprocess.check_output(["identify"])
        except:
            print("ImageMagick does not appear to be installed or in working order. Please reinstall. Rerun without -id.")
            exit(1)

    #valtype to task
    valtype_to_task = {'SSD':'manipulation-image',
                       'SSD-video':'manipulation-video',
                       'SSD-event':'eventverification',
                       'SSD-camera':'camera',
                       'DSD':'splice'}
    try:
        task = valtype_to_task[args.valtype]
    except:
        print("ERROR: Expected one of {} for validation type. Got {}.".format(valtype_to_task.keys(),args.valtype))
        exit(1)

    myval_params = validation_params(ncid=args.ncid,task=task,outputRewrite=args.output_revised_system,doNameCheck=args.nameCheck,optOut=args.optOut,identify=args.identify,neglectMask=args.neglectMask,indexFilter=args.indexFilter,ref=args.inRef,processors=args.processors,ignore_eof=args.ignore_eof,ignore_overlap=args.ignore_overlap)
    if args.valtype in ['SSD','SSD-video','SSD-event','SSD-camera']:
        ssd_validation = SSD_Validator(args.inSys,args.inIndex,verbose)
        exit(ssd_validation.fullCheck(myval_params))
#         exit(ssd_validation.fullCheck(args.nameCheck,args.identify,args.ncid,args.neglectMask,args.inRef,args.processors))
    elif args.valtype == 'DSD':
        dsd_validation = DSD_Validator(args.inSys,args.inIndex,verbose)
        exit(dsd_validation.fullCheck(myval_params))
#         exit(dsd_validation.fullCheck(args.nameCheck,args.identify,args.ncid,args.neglectMask,args.inRef,args.processors))
    else:
        print("Validation type must be one of: {}.".format(valtype_to_task.keys()))
        exit(1)
