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
from abc import ABCMeta, abstractmethod
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../lib')
sys.path.append(lib_path)
from printbuffer import printbuffer

print_lock = multiprocessing.Lock() #for printout

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
    def contentCheck(self,identify=False,neglectMask=False,reffname=0): pass
    def checkMoreProbes(self,maskData): pass
    def fullCheck(self,nc,identify,NCID,neglectMask,reffname=0,processors=1):
        #check for existence of files
        eflag = False
        self.procs = processors
        self.printbuffer = printbuffer(self.verbose)
        if not os.path.isfile(self.sysname):
#            printq("ERROR: I can't find your system output " + self.sysname + "! Where is it?",True)
            self.printbuffer.append("ERROR: I can't find your system output {}! Where is it?".format(self.sysname))
            eflag = True
        if not os.path.isfile(self.idxname):
#            printq("ERROR: I can't find your index file " + self.idxname + "! Where is it?",True)
            self.printbuffer.append("ERROR: I can't find your index file {}! Where is it?".format(self.idxname))
            eflag = True
        if eflag:
            self.printbuffer.atomprint(print_lock)
            return 1

        #option to do a namecheck
        if nc:
            if self.nameCheck(NCID) == 1:
                self.printbuffer.atomprint(print_lock)
                return 1

#        printq("Checking if index file is a pipe-separated csv...")
        self.printbuffer.append("Checking if index file is a pipe-separated csv...")
        idx_pieces = self.idxname.split('.')
        idx_ext = idx_pieces[-1]

        if idx_ext != 'csv':
#            printq("ERROR: Your index file should have csv as an extension! (It is separated by '|', I know...)",True)
            self.printbuffer.append("ERROR: Your index file should have csv as an extension! (It is separated by '|', I know...)")
            self.printbuffer.atomprint(print_lock)
            return 1

#        printq("Your index file appears to be a pipe-separated csv, for now. Hope it isn't separated by commas.")
        self.printbuffer.append("Your index file appears to be a pipe-separated csv, for now. Hope it isn't separated by commas.")

        if self.contentCheck(identify,neglectMask,reffname) == 1:
            self.printbuffer.atomprint(print_lock)
            return 1
        self.printbuffer.atomprint(print_lock)
        return 0


class SSD_Validator(validator):
    def nameCheck(self,NCID):
#        printq('Validating the name of the system file...')
        self.printbuffer.append('Validating the name of the system file...')

        sys_pieces = self.sysname.rsplit('.',1)
        sys_ext = sys_pieces[1]
        if sys_ext != 'csv':
            self.printbuffer.append('ERROR: Your system output is not a csv!')
            return 1
    
        fileExpid = sys_pieces[0].split('/')
        dirExpid = fileExpid[-2]
        fileExpid = fileExpid[-1]
        if fileExpid != dirExpid:
            self.printbuffer.append("ERROR: Please follow the naming convention. The system output should follow the naming <EXPID>/<EXPID>.csv.")
            return 1
    
        taskFlag = 0
        ncidFlag = 0
        teamFlag = 0
        sysPath = os.path.dirname(self.sysname)
        sysfName = os.path.basename(self.sysname)

        arrSplit = sysfName.split('_')
        if len(arrSplit) < 7:
            self.printbuffer.append("ERROR: There are not enough arguments to verify in the name.")
            return 1
        elif len(arrSplit) > 7:
            self.printbuffer.append("ERROR: The team name must not include underscores.")
            teamFlag = 1

        team = arrSplit[0]
        ncid = arrSplit[1]
        data = arrSplit[2]
        task = arrSplit[3]
        self.condition = arrSplit[4]
        sys = arrSplit[5]
        version = arrSplit[6]

        if ncid != NCID:
            self.printbuffer.append("ERROR: The NCID must be {}.".format(NCID))
            ncidFlag = 1
        if team == '':
            self.printbuffer.append("ERROR: The team name must not include underscores.")
            teamFlag = 1
        task = task.lower()
        if (task != 'manipulation'): # and (task != 'provenance') and (task != 'provenancefiltering'):
            self.printbuffer.append('ERROR: What kind of task is {}? It should be manipulation!'.format(task)) #, provenance, or provenancefiltering!',True)
            taskFlag = 1
    
        if (taskFlag == 0) and (ncidFlag == 0) and (teamFlag == 0):
            self.printbuffer.append('The name of this file is valid!')
            return 0
        else:
            self.printbuffer.append('The name of the file is not valid. Please review the requirements.')
            return 1 

    def contentCheck(self,identify,neglectMask,reffname):
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

        if reffname is not 0:
            #filter idxfile based on ProbeFileID's in reffile
            reffile = pd.read_csv(reffname,sep='|',na_filter=False)
            gt_ids = reffile.query("IsTarget=='Y'")['ProbeFileID'].tolist()
            idxmini = idxfile.query("ProbeFileID=={}".format(gt_ids))
    
        dupFlag = 0
        xrowFlag = 0
        scoreFlag = 0
        maskFlag = 0
        matchFlag = 0
        
        if sysfile.shape[1] < 3:
            self.printbuffer.append("ERROR: The number of columns of the system output file must be at least 3. Are you using '|' to separate your columns?")
            return 1

        sysHeads = list(sysfile.columns)
        allClear = True
#        truelist = ["ProbeFileID","ConfidenceScore","OutputProbeMaskFileName","IsOptOut"]
        truelist = ["ProbeFileID","ConfidenceScore"]
        testMask = False
        if self.condition in ["ImgOnly","ImgMeta"]:
            truelist.append("OutputProbeMaskFileName")
            testMask = True 
        self.testMask = testMask

        #either IsOptOut or ProbeStatus must be in the file header
        optOut = 0
        if not (("IsOptOut" in sysHeads) or ("ProbeStatus" in sysHeads)):
            self.printbuffer.append("ERROR: Either 'IsOptOut' or 'ProbeStatus' must be in the column headers.")
            allClear = False
        else:
            if ("IsOptOut" in sysHeads) and ("ProbeStatus" in sysHeads):
                self.printbuffer.append("The system output has both 'IsOptOut' and 'ProbeStatus' in the column headers. It is advised for the performer not to confuse him or herself.")

            if "IsOptOut" in sysHeads:
                optOut = 1
            elif "ProbeStatus" in sysHeads:
                optOut = 2
        self.optOut = optOut

        #check for ProbeOptOutPixelValue
        self.pixOptOut = False
        if 'ProbeOptOutPixelValue' in sysHeads:
            self.pixOptOut = True

        for i in xrange(len(truelist)):
            headcheck = truelist[i] in sysHeads
            allClear = allClear and headcheck
            if not headcheck:
#                headlist = []
#                properhl = []
#                for i in range(0,len(truelist)):
#                    if sysHeads[i] != truelist[i]:
#                        headlist.append(sysHeads[i])
#                        properhl.append(truelist[i]) 
#                printq("ERROR: Your header(s) " + ', '.join(headlist) + " should be " + ', '.join(properhl) + " respectively.",True)
                self.printbuffer.append("ERROR: The required column {} is absent.".format(truelist[i]))

        if not allClear:
            return 1

#        if "IsOptOut" in sysHeads:
#            optOut=True
#        self.optOut = optOut

        if self.condition in ["VidOnly","VidMeta"]:
            neglectMask = True
        self.neglectMask = neglectMask

        if sysfile.shape[0] != sysfile.drop_duplicates().shape[0]:
            rowlist = xrange(sysfile.shape[0])
            self.printbuffer.append(" ".join(["ERROR: Your system output contains duplicate rows for ProbeFileID's:",
                    ' ,'.join(list(map(str,sysfile['ProbeFileID'][sysfile.duplicated()]))),"at row(s):",
                    ' ,'.join(list(map(str,[i for i in rowlist if sysfile.duplicated()[i]]))),"after the header. I recommended you delete these row(s)."]))
            dupFlag = 1
        
        if sysfile.shape[0] != idxfile.shape[0]:
            self.printbuffer.append("ERROR: The number of rows in your system output ({}) does not match the number of rows in the index file ({}).".format(sysfile.shape[0]+1,idxfile.shape[0]+1))
            xrowFlag = 1
        
        if not ((dupFlag == 0) and (xrowFlag == 0)):
            self.printbuffer.append("The contents of your file are not valid!")
            return 1
        
        sysfile['ProbeFileID'] = sysfile['ProbeFileID'].astype(str)

        try:
            sysfile['ConfidenceScore'] = sysfile['ConfidenceScore'].astype(np.float64)
        except ValueError:
            confprobes = sysfile[~sysfile['ConfidenceScore'].map(is_finite_number)]['ProbeFileID']
            self.printbuffer.append("ERROR: Your Confidence Scores for probes {} are not numeric.".format(confprobes))
            scoreFlag = 1

        if testMask:
            sysfile['OutputProbeMaskFileName'] = sysfile['OutputProbeMaskFileName'].astype(str) 

#        idxfile['ProbeFileID'] = idxfile['ProbeFileID'].astype(str) 
#        idxfile['ProbeHeight'] = idxfile['ProbeHeight'].astype(np.float64) 
#        idxfile['ProbeWidth'] = idxfile['ProbeWidth'].astype(np.float64) 


        sysPath = os.path.dirname(self.sysname)

        sysProbes = sysfile['ProbeFileID'].unique()
        idxProbes = idxfile['ProbeFileID'].unique()
        iminiProbes = 0
        if idxmini is not 0:
            iminiProbes = idxmini['ProbeFileID'].unique()

        self.sysProbes = sysProbes
        self.idxProbes = idxProbes
        self.iminiProbes = iminiProbes

        for probeID in idxProbes:
            if not (probeID in sysProbes):
                self.printbuffer.append("ERROR: {} seems to have been missed by the system file.".format(probeID))
                matchFlag = 1
                continue

        self.sysfile = sysfile
        self.idxfile = idxfile
        self.sysPath = sysPath

        self.sysfile['maskFlag'] = 0
        self.sysfile['matchFlag'] = 0
        self.sysfile['Message'] = '' #for return print messages

        #parallelize with multiprocessing a'la localization scorer
        sysfile=self.checkProbes(sysfile,self.procs)
        maskFlag = np.sum(sysfile['maskFlag'])
        matchFlag = np.sum(sysfile['matchFlag'])
        
        #final validation
        flagSum = maskFlag + dupFlag + xrowFlag + scoreFlag + matchFlag
        if flagSum == 0:
            self.printbuffer.append("The contents of your file are valid!")
            return 0
        else:
            self.printbuffer.append("The contents of your file are not valid!")
            return 1

    def checkProbes(self,maskData,processors):
        maxprocs = multiprocessing.cpu_count() - 2
        nrow = len(maskData)
        if (processors > nrow) and (nrow > 0):
            print("Warning: too many processors for rows in the data. Defaulting to rows in data ({}).".format(nrow))
            processors = nrow
        if processors > maxprocs:
            print("Warning: the machine does not have that many processors available. Defaulting to max ({}).".format(max(maxprocs,1)))
            processors = max(maxprocs,1)

        chunksize = nrow//processors
        maskDataS = [[self,maskData[i:(i+chunksize)]] for i in range(0,nrow,chunksize)]
        p = multiprocessing.Pool(processes=processors)
        maskDataS = p.map(checkProbe,maskDataS)
        p.close()
        maskData = pd.concat(maskDataS)
        #add all mask 'Message' entries to printBuffer
        maskData.apply(lambda x: self.printbuffer.append(x['Message']),axis=1,reduce=False)

        return maskData

    def checkMoreProbes(self,maskData):
        return maskData.apply(self.checkOneProbe,axis=1,reduce=False)

    #attach the flag to each row and send the row back
    def checkOneProbe(self,sysrow):
        probeFileID = sysrow['ProbeFileID']
        maskFlag = 0
        matchFlag = 0
        if not (probeFileID in self.idxProbes):
            sysrow['Message']="ERROR: {} does not exist in the index file.".format(probeFileID)
            sysrow['matchFlag'] = 1
            return sysrow
        elif self.iminiProbes is not 0:
            if not (probeFileID in self.iminiProbes):
#                sysrow['Message']="Neglecting mask validation for probe {}.".format(probeFileID)
                return sysrow

#                printq("The contents of your file are not valid!",True)
#                return 1

        if self.pixOptOut:
            oopixval = str(sysrow['ProbeOptOutPixelValue'])
            #check if ProbeOptOutPixelValue is blank or an integer if it exists in the header.
            if not ((oopixval == '') or (oopixval.isdigit())):
                sysrow['Message']="ERROR: ProbeOptOutPixelValue for probe {} is not blank ('') or an integer.".format(probeFileID)
                sysrow['matchFlag'] = 1

        #check mask validation
        if self.testMask and not self.neglectMask:
            probeOutputMaskFileName = sysrow['OutputProbeMaskFileName']
            if probeOutputMaskFileName in [None,'',np.nan,'nan']:
                sysrow['Message']=" ".join([sysrow['Message'],"The mask for file {} appears to be absent. Skipping it.".format(probeFileID)])
                return sysrow
            #if IsOptOut or ProbeStatus is present
            #check if IsOptOut is 'Y' or 'Detection'. Likewise for ProbeStatus as relevant
            if self.optOut == 1:
                #throw error if not in set of allowed values
                all_statuses = ['Y','Detection','Localization','N']
                if not sysrow['IsOptOut'] in all_statuses:
                    sysrow['Message'] = " ".join([sysrow['Message'],"Probe status {} for probe {} is not recognized.".format(sysrow['IsOptOut'],sysrow['ProbeFileID'])])
                    sysrow['matchFlag'] = 1
                if sysrow['IsOptOut'] in ['Y','Localization']:
                    #no need for localization checking
                    return sysrow
            elif self.optOut == 2:
                all_statuses = ['Processed','NonProcessed','OptOutAll','OptOutDetection','OptOutLocalization']
                if not sysrow['ProbeStatus'] in all_statuses:
                    sysrow['Message'] = " ".join([sysrow['Message'],"Probe status {} for probe {} is not recognized.".format(sysrow['ProbeStatus'],sysrow['ProbeFileID'])])
                    sysrow['matchFlag'] = 1
                if sysrow['ProbeStatus'] in ['OptOutAll','OptOutLocalization']:
                    return sysrow

            mskflag,msg = self.maskCheck(os.path.join(self.sysPath,probeOutputMaskFileName),probeFileID,self.idxfile,self.identify)
            sysrow['Message'] = "\n".join([sysrow['Message'],msg])
            sysrow['maskFlag'] = sysrow['maskFlag'] | mskflag 
        return sysrow

    def maskCheck(self,maskname,fileid,indexfile,identify):
        #check to see if index file input image files are consistent with system output
        flag = 0
        msg=["Validating {} for file {}...".format(maskname,fileid)]
        mask_pieces = maskname.split('.')
        mask_ext = mask_pieces[-1]
        if mask_ext != 'png':
            msg.append('ERROR: Mask image {} for FileID {} is not a png. Make it into a png!'.format(maskname,fileid))
            return 1,"\n".join(msg)
        if not os.path.isfile(maskname):
            msg.append("ERROR: {} does not exist! Did you name it wrong?".format(maskname))
            return 1,"\n".join(msg)
        baseHeight = list(map(int,indexfile['ProbeHeight'][indexfile['ProbeFileID'] == fileid]))[0] 
        baseWidth = list(map(int,indexfile['ProbeWidth'][indexfile['ProbeFileID'] == fileid]))[0]
    
        #subprocess with imagemagick identify for speed
        if identify:
            dimoutput = subprocess.check_output(["identify","-format","'%f|%w|%h'",maskname]).rstrip().replace("'","").split('|')
            dims = (int(dimoutput[2]),int(dimoutput[1]))
        else:
            dims = cv2.imread(maskname,cv2.IMREAD_UNCHANGED).shape
    
        if identify:
            channel = subprocess.check_output(["identify","-format","%[channels]",maskname]).rstrip()
            if channel != "gray":
                msg.append("ERROR: {} is not single-channel. It is {}. Make it single-channel.".format(maskname,channel))
                flag = 1
        elif len(dims)>2:
            msg.append("ERROR: {} is not single-channel. It has {} channels. Make it single-channel.".format(maskname,dims[2]))
            flag = 1
    
        if (baseHeight != dims[0]) or (baseWidth != dims[1]):
            msg.append("Dimensions for ProbeImg of ProbeFileID {}: {},{}".format(fileid,baseHeight,baseWidth))
            msg.append("Dimensions of mask {}: {},{}".format(maskname,dims[0],dims[1]))
            msg.append("ERROR: The mask image's length and width do not seem to be the same as the base image's.")
            flag = 1
            
        #maskImg <- readPNG(maskname) #EDIT: expensive for only getting the number of channels. Find cheaper option
        #note: No need to check for third channel. The imread option automatically reads as grayscale.
    
        if flag == 0:
            msg.append(maskname + " is valid.")
    
        return flag,"\n".join(msg)

class DSD_Validator(validator):
    def nameCheck(self,NCID):
        self.printbuffer.append('Validating the name of the system file...')

        sys_pieces = self.sysname.rsplit('.',1)
        sys_ext = sys_pieces[1]
        if sys_ext != 'csv':
            self.printbuffer.append('ERROR: Your system output is not a csv!')
            return 1
    
        fileExpid = sys_pieces[0].split('/')
        dirExpid = fileExpid[-2]
        fileExpid = fileExpid[-1]
        if fileExpid != dirExpid:
            self.printbuffer.append("ERROR: Please follow the naming convention. The system output should follow the naming <EXPID>/<EXPID>.csv.")
            return 1
    
        taskFlag = 0
        ncidFlag = 0
        teamFlag = 0
        sysPath = os.path.dirname(self.sysname)
        sysfName = os.path.basename(self.sysname)

        arrSplit = sysfName.split('_')
        if len(arrSplit) < 7:
            self.printbuffer.append("ERROR: There are not enough arguments to verify in the name.")
            return 1
        elif len(arrSplit) > 7:
            self.printbuffer.append("ERROR: The team name must not include underscores.")
            teamFlag = 1

        team = arrSplit[0]
        ncid = arrSplit[1]
        data = arrSplit[2]
        task = arrSplit[3]
        self.condition = arrSplit[4]
        sys = arrSplit[5]
        version = arrSplit[6]
    
        if team == '':
            self.printbuffer.append("ERROR: The team name must not include underscores.")
            teamFlag = 1
    
        if ncid != NCID:
            self.printbuffer.append("ERROR: The NCID must be {}.".format(NCID))
            ncidFlag = 1

        task = task.lower()
        if task != 'splice':
            self.printbuffer.append('ERROR: What kind of task is ' + task + '? It should be splice!')
            taskFlag = 1
    
        if (taskFlag == 0) and (ncidFlag == 0) and (teamFlag == 0):
            self.printbuffer.append('The name of this file is valid!')
            return 0
        else:
            self.printbuffer.append('The name of the file is not valid. Please review the requirements.')
            return 1 

    #redesigned pipeline
    def contentCheck(self,identify,neglectMask,reffname):
        self.printbuffer.append('Validating the syntactic content of the system output.')
        #read csv line by line
        dupFlag = 0
        xrowFlag = 0
        scoreFlag = 0
        colFlag = 0
        maskFlag = 0
        keyFlag = 0

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
                self.printbuffer.append("ERROR: The number of rows in your system output ({}) does not match the number of rows in the index file ({}).".format(s_len,i_len))
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

        self.printbuffer.append("Index read")

        sysPath = os.path.dirname(self.sysname)
        testMask = False
        optOut = 0
        with open(self.sysname) as sysfile:
            for idx,l in enumerate(sysfile):
                self.printbuffer.append("Process {} ".format(idx) + l)

                if idx == 0:
                    #parse headers
                    s_headnames = l.rstrip().split('|')
                    #header checking
                    if len(s_headnames) < 5:
                        #check number of headers
                        self.printbuffer.append("ERROR: The number of columns of the system output file must be at least 6. Are you using '|' to separate your columns?")
                        return 1
                    allClear = True
                    truelist = ["ProbeFileID","DonorFileID","ConfidenceScore"]
                    if self.condition in ["ImgOnly","ImgMeta"]:
                        truelist.extend(["OutputProbeMaskFileName","OutputDonorMaskFileName"])
                        testMask = True

                    
                    if not (("IsOptOut" in s_headnames) or (("ProbeStatus" in s_headnames) and ("DonorStatus" in s_headnames))):
                        self.printbuffer.append("ERROR: The column 'IsOptOut', or 'ProbeStatus' and 'DonorStatus', must be in the column headers.")
                    else:
                        if ("IsOptOut" in s_headnames) and ("ProbeStatus" in s_headnames) and ("DonorStatus" in s_headnames):
                            self.printbuffer.append("The system output has both 'IsOptOut' and 'ProbeStatus' in the column headers. It is advised for the performer not to confuse him or herself.")

                        if "IsOptOut" in s_headnames:
                            optOut = 1
                        elif ("ProbeStatus" in s_headnames) and ("DonorStatus" in s_headnames):
                            optOut = 2
                             
                    for th in truelist:
                        headcheck = th in s_headnames
                        allClear = allClear and headcheck
                        if not headcheck:
                            self.printbuffer.append("ERROR: The required column {} is absent.".format(th))
                    if not allClear:
                        return 1
            
#                    if ("OutputProbeMaskFileName" in s_headnames) and ("OutputDonorMaskFileName" in s_headnames):
#                        testMask = True

                    for i,h in enumerate(s_headnames):
                        #drop into dictionary for indexing
                        s_heads[h] = i
                    continue

                #for non-headers
                l_content = l.rstrip().replace("\"","").split('|')
                probeID = l_content[s_heads['ProbeFileID']]
                donorID = l_content[s_heads['DonorFileID']]
                key = ":".join([probeID,donorID])

                #try catch the key lookup
                try:
                    indRec = ind[key]
                except KeyError:
                    self.printbuffer.append("ERROR: The pair ({},{}) does not exist in the index file.".format(probeID,donorID))
                    keyFlag = 1
                    continue

                if reffname is not 0:
                    if r_files[key] == 'N':
#                        self.printbuffer.append("Skipping the pair ({},{}).".format(probeID,donorID))
                        continue

                #confidence score checking
                if not is_finite_number(l_content[s_heads['ConfidenceScore']]):
                    self.printbuffer.append("ERROR: Your Confidence Score for probe-donor pair ({}) is not a real numeric number.".format(key.replace(":",",")))
                    scoreFlag = 1

                if testMask and not neglectMask:
                    probeOutputMaskFileName = l_content[s_heads['OutputProbeMaskFileName']]
                    donorOutputMaskFileName = l_content[s_heads['OutputDonorMaskFileName']]

                    if (probeOutputMaskFileName == '') or (donorOutputMaskFileName == ''):
                        self.printbuffer.append("At least one mask for the pair ({},{}) appears to be absent. Skipping this pair.".format(probeID,donorID))
                        continue

                    optOutOption = 0
                    if optOut == 1:
                        all_statuses = ['Y','Detection','Localization','N']
                        if not l_content[s_heads['IsOptOut']] in all_statuses:
                            self.printbuffer.append("Probe status {} for probe {} is not recognized.".format(l_content[s_heads['IsOptOut']],probeID))
                            colFlag = 1
                        if l_content[s_heads['IsOptOut']] in ['Y','Localization']:
                            continue
                    elif optOut == 2:
                        #if ProbeStatus is opting out, opt out of Probe; likewise for Donor.
                        #split into donor and probe checking with optOutOption
                        all_statuses = ['Processed','NonProcessed','OptOutAll','OptOutDetection','OptOutLocalization']
                        if not l_content[s_heads['ProbeStatus']] in all_statuses:
                            self.printbuffer.append("Probe status {} for probe {} is not recognized.".format(l_content[s_heads['ProbeStatus']],probeID))
                            colFlag = 1
                        if not l_content[s_heads['DonorStatus']] in ['Processed','NonProcessed','OptOutLocalization']:
                            self.printbuffer.append("Donor status {} for donor {} is not recognized.".format(l_content[s_heads['DonorStatus']],donorID))
                            colFlag = 1

                        if l_content[s_heads['ProbeStatus']] in ["OptOutAll","OptOutLocalization"]:
                            optOutOption = optOutOption + 1
                        if l_content[s_heads['DonorStatus']] == "OptOutLocalization":
                            optOutOption = optOutOption + 2

                    probeWidth = int(indRec[i_heads['ProbeWidth']])
                    probeHeight = int(indRec[i_heads['ProbeHeight']])
                    donorWidth = int(indRec[i_heads['DonorWidth']])
                    donorHeight = int(indRec[i_heads['DonorHeight']])

                    maskFlag = maskFlag | self.maskCheck(os.path.join(sysPath,probeOutputMaskFileName),os.path.join(sysPath,donorOutputMaskFileName),probeID,donorID,probeWidth,probeHeight,donorWidth,donorHeight,idx,identify,optOutOption)

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
        if flagSum == 0:
            self.printbuffer.append("The contents of your file are valid!")
        else:
            self.printbuffer.append("The contents of your file are not valid!")
            return 1

    def contentCheck_0(self,identify,neglectMask,reffname):
        index_dtype = {'TaskID':str,
                 'ProbeFileID':str,
                 'ProbeFileName':str,
                 'ProbeWidth':np.int64,
                 'ProbeHeight':np.int64,
                 'DonorFileID':str,
                 'DonorFileName':str,
                 'DonorWidth':np.int64,
                 'DonorHeight':np.int64}
        idxfile = pd.read_csv(self.idxname,sep='|',dtype=index_dtype,na_filter=False)
        sysfile = pd.read_csv(self.sysname,sep='|',na_filter=False)

        dupFlag = 0
        xrowFlag = 0
        scoreFlag = 0
        maskFlag = 0
        
        if sysfile.shape[1] < 5:
            self.printbuffer.append("ERROR: The number of columns of the system output file must be at least 5. Are you using '|' to separate your columns?")
            return 1

        sysHeads = list(sysfile.columns)
        allClear = True
        #truelist = ["ProbeFileID","DonorFileID","ConfidenceScore","OutputProbeMaskFileName","OutputDonorMaskFileName","OptOut"]
        truelist = ["ProbeFileID","DonorFileID","ConfidenceScore","OutputProbeMaskFileName","OutputDonorMaskFileName"]

        for i in xrange(len(truelist)):
            allClear = allClear and (truelist[i] in sysHeads)
            if not (truelist[i] in sysHeads):
    #            headlist = []
    #            properhl = []
    #            for i in range(0,len(truelist)):
    #                if (sysHeads[i] != truelist[i]):
    #                    headlist.append(sysHeads[i])
    #                    properhl.append(truelist[i]) 
    #            printq("ERROR: Your header(s) " + ', '.join(headlist) + " should be " + ', '.join(properhl) + " respectively.",True)
                self.printbuffer.append("ERROR: The required column {} is absent.".format(truelist[i]))

        if not allClear:
            return 1
        
        if sysfile.shape[0] != sysfile.drop_duplicates().shape[0]:
            rowlist = xrange(sysfile.shape[0])
            self.printbuffer.append("ERROR: Your system output contains duplicate rows for ProbeFileID's: " + ' ,'.join(list(sysfile['ProbeFileID'][sysfile.duplicated()])) + " at row(s): " +\
                                     ' ,'.join(list(map(str,[i for i in rowlist if sysfile.duplicated()[i]]))) + " after the header. I recommended you delete these row(s).")
            dupFlag = 1
        
        if sysfile.shape[0] != idxfile.shape[0]:
#            printq("ERROR: The number of rows in your system output does not match the number of rows in the index file.",True)
            self.printbuffer.append("ERROR: The number of rows in your system output ({}) does not match the number of rows in the index file ({}).".format(sysfile.shape[0]+1,idxfile.shape[0]+1))
            xrowFlag = 1
        
        sysfile['ProbeFileID'] = sysfile['ProbeFileID'].astype(str) 
        try:
            sysfile['ConfidenceScore'] = sysfile['ConfidenceScore'].astype(np.float64)
        except ValueError:
            confprobes = sysfile[~sysfile['ConfidenceScore'].map(is_finite_number)][['ProbeFileID','DonorFileID']]
            self.printbuffer.append("ERROR: Your Confidence Scores for probe-donor pairs {} are not numeric.".format(confprobes))
            scoreFlag = 1
        sysfile['OutputProbeMaskFileName'] = sysfile['OutputProbeMaskFileName'].astype(str) 

        idxfile['ProbeFileID'] = idxfile['ProbeFileID'].astype(str) 
        idxfile['ProbeHeight'] = idxfile['ProbeHeight'].astype(np.uint32) 
        idxfile['ProbeWidth'] = idxfile['ProbeWidth'].astype(np.uint32) 

        sysfile['DonorFileID'] = sysfile['DonorFileID'].astype(str) 
        sysfile['OutputDonorMaskFileName'] = sysfile['OutputDonorMaskFileName'].astype(str) 

        idxfile['DonorFileID'] = idxfile['DonorFileID'].astype(str) 
        idxfile['DonorHeight'] = idxfile['DonorHeight'].astype(np.uint32) 
        idxfile['DonorWidth'] = idxfile['DonorWidth'].astype(np.uint32) 
        
        sysPath = os.path.dirname(self.sysname)

        if not ((dupFlag == 0) and (xrowFlag == 0) and (scoreFlag == 0)):
            self.printbuffer.append("The contents of your file are not valid!")
            return 1
        
        #NOTE: parallelize this a'la SSD has no point. Dictionary implementation would win even with several processors.
        for i in xrange(sysfile.shape[0]):
            if not (sysfile['ProbeFileID'][i] in idxfile['ProbeFileID'].unique()):
                self.printbuffer.append("ERROR: {} does not exist in the index file.".format(sysfile['ProbeFileID'][i]))
                self.printbuffer.append("The contents of your file are not valid!",True)
                return 1

            #First get all the matching probe rows
            rowset = idxfile[idxfile['ProbeFileID'] == sysfile['ProbeFileID'][i]].copy()
            #search in these rows for DonorFileID matches. If empty, pair does not exist. Quit status 1
            if not (sysfile['DonorFileID'][i] in rowset['DonorFileID'].unique()):
                self.printbuffer.append("ERROR: The pair ({},{}) does not exist in the index file.".format(sysfile['ProbeFileID'][i],sysfile['DonorFileID'][i]))
                self.printbuffer.append("The contents of your file are not valid!")
                return 1

            #check mask validation
            probeFileID = sysfile['ProbeFileID'][i]
            donorFileID = sysfile['DonorFileID'][i]
            probeOutputMaskFileName = sysfile['OutputProbeMaskFileName'][i]
            donorOutputMaskFileName = sysfile['OutputDonorMaskFileName'][i]
            idxStats = idxfile.query("ProbeFileID=='{}' and DonorFileID=='{}'".format(probeFileID,donorFileID))
            idxProbeWidth = idxStats['ProbeWidth'].iloc[0]
            idxProbeHeight = idxStats['ProbeHeight'].iloc[0]
            idxDonorWidth = idxStats['DonorWidth'].iloc[0]
            idxDonorHeight = idxStats['DonorHeight'].iloc[0]

            if (probeOutputMaskFileName in [None,'',np.nan,'nan']) or (donorOutputMaskFileName in [None,'',np.nan,'nan']):
                self.printbuffer.append("At least one mask for the pair ({},{}) appears to be absent. Skipping this pair.".format(probeFileID,donorFileID))
                continue
#            maskFlag = maskFlag | maskCheck2(os.path.join(sysPath,probeOutputMaskFileName),os.path.join(sysPath,donorOutputMaskFileName),sysfile['ProbeFileID'][i],sysfile['DonorFileID'][i],idxfile,i,identify)
            maskFlag = maskFlag | self.maskCheck(os.path.join(sysPath,probeOutputMaskFileName),os.path.join(sysPath,donorOutputMaskFileName),probeFileID,donorFileID,idxProbeWidth,idxProbeHeight,idxDonorWidth,idxDonorHeight,i,identify)
        
        #final validation
        if maskFlag==0:
            self.printbuffer.append("The contents of your file are valid!")
        else:
            self.printbuffer.append("The contents of your file are not valid!")
            return 1

    def maskCheck(self,pmaskname,dmaskname,probeid,donorid,pbaseWidth,pbaseHeight,dbaseWidth,dbaseHeight,rownum,identify,optOutOption=0):
        #check to see if index file input image files are consistent with system output
        flag = 0
        self.printbuffer.append("Validating probe and donor mask pair({},{}) for ({},{}) pair at row {}...".format(pmaskname,dmaskname,probeid,donorid,rownum))
        checkThisProbe = True
        checkThisDonor = True
        if optOutOption % 2 == 1:
            checkThisProbe = False
        if optOutOption // 2 == 1:
            checkThisDonor = False
   
        pngflag = False 
        eflag = False
        #check to see if index file input image files are consistent with system output
        if checkThisProbe:
            pmask_pieces,pmask_ext = os.path.splitext(pmaskname)
            if pmask_ext != '.png':
                self.printbuffer.append('ERROR: Probe mask image {} for pair ({},{}) at row {} is not a png. Make it into a png!'.format(pmaskname,probeid,donorid,rownum))
                pngflag = True
        
            #check to see if png files exist before doing anything with them.
            if not os.path.isfile(pmaskname):
                self.printbuffer.append("ERROR: {} does not exist! Did you name it wrong?".format(pmaskname))
                eflag = True

        if checkThisDonor:
            dmask_pieces,dmask_ext = os.path.splitext(dmaskname)
            if dmask_ext != '.png':
                self.printbuffer.append('ERROR: Donor mask image {} for pair ({},{}) at row {} is not a png. Make it into a png!'.format(dmaskname,probeid,donorid,rownum))
                pngflag = True
        
            if not os.path.isfile(dmaskname):
                self.printbuffer.append("ERROR: {} does not exist! Did you name it wrong?".format(dmaskname))
                eflag = True
        if eflag or pngflag:
            return 1
    
        #subprocess with imagemagick identify for speed
        if checkThisProbe:
            if identify:
                dimoutput = subprocess.check_output(["identify","-format","'%f|%w|%h'",pmaskname]).rstrip().replace("'","").split('|')
                pdims = (int(dimoutput[2]),int(dimoutput[1]))
            else:
                pdims = cv2.imread(pmaskname,cv2.IMREAD_UNCHANGED).shape
        
            if (pbaseHeight != pdims[0]) or (pbaseWidth != pdims[1]):
                self.printbuffer.append("Dimensions for ProbeImg of pair ({},{}): {},{}".format(probeid,donorid,pbaseHeight,pbaseWidth))
                self.printbuffer.append("Dimensions of probe mask {}: {},{}".format(pmaskname,pdims[0],pdims[1]))
                self.printbuffer.append("ERROR: The mask image's length and width do not seem to be the same as the base image's.")
                flag = 1
        
            if identify:
                channel = subprocess.check_output(["identify","-format","%[channels]",pmaskname]).rstrip()
                if channel != "gray":
                    self.printbuffer.append("ERROR: {} is not single-channel. It is {}. Make it single-channel.".format(pmaskname,channel))
                    flag = 1
            elif len(pdims)>2:
                self.printbuffer.append("ERROR: {} is not single-channel. It has {} channels. Make it single-channel.".format(pmaskname,pdims[2]))
                flag = 1

        if checkThisDonor:
            if identify:
                dimoutput = subprocess.check_output(["identify","-format","'%f|%w|%h'",dmaskname]).rstrip().replace("'","").split('|')
                ddims = (int(dimoutput[2]),int(dimoutput[1]))
            else: 
                ddims = cv2.imread(dmaskname,cv2.IMREAD_UNCHANGED).shape
        
            if identify:
                channel = subprocess.check_output(["identify","-format","%[channels]",dmaskname]).rstrip()
                if channel != "gray":
                    self.printbuffer.append("ERROR: {} is not single-channel. It is {}. Make it single-channel.".format(dmaskname,channel))
                    flag = 1
            elif len(ddims)>2:
                self.printbuffer.append("ERROR: {} is not single-channel. It has {} channels. Make it single-channel.".format(dmaskname,ddims[2]))
                flag = 1
        
            if (dbaseHeight != ddims[0]) or (dbaseWidth != ddims[1]):
                self.printbuffer.append("Dimensions for DonorImg of pair ({},{}): {},{}".format(probeid,donorid,dbaseHeight,dbaseWidth))
                self.printbuffer.append("Dimensions of probe mask {}: {},{}".format(dmaskname,ddims[0],ddims[1]))
                self.printbuffer.append("ERROR: The mask image's length and width do not seem to be the same as the base image's.")
                flag = 1
     
        if flag == 0:
            self.printbuffer.append("Your masks {} and {} are valid.".format(pmaskname,dmaskname))
        return flag

    def maskCheck_0(self,pmaskname,dmaskname,probeid,donorid,indexfile,rownum):
        #check to see if index file input image files are consistent with system output
        flag = 0
        self.printbuffer.append("Validating probe and donor mask pair({},{}) for ({},{}) pair at row {}...".format(pmaskname,dmaskname,probeid,donorid,rownum))
    
        #check to see if index file input image files are consistent with system output
        pmask_pieces = pmaskname.split('.')
        pmask_ext = pmask_pieces[-1]
        if pmask_ext != 'png':
            self.printbuffer.append('ERROR: Probe mask image {} for pair ({},{}) at row {} is not a png. Make it into a png!'.format(pmaskname,probeid,donorid,rownum))
            return 1
    
        dmask_pieces = dmaskname.split('.')
        dmask_ext = dmask_pieces[-1]
        if dmask_ext != 'png':
            self.printbuffer.append('ERROR: Donor mask image {} for pair ({},{}) at row {} is not a png. Make it into a png!'.format(dmaskname,probeid,donorid,rownum))
            return 1
    
        #check to see if png files exist before doing anything with them.
        eflag = False
        if not os.path.isfile(pmaskname):
            self.printbuffer.append("ERROR: {} does not exist! Did you name it wrong?".format(pmaskname))
            eflag = True
        if not os.path.isfile(dmaskname):
            self.printbuffer.append("ERROR: {} does not exist! Did you name it wrong?".format(dmaskname))
            eflag = True
        if eflag:
            return 1
    
        pbaseHeight = list(map(int,indexfile['ProbeHeight'][(indexfile['ProbeFileID'] == probeid) & (indexfile['DonorFileID'] == donorid)]))[0] 
        pbaseWidth = list(map(int,indexfile['ProbeWidth'][(indexfile['ProbeFileID'] == probeid) & (indexfile['DonorFileID'] == donorid)]))[0]
        pdims = cv2.imread(pmaskname,cv2.IMREAD_UNCHANGED).shape
    
        if len(pdims)>2:
            self.printbuffer.append("ERROR: {} is not single-channel. It has {} channels. Make it single-channel.".format(pmaskname,pdims[2]))
            flag = 1
    
        if (pbaseHeight != pdims[0]) or (pbaseWidth != pdims[1]):
            self.printbuffer.append("Dimensions for ProbeImg of pair ({},{}): {},{}".format(probeid,donorid,pbaseHeight,pbaseWidth))
            self.printbuffer.append("Dimensions of probe mask {}: {},{}".format(pmaskname,pdims[0],pdims[1]))
            self.printbuffer.append("ERROR: The mask image's length and width do not seem to be the same as the base image's.")
            flag = 1
         
        dbaseHeight = list(map(int,indexfile['DonorHeight'][(indexfile['ProbeFileID'] == probeid) & (indexfile['DonorFileID'] == donorid)]))[0] 
        dbaseWidth = list(indexfile['DonorWidth'][(indexfile['ProbeFileID'] == probeid) & (indexfile['DonorFileID'] == donorid)])[0]
        ddims = cv2.imread(dmaskname,cv2.IMREAD_UNCHANGED).shape
    
        if len(ddims)>2:
            self.printbuffer.append("ERROR: {} is not single-channel. It has {} channels. Make it single-channel.".format(dmaskname,ddims[2]))
            flag = 1
    
        if (dbaseHeight != ddims[0]) or (dbaseWidth != ddims[1]):
            self.printbuffer.append("Dimensions for DonorImg of pair ({},{}): {},{}".format(probeid,donorid,dbaseHeight,dbaseWidth))
            self.printbuffer.append("Dimensions of probe mask {}: {},{}".format(dmaskname,ddims[0],ddims[1]))
            self.printbuffer.append("ERROR: The mask image's length and width do not seem to be the same as the base image's.")
            flag = 1
     
        if flag == 0:
            self.printbuffer.append("Your masks {} and {} are valid.".format(pmaskname,dmaskname))
        return flag


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Validate the file and data format of the Single-Source Detection (SSD) or Double-Source Detection (DSD) files.')
    parser.add_argument('-x','--inIndex',type=str,default=None,\
    help='required index file',metavar='character')
    parser.add_argument('-s','--inSys',type=str,default=None,\
    help='required system output file',metavar='character')
    parser.add_argument('-r','--inRef',type=str,default=0,\
    help='optional reference file for filtration',metavar='character')
    parser.add_argument('-vt','--valtype',type=str,default=None,\
    help='required validator type',metavar='character')
    parser.add_argument('-nc','--nameCheck',action="store_true",\
    help='Check the format of the name of the file in question to make sure it matches up with the evaluation plan.')
    parser.add_argument('-id','--identify',action="store_true",\
    help='use ImageMagick\'s identify to get dimensions of mask. OpenCV reading is used by default.')
    parser.add_argument('-v','--verbose',type=int,default=None,\
    help='Control print output. Select 1 to print all non-error print output and 0 to suppress all printed output (bar argument-parsing errors).',metavar='0 or 1')
    parser.add_argument('-p','--processors',type=int,default=1,\
    help='The number of processors to use for validation. Choosing too many processors will cause the program to forcibly default to a smaller number. [default=1].',metavar='positive integer')
    parser.add_argument('-nm','--neglectMask',action="store_true",\
    help="neglect mask dimensionality validation.")
    parser.add_argument('--ncid',type=str,default="NC17",\
    help="the NCID to validate against.")

    if len(sys.argv) > 1:

        args = parser.parse_args()
        verbose = args.verbose
        #TODO: remove later
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

        if args.valtype == 'SSD':
            ssd_validation = SSD_Validator(args.inSys,args.inIndex,verbose)
            exit(ssd_validation.fullCheck(args.nameCheck,args.identify,args.ncid,args.neglectMask,args.inRef,args.processors))

        elif args.valtype == 'DSD':
            dsd_validation = DSD_Validator(args.inSys,args.inIndex,verbose)
            exit(dsd_validation.fullCheck(args.nameCheck,args.identify,args.ncid,args.neglectMask,args.inRef,args.processors))

        else:
            print("Validation type must be 'SSD' or 'DSD'.")
            exit(1)
    else:
        parser.print_help()
