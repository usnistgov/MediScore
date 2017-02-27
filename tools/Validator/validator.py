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
from abc import ABCMeta, abstractmethod

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

class validator:
    __metaclass__ = ABCMeta

    def __init__(self,sysfname,idxfname):
        self.sysname=sysfname
        self.idxname=idxfname
    @abstractmethod
    def nameCheck(self): pass
    @abstractmethod
    def contentCheck(self,identify): pass
    def fullCheck(self,nc,identify):
        #check for existence of files
        eflag = False
        if not os.path.isfile(self.sysname):
            printq("ERROR: I can't find your system output " + self.sysname + "! Where is it?",True)
            eflag = True
        if not os.path.isfile(self.idxname):
            printq("ERROR: I can't find your index file " + self.idxname + "! Where is it?",True)
            eflag = True
        if eflag:
            return 1

        #option to do a namecheck
        if nc:
            if self.nameCheck() == 1:
                return 1

        printq("Checking if index file is a pipe-separated csv...")
        idx_pieces = self.idxname.split('.')
        idx_ext = idx_pieces[-1]

        if idx_ext != 'csv':
            printq("ERROR: Your index file should have csv as an extension! (It is separated by '|', I know...)",True)
            return 1

        printq("Your index file appears to be a pipe-separated csv, for now. Hope it isn't separated by commas.")

        if self.contentCheck(identify) == 1:
            return 1
        return 0


class SSD_Validator(validator):
    def nameCheck(self):
        printq('Validating the name of the system file...')

        sys_pieces = self.sysname.rsplit('.',1)
        sys_ext = sys_pieces[1]
        if sys_ext != 'csv':
            printq('ERROR: Your system output is not a csv!',True)
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
        team = arrSplit[0]
        ncid = arrSplit[1]
        data = arrSplit[2]
        task = arrSplit[3]
        condition = arrSplit[4]
        sys = arrSplit[5]
        version = arrSplit[6]
    
        if ('+' in team) or (team == ''):
            printq("ERROR: The team name must not include characters + or _",True)
            teamFlag = 1
        task = task.lower()
        if (task != 'manipulation') and (task != 'provenance') and (task != 'provenancefiltering'):
            printq('ERROR: What kind of task is ' + task + '? It should be manipulation, provenance, or provenancefiltering!',True)
            taskFlag = 1
    
        if (taskFlag == 0) and (teamFlag == 0):
            printq('The name of this file is valid!')
        else:
            printq('The name of the file is not valid. Please review the requirements.',True)
            return 1 

    def contentCheck(self,identify):
        printq('Validating the syntactic content of the system output.')
        index_dtype = {'TaskID':str,
                 'ProbeFileID':str,
                 'ProbeFileName':str,
                 'ProbeWidth':np.int64,
                 'ProbeHeight':np.int64}
        idxfile = pd.read_csv(self.idxname,sep='|',dtype=index_dtype,na_filter=False)
        sysfile = pd.read_csv(self.sysname,sep='|',na_filter=False)
    
        dupFlag = 0
        xrowFlag = 0
        maskFlag = 0
        
        if sysfile.shape[1] < 3:
            printq("ERROR: The number of columns of the system output file must be at least 3. Are you using '|' to separate your columns?",True)
            return 1

        sysHeads = list(sysfile.columns)
        allClear = True
#        truelist = ["ProbeFileID","ConfidenceScore","OutputProbeMaskFileName","OptOut"]
        truelist = ["ProbeFileID","ConfidenceScore","OutputProbeMaskFileName"]

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
                return 1

        if not allClear:
            return 1
        
        if sysfile.shape[0] != sysfile.drop_duplicates().shape[0]:
            rowlist = range(0,sysfile.shape[0])
            printq("ERROR: Your system output contains duplicate rows for ProbeFileID's: "
                    + ' ,'.join(list(map(str,sysfile['ProbeFileID'][sysfile.duplicated()])))    + " at row(s): "
                    + ' ,'.join(list(map(str,[i for i in rowlist if sysfile.duplicated()[i]]))) + " after the header. I recommended you delete these row(s).",True)
            dupFlag = 1
        
        if sysfile.shape[0] != idxfile.shape[0]:
            printq("ERROR: The number of rows in your system output does not match the number of rows in the index file.",True)
            xrowFlag = 1
        
        if not ((dupFlag == 0) and (xrowFlag == 0)):
            printq("The contents of your file are not valid!",True)
            return 1
        
        sysfile['ProbeFileID'] = sysfile['ProbeFileID'].astype(str)
        sysfile['ConfidenceScore'] = sysfile['ConfidenceScore'].astype(np.float64)
        sysfile['OutputProbeMaskFileName'] = sysfile['OutputProbeMaskFileName'].astype(str) 

        idxfile['ProbeFileID'] = idxfile['ProbeFileID'].astype(str) 
        idxfile['ProbeHeight'] = idxfile['ProbeHeight'].astype(np.float64) 
        idxfile['ProbeWidth'] = idxfile['ProbeWidth'].astype(np.float64) 

        sysPath = os.path.dirname(self.sysname)
    
        for i in range(0,sysfile.shape[0]):
            if not (sysfile['ProbeFileID'][i] in idxfile['ProbeFileID'].unique()):
                printq("ERROR: " + sysfile['ProbeFileID'][i] + " does not exist in the index file.",True)
                printq("The contents of your file are not valid!",True)
                return 1

            #check mask validation
            probeOutputMaskFileName = sysfile['OutputProbeMaskFileName'][i]
            if probeOutputMaskFileName in [None,'',np.nan,'nan']:
                printq("The mask for file " + sysfile['ProbeFileID'][i] + " appears to be absent. Skipping it.")
                continue
            maskFlag = maskFlag | maskCheck1(os.path.join(sysPath,sysfile['OutputProbeMaskFileName'][i]),sysfile['ProbeFileID'][i],idxfile,identify)
        
        #final validation
        if maskFlag == 0:
            printq("The contents of your file are valid!")
        else:
            printq("The contents of your file are not valid!",True)
            return 1

class DSD_Validator(validator):
    def nameCheck(self):
        printq('Validating the name of the system file...')

        sys_pieces = self.sysname.rsplit('.',1)
        sys_ext = sys_pieces[1]
        if sys_ext != 'csv':
            printq('ERROR: Your system output is not a csv!',True)
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
        team = arrSplit[0]
        ncid = arrSplit[1]
        data = arrSplit[2]
        task = arrSplit[3]
        condition = arrSplit[4]
        sys = arrSplit[5]
        version = arrSplit[6]
    
        if '+' in team or team == '':
            printq("ERROR: The team name must not include characters + or _",True)
            teamFlag = 1
    
        task = task.lower()
        if task != 'splice':
            printq('ERROR: What kind of task is ' + task + '? It should be splice!',True)
            taskFlag = 1
    
        if (taskFlag == 0) and (teamFlag == 0):
            printq('The name of this file is valid!')
        else:
            printq('The name of the file is not valid. Please review the requirements.',True)
            return 1 

    #redesigned pipeline
    def contentCheck(self,identify):
        printq('Validating the syntactic content of the system output.')
        #read csv line by line
        dupFlag = 0
        xrowFlag = 0
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
                printq("ERROR: The number of rows in your system output does not match the number of rows in the index file.",True)
                xrowFlag = 1

        with open(self.idxname) as idxfile:
            for i,l in enumerate(idxfile):
                if i==0:
                    i_headnames = l.split('|')
                    for i,h in enumerate(i_headnames):
                        i_heads[h.replace('\n','')] = i
                else: break

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
                    ind[i_data[i_heads['ProbeFileID']] + ":" + i_data[i_heads['DonorFileID']]] = i_data

        printq("Index read")

        sysPath = os.path.dirname(self.sysname)
        with open(self.sysname) as sysfile:
            for idx,l in enumerate(sysfile):
                printq("Process {} ".format(idx) + l)

                if idx == 0:
                    #parse headers
                    s_headnames = l.rstrip().split('|')
                    #header checking
                    if len(s_headnames) < 5:
                        #check number of headers
                        printq("ERROR: The number of columns of the system output file must be at least 5. Are you using '|' to separate your columns?",True)
                        return 1
                    allClear = True
                    truelist = ["ProbeFileID","DonorFileID","ConfidenceScore","OutputProbeMaskFileName","OutputDonorMaskFileName"]
                    for th in truelist:
                        allClear = allClear and (th in s_headnames)
                        if not (th in s_headnames):
                            printq("ERROR: The required column {} is absent.".format(th),True)
                    if not allClear:
                        return 1
            
                    for i,h in enumerate(s_headnames):
                        #drop into dictionary for indexing
                        s_heads[h] = i
                else:
                    #for non-headers
                    l_content = l.rstrip().replace("\"","").split('|')
                    probeID = l_content[s_heads['ProbeFileID']]
                    donorID = l_content[s_heads['DonorFileID']]
                    probeOutputMaskFileName = l_content[s_heads['OutputProbeMaskFileName']]
                    donorOutputMaskFileName = l_content[s_heads['OutputDonorMaskFileName']]

                    if (probeOutputMaskFileName == '') or (donorOutputMaskFileName == ''):
                        printq("At least one mask for the pair (" + probeID + "," + donorID + ") appears to be absent. Skipping this pair.")
                        continue
 
                    key = l_content[s_heads['ProbeFileID']] + ":" + l_content[s_heads['DonorFileID']]

                    #try catch the key lookup
                    try:
                        indRec = ind[key]
                    except KeyError:
                        printq("ERROR: The pair ({},{}) does not exist in the index file.".format(probeID,donorID),True)
                        keyFlag = 1
                        continue

                    probeWidth = int(indRec[i_heads['ProbeWidth']])
                    probeHeight = int(indRec[i_heads['ProbeHeight']])
                    donorWidth = int(indRec[i_heads['DonorWidth']])
                    donorHeight = int(indRec[i_heads['DonorHeight']])

                    maskFlag = maskFlag | maskCheck2(os.path.join(sysPath,probeOutputMaskFileName),os.path.join(sysPath,donorOutputMaskFileName),probeID,donorID,probeWidth,probeHeight,donorWidth,donorHeight,idx,identify)

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
        if (maskFlag == 0) and (dupFlag == 0) and (xrowFlag == 0) and (keyFlag == 0):
            printq("The contents of your file are valid!")
        else:
            printq("The contents of your file are not valid!",True)
            return 1

    def contentCheck_0(self,identify):
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
        maskFlag = 0
        
        if sysfile.shape[1] < 5:
            printq("ERROR: The number of columns of the system output file must be at least 5. Are you using '|' to separate your columns?",True)
            return 1

        sysHeads = list(sysfile.columns)
        allClear = True
        #truelist = ["ProbeFileID","DonorFileID","ConfidenceScore","OutputProbeMaskFileName","OutputDonorMaskFileName","OptOut"]
        truelist = ["ProbeFileID","DonorFileID","ConfidenceScore","OutputProbeMaskFileName","OutputDonorMaskFileName"]

        for i in range(0,len(truelist)):
            allClear = allClear and (truelist[i] in sysHeads)
            if not (truelist[i] in sysHeads):
    #            headlist = []
    #            properhl = []
    #            for i in range(0,len(truelist)):
    #                if (sysHeads[i] != truelist[i]):
    #                    headlist.append(sysHeads[i])
    #                    properhl.append(truelist[i]) 
    #            printq("ERROR: Your header(s) " + ', '.join(headlist) + " should be " + ', '.join(properhl) + " respectively.",True)
                printq("ERROR: The required column {} is absent.".format(truelist[i]),True)

        if not allClear:
            return 1
        
        if sysfile.shape[0] != sysfile.drop_duplicates().shape[0]:
            rowlist = range(0,sysfile.shape[0])
            printq("ERROR: Your system output contains duplicate rows for ProbeFileID's: " + ' ,'.join(list(sysfile['ProbeFileID'][sysfile.duplicated()])) + " at row(s): " +\
                                     ' ,'.join(list(map(str,[i for i in rowlist if sysfile.duplicated()[i]]))) + " after the header. I recommended you delete these row(s).",True)
            dupFlag = 1
        
        if sysfile.shape[0] != idxfile.shape[0]:
            printq("ERROR: The number of rows in your system output does not match the number of rows in the index file.",True)
            xrowFlag = 1
        
        if not ((dupFlag == 0) and (xrowFlag == 0)):
            printq("The contents of your file are not valid!",True)
            return 1
        
        sysfile['ProbeFileID'] = sysfile['ProbeFileID'].astype(str) 
        sysfile['ConfidenceScore'] = sysfile['ConfidenceScore'].astype(np.float64) 
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
    
        for i in range(0,sysfile.shape[0]):
            if not (sysfile['ProbeFileID'][i] in idxfile['ProbeFileID'].unique()):
                printq("ERROR: " + sysfile['ProbeFileID'][i] + " does not exist in the index file.",True)
                printq("The contents of your file are not valid!",True)
                return 1

            #First get all the matching probe rows
            rowset = idxfile[idxfile['ProbeFileID'] == sysfile['ProbeFileID'][i]].copy()
            #search in these rows for DonorFileID matches. If empty, pair does not exist. Quit status 1
            if not (sysfile['DonorFileID'][i] in rowset['DonorFileID'].unique()):
                printq("ERROR: The pair (" + sysfile['ProbeFileID'][i] + "," + sysfile['DonorFileID'][i] + ") does not exist in the index file.",True)
                printq("The contents of your file are not valid!",True)
                return 1

            #check mask validation
            probeOutputMaskFileName = sysfile['OutputProbeMaskFileName'][i]
            donorOutputMaskFileName = sysfile['OutputDonorMaskFileName'][i]

            if (probeOutputMaskFileName in [None,'',np.nan,'nan']) or (donorOutputMaskFileName in [None,'',np.nan,'nan']):
                printq("At least one mask for the pair (" + sysfile['ProbeFileID'][i] + "," + sysfile['DonorFileID'][i] + ") appears to be absent. Skipping this pair.")
                continue
            maskFlag = maskFlag | maskCheck2(os.path.join(sysPath,probeOutputMaskFileName),os.path.join(sysPath,donorOutputMaskFileName),sysfile['ProbeFileID'][i],sysfile['DonorFileID'][i],idxfile,i,identify)
        
        #final validation
        if maskFlag==0:
            printq("The contents of your file are valid!")
        else:
            printq("The contents of your file are not valid!",True)
            return 1

######### Functions that don't depend on validator ####################################################

def maskCheck1(maskname,fileid,indexfile,identify):
    #check to see if index file input image files are consistent with system output
    flag = 0
    printq("Validating {} for file {}...".format(maskname,fileid))
    mask_pieces = maskname.split('.')
    mask_ext = mask_pieces[-1]
    if mask_ext != 'png':
        printq('ERROR: Mask image {} for FileID {} is not a png. Make it into a png!'.format(maskname,fileid),True)
        return 1
    if not os.path.isfile(maskname):
        printq("ERROR: " + maskname + " does not exist! Did you name it wrong?",True)
        return 1
    baseHeight = list(map(int,indexfile['ProbeHeight'][indexfile['ProbeFileID'] == fileid]))[0] 
    baseWidth = list(map(int,indexfile['ProbeWidth'][indexfile['ProbeFileID'] == fileid]))[0]

    #subprocess with imagemagick identify for speed
    if identify:
        dimoutput = subprocess.check_output(["identify","-format","'%f|%w|%h'",maskname]).rstrip().replace("'","").split('|')
        dims = (int(dimoutput[2]),int(dimoutput[1]))
    else:
        dims = cv2.imread(maskname,cv2.IMREAD_UNCHANGED).shape

    if identify:
        channel = subprocess.check_output(["identify","-format","%[channels]",maskname])
        if channel != "gray\n":
            printq("ERROR: {} is not single-channel. Make it single-channel.".format(maskname),True)
            flag = 1
    elif len(dims)>2:
        printq("ERROR: {} is not single-channel. Make it single-channel.".format(maskname),True)
        flag = 1

    if (baseHeight != dims[0]) or (baseWidth != dims[1]):
        printq("Dimensions for ProbeImg of ProbeFileID {}: {},{}".format(fileid,baseHeight,baseWidth),True)
        printq("Dimensions of mask {}: {},{}".format(maskname,dims[0],dims[1]),True)
        printq("ERROR: The mask image's length and width do not seem to be the same as the base image's.",True)
        flag = 1
        
    #maskImg <- readPNG(maskname) #EDIT: expensive for only getting the number of channels. Find cheaper option
    #note: No need to check for third channel. The imread option automatically reads as grayscale.

    if flag == 0:
        printq(maskname + " is valid.")

    return flag

def maskCheck2(pmaskname,dmaskname,probeid,donorid,pbaseWidth,pbaseHeight,dbaseWidth,dbaseHeight,rownum,identify):
    #check to see if index file input image files are consistent with system output
    flag = 0
    printq("Validating probe and donor mask pair({},{}) for ({},{}) pair at row {}...".format(pmaskname,dmaskname,probeid,donorid,rownum))

    #check to see if index file input image files are consistent with system output
    pmask_pieces,pmask_ext = os.path.splitext(pmaskname)
    if pmask_ext != '.png':
        printq('ERROR: Probe mask image {} for pair ({},{}) at row {} is not a png. Make it into a png!'.format(pmaskname,probeid,donorid,rownum),True)
        return 1

    dmask_pieces,dmask_ext = os.path.splitext(dmaskname)
    if dmask_ext != '.png':
        printq('ERROR: Donor mask image {} for pair ({},{}) at row {} is not a png. Make it into a png!'.format(dmaskname,probeid,donorid,rownum),True)
        return 1

    #check to see if png files exist before doing anything with them.
    eflag = False
    if not os.path.isfile(pmaskname):
        printq("ERROR: {} does not exist! Did you name it wrong?".format(pmaskname),True)
        eflag = True
    if not os.path.isfile(dmaskname):
        printq("ERROR: {} does not exist! Did you name it wrong?".format(dmaskname),True)
        eflag = True
    if eflag:
        return 1

    #subprocess with imagemagick identify for speed
    if identify:
        dimoutput = subprocess.check_output(["identify","-format","'%f|%w|%h'",pmaskname]).rstrip().replace("'","").split('|')
        pdims = (int(dimoutput[2]),int(dimoutput[1]))
    else:
        pdims = cv2.imread(pmaskname,cv2.IMREAD_UNCHANGED).shape

    if identify:
        channel = subprocess.check_output(["identify","-format","%[channels]",pmaskname])
        if channel != "gray\n":
            printq("ERROR: {} is not single-channel. Make it single-channel.".format(pmaskname),True)
            flag = 1
    elif len(pdims)>2:
        printq("ERROR: {} is not single-channel. Make it single-channel.".format(pmaskname),True)
        flag = 1

    if (pbaseHeight != pdims[0]) or (pbaseWidth != pdims[1]):
        printq("Dimensions for ProbeImg of pair ({},{}): {},{}".format(probeid,donorid,pbaseHeight,pbaseWidth),True)
        printq("Dimensions of probe mask {}: {},{}".format(pmaskname,pdims[0],pdims[1]),True)
        printq("ERROR: The mask image's length and width do not seem to be the same as the base image's.",True)
        flag = 1

    if identify:
        dimoutput = subprocess.check_output(["identify","-format","'%f|%w|%h'",dmaskname]).rstrip().replace("'","").split('|')
        ddims = (int(dimoutput[2]),int(dimoutput[1]))
    else: 
        ddims = cv2.imread(dmaskname,cv2.IMREAD_UNCHANGED).shape

    if identify:
        channel = subprocess.check_output(["identify","-format","%[channels]",dmaskname])
        if channel != "gray\n":
            printq("ERROR: {} is not single-channel. Make it single-channel.".format(dmaskname),True)
            flag = 1
    elif len(ddims)>2:
        printq("ERROR: {} is not single-channel. Make it single-channel.".format(dmaskname),True)
        flag = 1

    if (dbaseHeight != ddims[0]) or (dbaseWidth != ddims[1]):
        printq("Dimensions for DonorImg of pair ({},{}): {},{}".format(probeid,donorid,dbaseHeight,dbaseWidth),True)
        printq("Dimensions of probe mask {}: {},{}".format(dmaskname,ddims[0],ddims[1]),True)
        printq("ERROR: The mask image's length and width do not seem to be the same as the base image's.",True)
        flag = 1
 
    if flag == 0:
        printq("Your masks {} and {} are valid.".format(pmaskname,dmaskname))
    return flag

def maskCheck2_0(pmaskname,dmaskname,probeid,donorid,indexfile,rownum):
    #check to see if index file input image files are consistent with system output
    flag = 0
    printq("Validating probe and donor mask pair({},{}) for ({},{}) pair at row {}...".format(pmaskname,dmaskname,probeid,donorid,rownum))

    #check to see if index file input image files are consistent with system output
    pmask_pieces = pmaskname.split('.')
    pmask_ext = pmask_pieces[-1]
    if pmask_ext != 'png':
        printq('ERROR: Probe mask image {} for pair ({},{}) at row {} is not a png. Make it into a png!'.format(pmaskname,probeid,donorid,rownum),True)
        return 1

    dmask_pieces = dmaskname.split('.')
    dmask_ext = dmask_pieces[-1]
    if dmask_ext != 'png':
        printq('ERROR: Donor mask image {} for pair ({},{}) at row {} is not a png. Make it into a png!'.format(dmaskname,probeid,donorid,rownum),True)
        return 1

    #check to see if png files exist before doing anything with them.
    eflag = False
    if not os.path.isfile(pmaskname):
        printq("ERROR: " + pmaskname + " does not exist! Did you name it wrong?",True)
        eflag = True
    if not os.path.isfile(dmaskname):
        printq("ERROR: " + dmaskname + " does not exist! Did you name it wrong?",True)
        eflag = True
    if eflag:
        return 1

    pbaseHeight = list(map(int,indexfile['ProbeHeight'][(indexfile['ProbeFileID'] == probeid) & (indexfile['DonorFileID'] == donorid)]))[0] 
    pbaseWidth = list(map(int,indexfile['ProbeWidth'][(indexfile['ProbeFileID'] == probeid) & (indexfile['DonorFileID'] == donorid)]))[0]
    pdims = cv2.imread(pmaskname,cv2.IMREAD_UNCHANGED).shape

    if len(pdims)>2:
        printq("ERROR: {} is not single-channel. Make it single-channel.".format(pmaskname),True)
        flag = 1

    if (pbaseHeight != pdims[0]) or (pbaseWidth != pdims[1]):
        printq("Dimensions for ProbeImg of pair ({},{}): {},{}".format(probeid,donorid,pbaseHeight,pbaseWidth),True)
        printq("Dimensions of probe mask {}: {},{}".format(pmaskname,pdims[0],pdims[1]),True)
        printq("ERROR: The mask image's length and width do not seem to be the same as the base image's.",True)
        flag = 1
     
    dbaseHeight = list(map(int,indexfile['DonorHeight'][(indexfile['ProbeFileID'] == probeid) & (indexfile['DonorFileID'] == donorid)]))[0] 
    dbaseWidth = list(indexfile['DonorWidth'][(indexfile['ProbeFileID'] == probeid) & (indexfile['DonorFileID'] == donorid)])[0]
    ddims = cv2.imread(dmaskname,cv2.IMREAD_UNCHANGED).shape

    if len(ddims)>2:
        printq("ERROR: {} is not single-channel. Make it single-channel.".format(dmaskname),True)
        flag = 1

    if (dbaseHeight != ddims[0]) or (dbaseWidth != ddims[1]):
        printq("Dimensions for DonorImg of pair ({},{}): {},{}".format(probeid,donorid,dbaseHeight,dbaseWidth),True)
        printq("Dimensions of probe mask {}: {},{}".format(dmaskname,ddims[0],ddims[1]),True)
        printq("ERROR: The mask image's length and width do not seem to be the same as the base image's.",True)
        flag = 1
 
    if flag == 0:
        printq("Your masks {} and {} are valid.".format(pmaskname,dmaskname))
    return flag


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Validate the file and data format of the Single-Source Detection (SSD) or Double-Source Detection (DSD) files.')
    parser.add_argument('-x','--inIndex',type=str,default=None,\
    help='required index file',metavar='character')
    parser.add_argument('-s','--inSys',type=str,default=None,\
    help='required system output file',metavar='character')
    parser.add_argument('-vt','--valtype',type=str,default=None,\
    help='required validator type',metavar='character')
    parser.add_argument('-v','--verbose',type=int,default=None,\
    help='Control print output. Select 1 to print all non-error print output and 0 to suppress all printed output (bar argument-parsing errors).',metavar='0 or 1')
    parser.add_argument('-nc','--nameCheck',action="store_true",\
    help='Check the format of the name of the file in question to make sure it matches up with the evaluation plan.')
    parser.add_argument('-id','--identify',action="store_true",\
    help='use ImageMagick\'s identify to get dimensions of mask. OpenCV reading is used by default.')

    if len(sys.argv) > 1:

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

        if args.valtype == 'SSD':
            ssd_validation = SSD_Validator(args.inSys,args.inIndex)
            ssd_validation.fullCheck(args.nameCheck,args.identify)

        elif args.valtype == 'DSD':
            dsd_validation = DSD_Validator(args.inSys,args.inIndex)
            dsd_validation.fullCheck(args.nameCheck,args.identify)

        else:
            print("Validation type must be 'SSD' or 'DSD'.")
            exit(1)
    else:
        parser.print_help()
