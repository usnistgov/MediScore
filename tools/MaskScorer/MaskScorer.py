#!/usr/bin/python
"""
* File: MaskScorer.py
* Date: 08/30/2016
* Translated by Daniel Zhou
* Original implemented by Yooyoung Lee 
* Status: Complete 

* Description: This calculates performance scores for localizing mainpulated area 
                between reference mask and system output mask 

* Requirements: This code requires the following packages:

    - opencv
    - pandas

  The rest are available on your system

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

########### packages ########################################################
import sys
import argparse
import os
import cv2
import pandas as pd
import argparse

########### Command line interface ########################################################

data_path = "../../data"
refFname = "reference/manipulation/NC2016-manipulation-ref.csv"
indexFname = "indexes/NC2016-manipulation-index.csv"
sysFname = data_path + "/SystemOutputs/dct0608/dct02.csv"

#################################################
## command-line arguments for "file"
#################################################

parser = argparse.ArgumentParser(description='Compute scores for the masks and generate a report.')
parser.add_argument('-t','--task',type=str,default='manipulation',
help='Three different types of tasks: [manipulation],[removal],[clone], and [splice]',metavar='character')
parser.add_argument('--refDir',type=str,default='.',
help='NC2016_Test directory path: [e.g., ../../data/NC2016_Test]',metavar='character')
parser.add_argument('--sysDir',type=str,default='.',
help='System output directory path: [e.g., ../../data/NC2016_Test]',metavar='character')
parser.add_argument('-r','--inRef',type=str,
help='Reference csv file name: [e.g., reference/manipulation/NC2016-manipulation-ref.csv]',metavar='character')
parser.add_argument('-s','--inSys',type=str,
help='System output csv file name: [e.g., ~/expid/system_output.csv]',metavar='character')
parser.add_argument('-x','--inIndex',type=str,default=indexFname,
help='Task Index csv file name: [e.g., indexes/NC2016-manipulation-index.csv]',metavar='character')
parser.add_argument('-oR','--outRoot',type=str,default='.',
help="Directory root + file name to save outputs.",metavar='character')
parser.add_argument('--eks',type=int,default=15,
help="Erosion kernel size number must be odd, [default=15]",metavar='integer')
parser.add_argument('--dks',type=int,default=9,
help="Dilation kernel size number must be odd, [default=9]",metavar='integer')
parser.add_argument('--rbin',type=int,default=254,
help="Binarize the reference mask in the relevant mask file to black and white with a numeric threshold in the interval [0,255]. Pick -1 to not binarize and leave the mask as is. [default=254]",metavar='integer')
parser.add_argument('--sbin',type=int,default=-1,
help="Binarize the system output mask to black and white with a numeric threshold in the interval [0,255]. Pick -1 to not binarize and leave the mask as is. [default=254]",metavar='integer')
#parser.add_argument('--avgOver',type=str,default='',
#help="A collection of features to average reports over, separated by commas.", metavar="character")
parser.add_argument('-v','--verbose',type=int,default=None,
help="Control print output. Select 1 to print all non-error print output and 0 to suppress all print output (bar argument-parsing errors).",metavar='0 or 1')
parser.add_argument('--precision',type=int,default=5,
help="The number of digits to round computed scores, [e.g. a score of 0.3333333333333... will round to 0.33333 for a precision of 5], [default=5].",metavar='positive integer')
parser.add_argument('-html',help="Output data to HTML files.",action="store_true")

# loading scoring and reporting libraries
lib_path = "../../lib"
execfile(os.path.join(lib_path,"masks.py")) #EDIT: find better way to import?
execfile('maskreport.py')

args = parser.parse_args()
verbose=args.verbose

#wrapper print function for print message suppression
if verbose:
    def printq(string):
        print(string)
else:
    printq = lambda *a:None

#wrapper print function when encountering an error. Will also cause script to exit after message is printed.

if verbose==0:
    printerr = lambda *a:None
else:
    def printerr(string,exitcode=1):
        if verbose != 0:
            parser.print_help()
            print(string)
            exit(exitcode)

if args.task not in [None,'manipulation','removal','clone','splice']:
    printerr("ERROR: Task type must be supplied.")
if args.refDir is None:
    printerr("ERROR: NC2016_Test directory path must be supplied.")

myRefDir = args.refDir 

if args.inRef is None:
    printerr("ERROR: Input file name for reference must be supplied.")

if args.inSys is None: 
    printerr("ERROR: Input file name for system output must be supplied.")

if args.inIndex is None:
    printerr("ERROR: Input file name for index files must be supplied.")

#create the folder and save the mask outputs
#set.seed(1)

#assume outRoot exists
if args.outRoot is None:
    printerr("ERROR: the folder name for outputs must be supplied.")

if not os.path.isdir(args.outRoot):
    myout = '/'.join(args.outRoot.split('/')[:-1])
    os.system('mkdir ' + myout)

printq("Starting a report ...")

#avglist = args.avgOver.replace(' ','') #strip all whitespace just in case
#avglist = avglist.split(',') #split by comma
avglist = ['']  #TODO: temporary fix until the averaging by factors procedure is finalized.
if avglist == ['']:
    avglist = []

if args.task in ['manipulation','removal','clone']:
    index_dtype = {'TaskID':str,
             'ProbeFileID':str,
             'ProbeFileName':str,
             'ProbeWidth':np.int64,
             'ProbeHeight':np.int64}
    sys_dtype = {'ProbeFileID':str,
             'ConfidenceScore':str, #this should be "string" due to the "nan" value, otherwise "nan"s will have different unique numbers
             'ProbeOutputMaskFileName':str}
elif args.task == 'splice':
    index_dtype = {'TaskID':str,
             'ProbeFileID':str,
             'ProbeFileName':str,
             'ProbeWidth':np.int64,
             'ProbeHeight':np.int64,
             'DonorFileID':str,
             'DonorFileName':str,
             'DonorWidth':np.int64,
             'DonorHeight':np.int64}
    sys_dtype = {'ProbeFileID':str,
             'DonorFileID':str,
             'ConfidenceScore':str, #this should be "string" due to the "nan" value, otherwise "nan"s will have different unique numbers
             'ProbeOutputMaskFileName':str,
             'DonorOutputMaskFileName':str}

mySysDir = os.path.join(args.sysDir,os.path.dirname(args.inSys))
mySysFile = os.path.join(args.sysDir,args.inSys)
myRef = pd.read_csv(os.path.join(myRefDir,args.inRef),sep='\|',header=0,engine='python')
mySys = pd.read_csv(mySysFile,sep='\|',header=0,engine='python',dtype=sys_dtype)
myIndex = pd.read_csv(os.path.join(myRefDir,args.inIndex),sep='\|',header=0,engine='python',dtype=index_dtype)

#TODO: Validate Index and Sys here?

# if the confidence score are 'nan', replace the values with the mininum score
mySys[pd.isnull(m_df['ConfidenceScore'])] = mySys['ConfidenceScore'].min()
# convert to the str type to the float type for computations
mySys['ConfidenceScore'] = mySys['ConfidenceScore'].astype(np.float)

reportq = 0
if args.verbose:
    reportq = 1

if args.precision < 1:
    printq("Precision should not be less than 1 for scores to be meaningful. Defaulting to 5 digits.")
    args.precision=5

if args.task in ['manipulation','removal','clone']:
    r_df = createReportSSD(myRef, mySys, myIndex, myRefDir, mySysDir,args.rbin,args.sbin,args.eks, args.dks, args.outRoot, html=args.html,verbose=reportq,precision=args.precision) # default eks 15, dks 9
    a_df = avg_scores_by_factors_SSD(r_df,args.task,avglist,precision=args.precision)
elif args.task == 'splice':
    r_df = createReportDSD(myRef, mySys, myIndex, myRefDir, mySysDir,args.rbin,args.sbin,args.eks, args.dks, args.outRoot, html=args.html,verbose=reportq,precision=args.precision) # default eks 15, dks 9
    a_df = avg_scores_by_factors_DSD(r_df,args.task,avglist,precision=args.precision)

precision = args.precision
if args.task in ['manipulation','removal','clone']:
    myavgs = [a_df[mets][0] for mets in ['NMM','MCC','HAM','WL1','HL1']]

    allmets = "Avg NMM: {}, Avg MCC: {}, Avg HAM: {}, Avg WL1: {}, Avg HL1: {}".format(round(myavgs[0],precision),
                                                                                       round(myavgs[1],precision),
                                                                                       round(myavgs[2],precision),
                                                                                       round(myavgs[3],precision),
                                                                                       round(myavgs[4],precision))
    printq(allmets)

elif args.task == 'splice':
    pavgs  = [a_df[mets][0] for mets in ['pNMM','pMCC','pHAM','pWL1','pHL1']]
    davgs  = [a_df[mets][0] for mets in ['dNMM','dMCC','dHAM','dWL1','dHL1']]
    pallmets = "Avg pNMM: {}, Avg pMCC: {}, Avg pHAM: {}, Avg pWL1: {}, Avg pHL1: {}".format(round(pavgs[0],precision),
                                                                                             round(pavgs[1],precision),
                                                                                             round(pavgs[2],precision),
                                                                                             round(pavgs[3],precision),
                                                                                             round(pavgs[4],precision))
    dallmets = "Avg dNMM: {}, Avg dMCC: {}, Avg dHAM: {}, Avg dWL1: {}, Avg dHL1: {}".format(round(davgs[0],precision),
                                                                                             round(davgs[1],precision),
                                                                                             round(davgs[2],precision),
                                                                                             round(davgs[3],precision),
                                                                                             round(davgs[4],precision))
    printq(pallmets)
    printq(dallmets)
else:
    printerr("ERROR: Task not recognized.")

outRoot = args.outRoot
if outRoot[-1]=='/':
    outRoot = outRoot[:-1]

r_df.to_csv(path_or_buf=outRoot + '-perimage.csv',index=False)
a_df.to_csv(path_or_buf=outRoot + ".csv",index=False)
