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
import numpy as np
import maskreport as mr

# loading scoring and reporting libraries
lib_path = "../../lib"
sys.path.append(lib_path)
import Partition_mask as pt
#import masks
#execfile(os.path.join(lib_path,"masks.py"))
#execfile('maskreport.py')


########### Command line interface ########################################################

#data_path = "../../data"
refFname = "reference/manipulation/NC2016-manipulation-ref.csv"
indexFname = "indexes/NC2016-manipulation-index.csv"
#sysFname = data_path + "/SystemOutputs/dct0608/dct02.csv"

#################################################
## command-line arguments for "file"
#################################################

parser = argparse.ArgumentParser(description='Compute scores for the masks and generate a report.')
parser.add_argument('-t','--task',type=str,default='manipulation',
help='Two different types of tasks: [manipulation] and [splice]',metavar='character')
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
help="Directory root to save outputs.",metavar='character')

#added from DetectionScorer.py
factor_group = parser.add_mutually_exclusive_group()
factor_group.add_argument('-f', '--factor', nargs='*',
help="Evaluate algorithm performance by given queries.", metavar='character')
factor_group.add_argument('-fp', '--factorp',
help="Evaluate algorithm performance with partitions given by one query (syntax : '==[]','<','<=')", metavar='character')

parser.add_argument('-tmt','--targetManiType',type=str,default='all',
help="An array of manipulations to be scored, separated by commas (e.g. 'remove,clone'). Select 'all' to score all manipulated regions regardless of manipulation.",metavar='character')

parser.add_argument('--eks',type=int,default=15,
help="Erosion kernel size number must be odd, [default=15]",metavar='integer')
parser.add_argument('--dks',type=int,default=9,
help="Dilation kernel size number must be odd, [default=9]",metavar='integer')
parser.add_argument('--ntdks',type=int,default=15,
help="Non-target dilation kernel for distraction no-score regions. Size number must be odd, [default=15]",metavar='integer')
parser.add_argument('-k','--kernel',type=str,default='box',
help="Convolution kernel type for erosion and dilation. Choose from [box],[disc],[diamond],[gaussian], or [line]. The default is 'box'.",metavar='character')
parser.add_argument('--rbin',type=int,default=-1,
help="Binarize the reference mask in the relevant mask file to black and white with a numeric threshold in the interval [0,255]. Pick -1 to evaluate the relevant regions based on the other arguments. [default=-1]",metavar='integer')
parser.add_argument('--sbin',type=int,default=-1,
help="Binarize the system output mask to black and white with a numeric threshold in the interval [0,255]. Pick -1 to choose the threshold for the mask at the maximal absolute MCC value. [default=-1]",metavar='integer')
#parser.add_argument('--avgOver',type=str,default='',
#help="A collection of features to average reports over, separated by commas.", metavar="character")
parser.add_argument('-v','--verbose',type=int,default=None,
help="Control print output. Select 1 to print all non-error print output and 0 to suppress all print output (bar argument-parsing errors).",metavar='0 or 1')
parser.add_argument('--precision',type=int,default=16,
help="The number of digits to round computed scores, [e.g. a score of 0.3333333333333... will round to 0.33333 for a precision of 5], [default=16].",metavar='positive integer')
parser.add_argument('-html',help="Output data to HTML files.",action="store_true")

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

if args.task not in ['manipulation','splice']:
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
    os.system('mkdir ' + args.outRoot)

printq("Starting a report ...")

if args.task == 'manipulation':
    index_dtype = {'TaskID':str,
             'ProbeFileID':str,
             'ProbeFileName':str,
             'ProbeWidth':np.int64,
             'ProbeHeight':np.int64}
    sys_dtype = {'ProbeFileID':str,
             'ConfidenceScore':str, #this should be "string" due to the "nan" value, otherwise "nan"s will have different unique numbers
             'OutputProbeMaskFileName':str}
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
             'OutputProbeMaskFileName':str,
             'OutputDonorMaskFileName':str}

mySysDir = os.path.join(args.sysDir,os.path.dirname(args.inSys))
mySysFile = os.path.join(args.sysDir,args.inSys)
myRef = pd.read_csv(os.path.join(myRefDir,args.inRef),sep='|',header=0)
mySys = pd.read_csv(mySysFile,sep='|',header=0,dtype=sys_dtype)
myIndex = pd.read_csv(os.path.join(myRefDir,args.inIndex),sep='|',header=0,dtype=index_dtype)

factor_mode = ''
query = ''
if args.factor:
    factor_mode = 'f'
    query = args.factor
elif args.factorp:
    factor_mode = 'fp'
    query = args.factorp

## if the confidence score are 'nan', replace the values with the mininum score
#mySys[pd.isnull(mySys['ConfidenceScore'])] = mySys['ConfidenceScore'].min()
## convert to the str type to the float type for computations
#mySys['ConfidenceScore'] = mySys['ConfidenceScore'].astype(np.float)

outRoot = args.outRoot

prefix = os.path.basename(args.inSys).split('.')[0]

reportq = 0
if args.verbose:
    reportq = 1

if args.precision < 1:
    printq("Precision should not be less than 1 for scores to be meaningful. Defaulting to 5 digits.")
    args.precision=5

sub_ref = myRef[myRef['IsTarget']=="Y"].copy()

# Merge the reference and system output for SSD/DSD reports
if args.task == 'manipulation':
    #update accordingly along with ProbeJournalJoin and JournalMask csv's in refDir
    refpfx = os.path.join(myRefDir,args.inRef.split('.')[0])
    probeJournalJoin = pd.read_csv(refpfx + '-probejournaljoin.csv',sep='|',header=0)
    journalMask = pd.read_csv(refpfx + '-journalmask.csv',sep='|',header=0)

    m_df = pd.merge(sub_ref, mySys, how='left', on='ProbeFileID')
    # get rid of inf values from the merge and entries for which there is nothing to work with.
    m_df = m_df.replace([np.inf,-np.inf],np.nan).dropna(subset=['OutputProbeMaskFileName'])

    # if the confidence score are 'nan', replace the values with the mininum score
    m_df.ix[pd.isnull(m_df['ConfidenceScore']),'ConfidenceScore'] = mySys['ConfidenceScore'].min()
    # convert to the str type to the float type for computations
    m_df['ConfidenceScore'] = m_df['ConfidenceScore'].astype(np.float)

    journalData0 = pd.merge(probeJournalJoin,journalMask,how='left',on=['JournalID','StartNodeID','EndNodeID'])
    n_journals = len(journalData0)
    journalData0['scored'] = pd.Series(['N']*n_journals) #add column for scored: 'Y'/'N'
    journalData = journalData0.copy()

    if (args.targetManiType != 'all'):
        journalData = journalData0.query("Purpose=={}".format(args.targetManiType.split(','))) #filter by targetManiType
        journalData0.loc[journalData0.query("Purpose=={}".format(args.targetManiType.split(','))).index,'scored'] = 'Y'
    else:
        journalData0.loc[journalData0.query("ProbeFileID=={}".format(sub_ref['ProbeFileID'].tolist())).index,'scored'] = 'Y'

    #m_df = pd.merge(m_df,probeJournalJoin,how='left',on='ProbeFileID')
    #m_df = pd.merge(journalMask,m_df,how='left',on='JournalID')

    #partition query here and filter further
    #selection = f.Partition(m_df,query,factor_mode)
    #DM_List = selection.part_dm_list
    #table_df = selection.render_table()

    r_df = mr.createReportSSD(m_df,journalData, myRefDir, mySysDir,args.rbin,args.sbin,args.targetManiType,args.eks, args.dks, args.ntdks, args.kernel, args.outRoot, html=args.html,verbose=reportq,precision=args.precision)
    #get the columns of journalData that were not scored and set the same columns in journalData0 to 'N'
    journalData0.ix[journalData.ProbeFileID.isin(r_df.query('MCC == -2')['ProbeFileID'].tolist()),'scored'] = 'N'

    r_df = r_df.query('MCC > -2') #remove the rows that were not scored due to no region being present. We set those rows to have MCC == -2.
    metrics = ['NMM','MCC','WL1']
    my_partition = pt.Partition(r_df,query,factor_mode,metrics) #average over queries
    df_list = my_partition.render_table(metrics)
 
    if args.factor:
        #use Partition for OOP niceness and to identify file to be written. 
        for i,temp_df in enumerate(df_list):
            temp_df.to_csv(path_or_buf="{}_{}.csv".format(os.path.join(outRoot,prefix + '-mask_scores'),i),index=False)
            
    elif args.factorp or (factor_mode == ''):
        a_df = df_list[0]
        a_df.to_csv(path_or_buf=os.path.join(args.outRoot,prefix + "-mask_score.csv"),index=False)

    journalData0.to_csv(path_or_buf=os.path.join(args.outRoot,prefix + '-journalResults.csv'),index=False)

#commenting out for the time being
#elif args.task in ['removal','clone']:
#    m_df = pd.merge(sub_ref, mySys, how='left', on='ProbeFileID')
#    # get rid of inf values from the merge and entries for which there is nothing to work with.
#    m_df = m_df.replace([np.inf,-np.inf],np.nan).dropna(subset=['ProbeMaskFileName'])
#
#    # if the confidence score are 'nan', replace the values with the mininum score
#    m_df.ix[pd.isnull(m_df['ConfidenceScore']),'ConfidenceScore'] = mySys['ConfidenceScore'].min()
#    # convert to the str type to the float type for computations
#    m_df['ConfidenceScore'] = m_df['ConfidenceScore'].astype(np.float)
#    r_df = createReportSSD(m_df, myRefDir, mySysDir,args.rbin,args.sbin,args.targetManiType,args.eks, args.dks, args.outRoot, html=args.html,verbose=reportq,precision=args.precision) # default eks 15, dks 9
#    a_df = avg_scores_by_factors_SSD(r_df,args.task,avglist,precision=args.precision)
#
elif args.task == 'splice':
    m_df = pd.merge(sub_ref, mySys, how='left', on=['ProbeFileID','DonorFileID'])

    # get rid of inf values from the merge
    m_df = m_df.replace([np.inf,-np.inf],np.nan).dropna(subset=['ProbeMaskFileName',
                                                                'DonorMaskFileName'])
    # if the confidence score are 'nan', replace the values with the mininum score
    m_df.ix[pd.isnull(m_df['ConfidenceScore']),'ConfidenceScore'] = mySys['ConfidenceScore'].min()
    # convert to the str type to the float type for computations
    m_df['ConfidenceScore'] = m_df['ConfidenceScore'].astype(np.float)

    r_df = mr.createReportDSD(m_df, myRefDir, mySysDir,args.rbin,args.sbin,args.eks, args.dks, kern=args.kernel, outputRoot=args.outRoot, html=args.html,verbose=reportq,precision=args.precision)

    metrics = ['pNMM','pMCC','pWL1','dNMM','dMCC','dWL1']
    my_partition = pt.Partition(r_df,query,factor_mode,metrics) #average over queries
    df_list = my_partition.render_table(metrics)

    if args.factor:
        #use Partition for OOP niceness and to identify file to be written. 
        for i,temp_df in enumerate(df_list):
            temp_df.to_csv(path_or_buf="{}_{}.csv".format(os.path.join(outRoot,prefix + '-mask_scores'),i),index=False)
            
    elif args.factorp or (factor_mode == ''):
        a_df = df_list[0]
        a_df.to_csv(path_or_buf=os.path.join(outRoot,prefix + "-mask_score.csv"),index=False)

if verbose: #to avoid complications of print formatting when not verbose
    precision = args.precision
    if args.task in ['manipulation']:
        myavgs = [a_df[mets][0] for mets in ['NMM','MCC','WL1']]
    
        allmets = "Avg NMM: {}, Avg MCC: {}, Avg WL1: {}".format(round(myavgs[0],precision),
                                                                 round(myavgs[1],precision),
                                                                 round(myavgs[2],precision))
        printq(allmets)
    
    elif args.task == 'splice':
        pavgs  = [a_df[mets][0] for mets in ['pNMM','pMCC','pWL1']]
        davgs  = [a_df[mets][0] for mets in ['dNMM','dMCC','dWL1']]
        pallmets = "Avg pNMM: {}, Avg pMCC: {}, Avg pWL1: {}".format(round(pavgs[0],precision),
                                                                     round(pavgs[1],precision),
                                                                     round(pavgs[2],precision))
        dallmets = "Avg dNMM: {}, Avg dMCC: {}, Avg dWL1: {}".format(round(davgs[0],precision),
                                                                     round(davgs[1],precision),
                                                                     round(davgs[2],precision))
        printq(pallmets)
        printq(dallmets)
    else:
        printerr("ERROR: Task not recognized.")

prefix = os.path.basename(args.inSys).split('.')[0]
r_df.to_csv(path_or_buf=os.path.join(args.outRoot,prefix + '-mask_scores_perimage.csv'),index=False)

