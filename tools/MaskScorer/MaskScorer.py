#!/usr/bin/env python2
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
#import maskreport as mr
#import pdb #debug purposes

# loading scoring and reporting libraries
#lib_path = "../../lib"
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib")
sys.path.append(lib_path)
from metricRunner import maskMetricRunner
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
factor_group.add_argument('-q', '--query', nargs='*',
help="Evaluate algorithm performance by given queries.", metavar='character')
factor_group.add_argument('-qp', '--queryPartition',
help="Evaluate algorithm performance with partitions given by one query (syntax : '==[]','<','<=','>','>=')", metavar='character')
factor_group.add_argument('-qm', '--queryManipulation', nargs='*',
help="Filter the data by given queries before evaluation. Each query will result in a separate evaluation run.", metavar='character')

parser.add_argument('--eks',type=int,default=15,
help="Erosion kernel size number must be odd, [default=15]",metavar='integer')
parser.add_argument('--dks',type=int,default=11,
help="Dilation kernel size number must be odd, [default=11]",metavar='integer')
parser.add_argument('--ntdks',type=int,default=15,
help="Non-target dilation kernel for distraction no-score regions. Size number must be odd, [default=15]",metavar='integer')
parser.add_argument('-k','--kernel',type=str,default='box',
help="Convolution kernel type for erosion and dilation. Choose from [box],[disc],[diamond],[gaussian], or [line]. The default is 'box'.",metavar='character')
parser.add_argument('--rbin',type=int,default=-1,
help="Binarize the reference mask in the relevant mask file to black and white with a numeric threshold in the interval [0,255]. Pick -1 to evaluate the relevant regions based on the other arguments. [default=-1]",metavar='integer')
parser.add_argument('--sbin',type=int,default=-1,
help="Binarize the system output mask to black and white with a numeric threshold in the interval [0,255]. -1 indicates that the threshold for the mask will be chosen at the maximal absolute MCC value. [default=-1]",metavar='integer')
parser.add_argument('--nspx',type=int,default=-1,
help="Set a pixel value in the system output mask to be the no-score region [0,255]. -1 indicates that no particular pixel value will be chosen to be the no-score zone. [default=-1]",metavar='integer')

#parser.add_argument('--avgOver',type=str,default='',
#help="A collection of features to average reports over, separated by commas.", metavar="character")
parser.add_argument('-v','--verbose',type=int,default=None,
help="Control print output. Select 1 to print all non-error print output and 0 to suppress all print output (bar argument-parsing errors).",metavar='0 or 1')
parser.add_argument('--precision',type=int,default=16,
help="The number of digits to round computed scores, [e.g. a score of 0.3333333333333... will round to 0.33333 for a precision of 5], [default=16].",metavar='positive integer')
parser.add_argument('-html',help="Output data to HTML files.",action="store_true")
parser.add_argument('--optOut',action='store_true',help="Evaluate algorithm performance on trials where the IsOptOut value is 'N' only.")
parser.add_argument('-xF','--indexFilter',action='store_true',help="Filter scoring to only files that are present in the index file. This option permits scoring to select index files for the purpose of testing, and may accept system outputs that have not passed the validator.")
parser.add_argument('--speedup',action='store_true',help="Run mask evaluation with a sped-up evaluator.")

args = parser.parse_args()
verbose=args.verbose

#wrapper print function for print message suppression
if verbose:
    def printq(string):
        print(string)
else:
    printq = lambda *a:None

#wrapper print function when encountering an error. Will also cause script to exit after message is printed.

#if verbose==0:
#    def printerr(string,exitcode=1):
#        exit(exitcode)
#else:
def printerr(string,exitcode=1):
    if verbose != 0:
        parser.print_help()
        print(string)
    exit(exitcode)

args.task = args.task.lower()

if args.task not in ['manipulation','splice']:
    printerr("ERROR: Task type must be supplied.")
if args.refDir is None:
    printerr("ERROR: NC2016_Test directory path must be supplied.")

mySysDir = os.path.join(args.sysDir,os.path.dirname(args.inSys))
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

#define HTML functions here
df2html = lambda *a:None
if args.task == 'manipulation':
    def createReport(m_df, journalData, probeJournalJoin, index, refDir, sysDir, rbin, sbin,erodeKernSize, dilateKernSize,distractionKernSize, kern,outputRoot,html,verbose,precision):
        # if the confidence score are 'nan', replace the values with the mininum score
        #m_df[pd.isnull(m_df['ConfidenceScore'])] = m_df['ConfidenceScore'].min()
        # convert to the str type to the float type for computations
        #m_df['ConfidenceScore'] = m_df['ConfidenceScore'].astype(np.float)
    
        metricRunner = maskMetricRunner(m_df,args.refDir,mySysDir,args.rbin,args.sbin,journalData,probeJournalJoin,index,mode=0,speedup=args.speedup)
        df = metricRunner.getMetricList(args.eks,args.dks,args.ntdks,args.nspx,args.kernel,args.outRoot,args.verbose,args.html,precision=args.precision)
    
        merged_df = pd.merge(m_df.drop('Scored',1),df,how='left',on='ProbeFileID')
        return merged_df

    if args.html:
        def df2html(df,average_df,outputRoot,queryManipulation,query):
            html_out = df.copy()
    
            #os.path.join doesn't seem to work with Pandas Series so just do a manual string addition
            if outputRoot[-1] == '/':
                outputRoot = outputRoot[:-1]
    
            #set links around the system output data frame files for images that are not NaN
            #html_out.ix[~pd.isnull(html_out['OutputProbeMaskFileName']),'ProbeFileName'] = '<a href="' + outputRoot + '/' + html_out.ix[~pd.isnull(html_out['OutputProbeMaskFileName']),'ProbeFileName'].str.split('/').str.get(-1).str.split('.').str.get(0) + '.html">' + html_out['ProbeFileName'] + '</a>'
            pd.set_option('display.max_colwidth',-1)
#            html_out.loc[~pd.isnull(html_out['OutputProbeMaskFileName']) & (html_out['Scored'] == 'Y'),'ProbeFileName'] = '<a href="' + html_out.ix[~pd.isnull(html_out['OutputProbeMaskFileName']),'ProbeFileID'] + '/' + html_out.ix[~pd.isnull(html_out['OutputProbeMaskFileName']),'ProbeFileName'].str.split('/').str.get(-1).str.split('.').str.get(0) + '.html">' + html_out.ix[~pd.isnull(html_out['OutputProbeMaskFileName']),'ProbeFileName'].str.split('/').str.get(-1) + '</a>'
            html_out.loc[html_out['Scored'] == 'Y','ProbeFileName'] = '<a href="' + html_out.ix[html_out['Scored'] == 'Y','ProbeFileID'] + '/' + html_out.ix[html_out['Scored'] == 'Y','ProbeFileName'].str.split('/').str.get(-1).str.split('.').str.get(0) + '.html">' + html_out.ix[html_out['Scored'] == 'Y','ProbeFileName'].str.split('/').str.get(-1) + '</a>'

            html_out = html_out.round({'NMM':3,'MCC':3,'BWL1':3,'GWL1':3})

            #final filtering
            html_out.loc[html_out.query("MCC == -2").index,'MCC'] = ''
            html_out.loc[html_out.query("MCC == 0 & Scored == 'N'").index,'Scored'] = 'Y'

            #write to index.html
            fname = os.path.join(outputRoot,'index.html')
            myf = open(fname,'w')
            myf.write(html_out.to_html(escape=False).replace("text-align: right;","text-align: center;").encode('utf-8'))
            myf.write('\n')
            #write the query if manipulated
            if queryManipulation:
                myf.write("\nFiltered by query: {}\n".format(query))

            if average_df is not 0:
                #write title and then average_df
                a_df_copy = average_df.copy().round({'NMM':3,'MCC':3,'BWL1':3,'GWL1':3})
                myf.write('<h3>Average Scores</h3>\n')
                myf.write(a_df_copy.to_html().replace("text-align: right;","text-align: center;"))
            myf.close()

elif args.task == 'splice':
    def createReport(m_df, journalData, probeJournalJoin, index, refDir, sysDir, rbin, sbin,erodeKernSize, dilateKernSize,distractionKernSize, kern,outputRoot,html,verbose,precision):
        #finds rows in index and sys which correspond to target reference
        #sub_index = index[sub_ref['ProbeFileID'].isin(index['ProbeFileID']) & sub_ref['DonorFileID'].isin(index['DonorFileID'])]
        #sub_sys = sys[sub_ref['ProbeFileID'].isin(sys['ProbeFileID']) & sub_ref['DonorFileID'].isin(sys['DonorFileID'])]
    
        # if the confidence score are 'nan', replace the values with the mininum score
        #m_df[pd.isnull(m_df['ConfidenceScore'])] = m_df['ConfidenceScore'].min()
        # convert to the str type to the float type for computations
        #m_df['ConfidenceScore'] = m_df['ConfidenceScore'].astype(np.float)
#        maskMetricRunner = mm.maskMetricList(m_df,refDir,sysDir,rbin,sbin,journalData,probeJournalJoin,index,mode=1)
        metricRunner = maskMetricRunner(m_df,args.refDir,mySysDir,args.rbin,args.sbin,journalData,probeJournalJoin,index,mode=1,speedup=args.speedup)
#        probe_df = maskMetricRunner.getMetricList(erodeKernSize,dilateKernSize,0,kern,outputRoot,verbose,html,precision=precision)
        probe_df = metricRunner.getMetricList(args.eks,args.dks,0,args.nspx,args.kernel,args.outRoot,args.verbose,args.html,precision=args.precision)
    
#        maskMetricRunner = mm.maskMetricList(m_df,refDir,sysDir,rbin,sbin,journalData,probeJournalJoin,index,mode=2) #donor images
        metricRunner = maskMetricRunner(m_df,args.refDir,mySysDir,args.rbin,args.sbin,journalData,probeJournalJoin,index,mode=2,speedup=args.speedup)
#        donor_df = maskMetricRunner.getMetricList(erodeKernSize,dilateKernSize,0,kern,outputRoot,verbose,html,precision=precision)
        donor_df = metricRunner.getMetricList(args.eks,args.dks,0,args.nspx,args.kernel,args.outRoot,args.verbose,args.html,precision=args.precision)
    
        probe_df.rename(index=str,columns={"NMM":"pNMM",
                                           "MCC":"pMCC",
                                           "BWL1":"pBWL1",
                                           "GWL1":"pGWL1",
                                           "ColMaskFileName":"ProbeColMaskFileName",
                                           "AggMaskFileName":"ProbeAggMaskFileName",
                                           "Scored":"ProbeScored"},inplace=True)
    
        donor_df.rename(index=str,columns={"NMM":"dNMM",
                                           "MCC":"dMCC",
                                           "BWL1":"dBWL1",
                                           "GWL1":"dGWL1",
                                           "ColMaskFileName":"DonorColMaskFileName",
                                           "AggMaskFileName":"DonorAggMaskFileName",
                                           "Scored":"DonorScored"},inplace=True)
    
        pd_df = pd.concat([probe_df,donor_df],axis=1)
        merged_df = pd.merge(m_df,pd_df,how='left',on=['ProbeFileID','DonorFileID']).drop('Scored',1)
        return merged_df

    if args.html:
        def df2html(df,average_df,outputRoot,queryManipulation,query):
            html_out = df.copy()
    
            #os.path.join doesn't seem to work with Pandas Series so just do a manual string addition
            if outputRoot[-1] == '/':
                outputRoot = outputRoot[:-1]
    
            #set links around the system output data frame files for images that are not NaN
            #html_out.ix[~pd.isnull(html_out['OutputProbeMaskFileName']),'ProbeFileName'] = '<a href="' + outputRoot + '/' + html_out.ix[~pd.isnull(html_out['OutputProbeMaskFileName']),'ProbeFileName'].str.split('/').str.get(-1).str.split('.').str.get(0) + '.html">' + html_out['ProbeFileName'] + '</a>'
            #html_out.ix[~pd.isnull(html_out['OutputDonorMaskFileName']),'DonorFileName'] = '<a href="' + outputRoot + '/' + html_out.ix[~pd.isnull(html_out['OutputDonorMaskFileName']),'DonorFileName'].str.split('/').str.get(-1).str.split('.').str.get(0) + '.html">' + html_out['DonorFileName'] + '</a>'
            pd.set_option('display.max_colwidth',-1)
#            html_out.loc[~pd.isnull(html_out['OutputProbeMaskFileName']),'ProbeFileName'] = '<a href="' + html_out.ix[~pd.isnull(html_out['OutputProbeMaskFileName']),'ProbeFileID'] + '_' + html_out.ix[~pd.isnull(html_out['OutputProbeMaskFileName']),'DonorFileID'] + '/probe/' + html_out.ix[~pd.isnull(html_out['OutputProbeMaskFileName']),'ProbeFileName'].str.split('/').str.get(-1).str.split('.').str.get(0) + '.html">' + html_out.ix[~pd.isnull(html_out['OutputProbeMaskFileName']),'ProbeFileName'].str.split('/').str.get(-1) + '</a>'
#            html_out.loc[~pd.isnull(html_out['OutputDonorMaskFileName']),'DonorFileName'] = '<a href="' + html_out.ix[~pd.isnull(html_out['OutputProbeMaskFileName']),'ProbeFileID'] + '_' + html_out.ix[~pd.isnull(html_out['OutputProbeMaskFileName']),'DonorFileID'] + '/donor/' + html_out.ix[~pd.isnull(html_out['OutputDonorMaskFileName']),'DonorFileName'].str.split('/').str.get(-1).str.split('.').str.get(0) + '.html">' + html_out.ix[~pd.isnull(html_out['OutputDonorMaskFileName']),'DonorFileName'].str.split('/').str.get(-1) + '</a>'

            html_out.loc[html_out['ProbeScored'] == 'Y','ProbeFileName'] = '<a href="' + html_out.ix[html_out['ProbeScored'] == 'Y','ProbeFileID'] + '_' + html_out.ix[html_out['ProbeScored'] == 'Y','DonorFileID'] + '/probe/' + html_out.ix[html_out['ProbeScored'] == 'Y','ProbeFileName'].str.split('/').str.get(-1).str.split('.').str.get(0) + '.html">' + html_out.ix[html_out['ProbeScored'] == 'Y','ProbeFileName'].str.split('/').str.get(-1) + '</a>'
            html_out.loc[html_out['DonorScored'] == 'Y','DonorFileName'] = '<a href="' + html_out.ix[html_out['DonorScored'] == 'Y','ProbeFileID'] + '_' + html_out.ix[html_out['DonorScored'] == 'Y','DonorFileID'] + '/donor/' + html_out.ix[html_out['DonorScored'] == 'Y','DonorFileName'].str.split('/').str.get(-1).str.split('.').str.get(0) + '.html">' + html_out.ix[html_out['DonorScored'] == 'Y','DonorFileName'].str.split('/').str.get(-1) + '</a>'

            html_out = html_out.round({'pNMM':3,'pMCC':3,'pBWL1':3,'pGWL1':3,'dNMM':3,'dMCC':3,'dBWL1':3,'dGWL1':3})

            html_out.loc[html_out.query("pMCC == -2").index,'pMCC'] = ''
            html_out.loc[html_out.query("pMCC == 0 & ProbeScored == 'N'").index,'ProbeScored'] = 'Y'
            html_out.loc[html_out.query("dMCC == -2").index,'dMCC'] = ''
            html_out.loc[html_out.query("dMCC == 0 & DonorScored == 'N'").index,'DonorScored'] = 'Y'
            #write to index.html
            fname = os.path.join(outputRoot,'index.html')
            myf = open(fname,'w')
            myf.write(html_out.to_html(escape=False).encode('utf-8'))
            myf.write('\n')
            #write the query if manipulated
            if queryManipulation:
                myf.write("\nFiltered by query: {}\n".format(query))

            if average_df is not 0:
                #write title and then average_df
                a_df_copy = average_df.copy().round({'pNMM':3,'pMCC':3,'pBWL1':3,'pGWL1':3,'dNMM':3,'dMCC':3,'dBWL1':3,'dGWL1':3})
                myf.write('<h3>Average Scores</h3>\n')
                myf.write(a_df_copy.to_html().replace("text-align: right;","text-align: center;"))
            myf.close()

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

printq("Beginning the mask scoring report...")

mySysFile = os.path.join(args.sysDir,args.inSys)
myRefFile = os.path.join(myRefDir,args.inRef)

mySys = pd.read_csv(mySysFile,sep="|",header=0,dtype=sys_dtype,na_filter=False)
if args.optOut and not ('IsOptOut' in list(mySys)):
    print("ERROR: No IsOptOut column detected. Filtration is meaningless.")
    exit(1)

ref_dtype = {}
with open(myRefFile,'r') as ref:
    ref_dtype = {h:str for h in ref.readline().rstrip().split('|')} #treat it as string

myRef = pd.read_csv(myRefFile,sep="|",header=0,dtype=ref_dtype,na_filter=False)
myIndex = pd.read_csv(os.path.join(myRefDir,args.inIndex),sep="|",header=0,dtype=index_dtype,na_filter=False)

factor_mode = ''
query = ['']
if args.query:
    factor_mode = 'q'
    query = args.query
elif args.queryPartition:
    factor_mode = 'qp'
    query = [args.queryPartition]
elif args.queryManipulation:
    factor_mode = 'qm'
    query = args.queryManipulation

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
    printq("Precision should not be less than 1 for scores to be meaningful. Defaulting to 16 digits.")
    args.precision=16

#sub_ref = myRef[myRef['IsTarget']=="Y"].copy()
sub_ref = myRef.copy()

#update accordingly along with ProbeJournalJoin and JournalMask csv's in refDir
refpfx = os.path.join(myRefDir,args.inRef.split('.')[0])
#try/catch this
try:
    probeJournalJoin = pd.read_csv(refpfx + '-probejournaljoin.csv',sep="|",header=0,na_filter=False)
except IOError:
    print("No probeJournalJoin file is present. This run will terminate.")
    exit(1)

try:
    journalMask = pd.read_csv(refpfx + '-journalmask.csv',sep="|",header=0,na_filter=False)
except IOError:
    print("No journalMask file is present. This run will terminate.")
    exit(1)
    
# Merge the reference and system output for SSD/DSD reports
if args.task == 'manipulation':
    m_df = pd.merge(sub_ref, mySys, how='left', on='ProbeFileID')
    # get rid of inf values from the merge and entries for which there is nothing to work with.
    m_df = m_df.replace([np.inf,-np.inf],np.nan).dropna(subset=['OutputProbeMaskFileName'])
    #for all columns unique to mySys except ConfidenceScore, replace np.nan with empty string
    sysCols = list(mySys)
    refCols = list(sub_ref)
    sysCols = [c for c in sysCols if c not in refCols]
    sysCols.remove('ConfidenceScore')
    for c in sysCols:
        m_df.loc[pd.isnull(m_df[c]),c] = ''
    if args.indexFilter:
        printq("Filtering the reference and system output by index file...")
        m_df = pd.merge(myIndex[['ProbeFileID','ProbeWidth']],m_df,how='left',on='ProbeFileID').drop('ProbeWidth',1)
    m_df = m_df.query("IsTarget=='Y'")

    # if the confidence score are 'nan', replace the values with the mininum score
    m_df.loc[pd.isnull(m_df['ConfidenceScore']),'ConfidenceScore'] = mySys['ConfidenceScore'].min()
    # convert to the str type to the float type for computations
    m_df['ConfidenceScore'] = m_df['ConfidenceScore'].astype(np.float)

    journalData0 = pd.merge(probeJournalJoin[['ProbeFileID','JournalName']].drop_duplicates(),journalMask,how='left',on=['JournalName']).drop_duplicates()
    n_journals = len(journalData0)
    journalData0.index = range(n_journals)

    if args.queryManipulation:
        queryM = query
    else:
        queryM = ['']

    for qnum,q in enumerate(queryM):
        #journalData0 = journalMask.copy() #pd.merge(probeJournalJoin,journalMask,how='left',on=['JournalName','StartNodeID','EndNodeID'])
        journalData_df = pd.merge(probeJournalJoin,journalMask,how='left',on=['JournalName','StartNodeID','EndNodeID'])

        m_dfc = m_df.copy()
        if args.queryManipulation:
            journalData0['Evaluated'] = pd.Series(['N']*n_journals)
        else:
            journalData0['Evaluated'] = pd.Series(['Y']*n_journals) #add column for Evaluated: 'Y'/'N'

        #journalData = journalData0.copy()
        #use big_df to filter from the list as a temporary thing
        if q is not '':
            #exit if query does not match
            printq("Merging main data and journal data and querying the result...")
            try:
                big_df = pd.merge(m_df,journalData_df,how='left',on=['ProbeFileID','JournalName']).query(q) #TODO: test on sample with a print?
            except pd.computation.ops.UndefinedVariableError:
                print("The query '{}' doesn't seem to refer to a valid key. Please correct the query and try again.".format(q))
                exit(1)

            m_dfc = m_dfc.query("ProbeFileID=={}".format(np.unique(big_df.ProbeFileID).tolist()))
            #journalData = journalData.query("ProbeFileID=={}".format(list(big_df.ProbeFileID)))
            journalData_df = journalData_df.query("ProbeFileID=={}".format(list(big_df.ProbeFileID)))
            journalData0.loc[journalData0.reset_index().merge(big_df[['JournalName','StartNodeID','EndNodeID','ProbeFileID','ProbeMaskFileName']],\
                             how='left',on=['JournalName','StartNodeID','EndNodeID']).set_index('index').dropna().drop('ProbeMaskFileName',1).index,'Evaluated'] = 'Y'
        m_dfc.index = range(len(m_dfc))
            #journalData.index = range(0,len(journalData))

        #if get empty journalData or if no ProbeFileID's match between the two, there is nothing to be scored.
        if (len(journalData_df) == 0) or not (True in journalData_df['ProbeFileID'].isin(m_df['ProbeFileID']).unique()):
            print("The query '{}' yielded no journal data over which computation may take place.".format(q))
            continue

        outRootQuery = outRoot
        if len(queryM) > 1:
            outRootQuery = os.path.join(outRoot,'index_{}'.format(qnum)) #affix outRoot with qnum suffix for some length
            if not os.path.isdir(outRootQuery):
                os.system('mkdir ' + outRootQuery)
        m_dfc['Scored'] = ['Y']*len(m_dfc)
        printq("Beginning mask scoring...")
        r_df = createReport(m_dfc,journalData0, probeJournalJoin, myIndex, myRefDir, mySysDir,args.rbin,args.sbin,args.eks, args.dks, args.ntdks, args.kernel, outRootQuery, html=args.html,verbose=reportq,precision=args.precision)
        #get the manipulations that were not scored and set the same columns in journalData0 to 'N'
        pjoins = probeJournalJoin.query("ProbeFileID=={}".format(r_df.query('MCC == -2')['ProbeFileID'].tolist()))[['JournalName','StartNodeID','EndNodeID','ProbeFileID']]
        pjoins['Foo'] = pd.Series([0]*len(pjoins)) #dummy variable to be deleted later
#        p_idx = pjoins.reset_index().merge(journalData0,how='left',on=['ProbeFileID','JournalName','StartNodeID','EndNodeID']).set_index('index').dropna().drop('Color',1).index
        p_idx = journalData0.reset_index().merge(pjoins,how='left',on=['ProbeFileID','JournalName','StartNodeID','EndNodeID']).set_index('index').dropna().drop('Foo',1).index

        #set where the rows are the same in the join
        journalData0.loc[p_idx,'Evaluated'] = 'N'
        journalcols_else = list(journalData0)
        journalcols_else.remove('ProbeFileID')
#        journalcols = ['ProbeFileID','JournalName','StartNodeID','EndNodeID']
        #journalcols.extend(list(journalData0))
        #journalData0 = pd.merge(journalData0,probeJournalJoin[['ProbeFileID','JournalName','StartNodeID','EndNodeID']],how='right',on=['JournalName','StartNodeID','EndNodeID'])
        journalcols = ['ProbeFileID']
        journalcols.extend(journalcols_else)
        journalData0 = journalData0[journalcols]
        journalData0.to_csv(path_or_buf=os.path.join(outRootQuery,prefix + '-journalResults.csv'),sep="|",index=False)
    
#        r_df['Scored'] = pd.Series(['Y']*len(r_df))
#        r_df.loc[r_df.query('MCC == -2').index,'Scored'] = 'N'
        r_df.loc[r_df.query('MCC == -2').index,'NMM'] = ''
        r_df.loc[r_df.query('MCC == -2').index,'BWL1'] = ''
        r_df.loc[r_df.query('MCC == -2').index,'GWL1'] = ''
        #remove the rows that were not scored due to no region being present. We set those rows to have MCC == -2.
    
        #reorder r_df's columns. Names first, then scores, then other metadata
        rcols = r_df.columns.tolist()
        firstcols = ['TaskID','ProbeFileID','ProbeFileName','ProbeMaskFileName','IsTarget','OutputProbeMaskFileName','ConfidenceScore','NMM','MCC','BWL1','GWL1','Scored']
        metadata = [t for t in rcols if t not in firstcols]
        firstcols.extend(metadata)
        r_df = r_df[firstcols]
    
        a_df = 0
        #filter nan out of the below
        if len(r_df.query("Scored=='Y'").dropna()) == 0:
            #if nothing was scored, print a message and return
            print("None of the masks that we attempted to score for this run had regions to be scored. Further factor analysis is futile. This is not an error.")
        else:
            metrics = ['NMM','MCC','BWL1','GWL1']
            r_dfc = r_df.copy()
            r_dfc.loc[r_dfc.query('MCC == -2').index,'MCC'] = ''
            r_dfc.loc[r_dfc.query('MCC == -2').index,'Scored'] = 'N'
#            if args.queryManipulation:
#                my_partition = pt.Partition(r_dfc.query("Scored=='Y'"),q,factor_mode,metrics) #average over queries
#            else:
#                my_partition = pt.Partition(r_dfc.query("Scored=='Y'"),query,factor_mode,metrics) #average over queries
            my_partition = pt.Partition(r_dfc,q,factor_mode,metrics,verbose) #average over queries
            df_list = my_partition.render_table(metrics)
            
            if args.query and (len(df_list) > 0): #don't print anything if there's nothing to print
                #use Partition for OOP niceness and to identify file to be written.
                #a_df get the headers of temp_df and tack entries on one after the other
                a_df = pd.DataFrame(columns=df_list[0].columns) 
                for i,temp_df in enumerate(df_list):
                    temp_df.to_csv(path_or_buf="{}_{}.csv".format(os.path.join(outRootQuery,prefix + '-mask_scores'),i),sep="|",index=False)
                    a_df = a_df.append(temp_df,ignore_index=True)
                #at the same time do an optOut filter where relevant and save that
                if args.optOut:
                    my_partition_o = pt.Partition(r_dfc.query("Scored=='Y'"),["({}) & (IsOptOut!='Y')".format(q) for q in query],factor_mode,metrics,verbose) #average over queries
                    df_list_o = my_partition_o.render_table(metrics)
                    if len(df_list_o) > 0:
                        for i,temp_df_o in enumerate(df_list_o):
                            temp_df_o.to_csv(path_or_buf="{}_{}.csv".format(os.path.join(outRootQuery,prefix + '-mask_scores_optout'),i),sep="|",index=False)
                            a_df = a_df.append(temp_df_o,ignore_index=True)
            elif (args.queryPartition or (factor_mode == '') or (factor_mode == 'qm')) and (len(df_list) > 0):
                a_df = df_list[0]
                if len(a_df) > 0: 
                    #add optOut scoring in addition to (not replacing) the averaging procedure
                    if args.optOut:
                        if q == '':
                            my_partition_o = pt.Partition(r_dfc.query("Scored=='Y'"),"IsOptOut!='Y'",factor_mode,metrics,verbose) #average over queries
                        else:
                            my_partition_o = pt.Partition(r_dfc.query("Scored=='Y'"),"({}) & (IsOptOut!='Y')".format(q),factor_mode,metrics,verbose) #average over queries
                        df_list_o = my_partition_o.render_table(metrics)
                        if len(df_list_o) > 0:
                            a_df = a_df.append(df_list_o[0],ignore_index=True)
                    a_df.to_csv(path_or_buf=os.path.join(outRootQuery,prefix + "-mask_score.csv"),sep="|",index=False)
                else:
                    a_df = 0
    
        #generate HTML table report
        df2html(r_df,a_df,outRootQuery,args.queryManipulation,q)

        prefix = os.path.basename(args.inSys).split('.')[0]
        r_df.loc[r_df.query('MCC == -2').index,'Scored'] = 'N'
        r_df.loc[r_df.query('MCC == -2').index,'MCC'] = ''
        r_df.to_csv(path_or_buf=os.path.join(outRootQuery,prefix + '-mask_scores_perimage.csv'),sep="|",index=False)
    
#commenting out for the time being
#elif args.task in ['removal','clone']:
#    m_df = pd.merge(sub_ref, mySys, how='left', on='ProbeFileID')
#    # get rid of inf values from the merge and entries for which there is nothing to work with.
#    m_df = m_df.replace([np.inf,-np.inf],np.nan).dropna(subset=['ProbeMaskFileName'])
#
#    # if the confidence score are 'nan', replace the values with the mininum score
#    m_df.loc[pd.isnull(m_df['ConfidenceScore']),'ConfidenceScore'] = mySys['ConfidenceScore'].min()
#    # convert to the str type to the float type for computations
#    m_df['ConfidenceScore'] = m_df['ConfidenceScore'].astype(np.float)
#    r_df = createReportSSD(m_df, myRefDir, mySysDir,args.rbin,args.sbin,args.eks, args.dks, args.outRoot, html=args.html,verbose=reportq,precision=args.precision) # default eks 15, dks 9
#    a_df = avg_scores_by_factors_SSD(r_df,args.task,avglist,precision=args.precision)
#
elif args.task == 'splice':
    param_pfx = ['Probe','Donor']
    param_ids = [e + 'FileID' for e in param_pfx]
    m_df = pd.merge(sub_ref, mySys, how='left', on=param_ids)

    # get rid of inf values from the merge
    m_df = m_df.replace([np.inf,-np.inf],np.nan).dropna(subset=[e + 'MaskFileName' for e in param_pfx])
    #for all columns unique to mySys except ConfidenceScore, replace np.nan with empty string
    sysCols = list(mySys)
    refCols = list(sub_ref)
    sysCols = [c for c in sysCols if c not in refCols]
    sysCols.remove('ConfidenceScore')
    for c in sysCols:
        m_df.loc[pd.isnull(m_df[c]),c] = ''
    if args.indexFilter:
        printq("Filtering the reference and system output by index file...")
        m_df = pd.merge(myIndex[param_ids],m_df,how='left',on=param_ids)
    m_df = m_df.query("IsTarget=='Y'")

    # if the confidence score are 'nan', replace the values with the mininum score
    m_df.loc[pd.isnull(m_df['ConfidenceScore']),'ConfidenceScore'] = mySys['ConfidenceScore'].min()
    # convert to the str type to the float type for computations
    m_df['ConfidenceScore'] = m_df['ConfidenceScore'].astype(np.float)

    joinfields = param_ids+['JournalName']
    journalData0 = pd.merge(probeJournalJoin[joinfields].drop_duplicates(),journalMask,how='left',on=['JournalName']).drop_duplicates()
    n_journals = len(journalData0)
    journalData0.index = range(n_journals)

    if args.queryManipulation:
        queryM = query
    else:
        queryM = ['']

    for qnum,q in enumerate(queryM):
        m_dfc = m_df.copy()

        for param in param_pfx:
            if args.queryManipulation:
                journalData0[param+'Evaluated'] = pd.Series(['N']*n_journals)
            else:
                journalData0[param+'Evaluated'] = pd.Series(['Y']*n_journals)

        #use big_df to filter from the list as a temporary thing
        journalData_df = pd.merge(probeJournalJoin,journalMask,how='left',on=['JournalName','StartNodeID','EndNodeID'])
        #journalData = journalData0.copy()

        if q is not '':
            #exit if query does not match
            printq("Merging main data and journal data and querying the result...")
            try:
                big_df = pd.merge(m_df,journalData_df,how='left',on=param_ids).query(q)
            except pd.computation.ops.UndefinedVariableError:
                print("The query '{}' doesn't seem to refer to a valid key. Please correct the query and try again.".format(q))
                exit(1)

            #do a join with the big dataframe and filter out the stuff that doesn't show up by pairs
            m_dfc = pd.merge(m_dfc,big_df[param_ids],how='left',on=joinfields).dropna().drop('JournalName',1)
            #journalData = pd.merge(journalData0,big_df[['ProbeFileID','DonorFileID','JournalName']],how='left',on=['ProbeFileID','DonorFileID','JournalName'])
            journalData0.loc[journalData0.reset_index().merge(big_df[['JournalName','StartNodeID','EndNodeID','ProbeFileID','DonorFileID','ProbeMaskFileName']],\
                             how='left',on=['JournalName','StartNodeID','EndNodeID']).set_index('index').dropna().drop('ProbeMaskFileName',1).index,'ProbeEvaluated'] = 'Y'
            journalData0.loc[journalData0.reset_index().merge(big_df[['JournalName','StartNodeID','EndNodeID','ProbeFileID','DonorFileID','DonorMaskFileName']],\
                             how='left',on=['JournalName','StartNodeID','EndNodeID']).set_index('index').dropna().drop('DonorMaskFileName',1).index,'DonorEvaluated'] = 'Y'

        m_dfc.index = range(len(m_dfc))
            #journalData.index = range(0,len(journalData))

        #if no (ProbeFileID,DonorFileID) pairs match between the two, there is nothing to be scored.
        if len(pd.merge(m_df,journalData_df,how='left',on=param_ids)) == 0:
            print("The query '{}' yielded no journal data over which computation may take place.".format(q))
            continue

        outRootQuery = outRoot
        if len(queryM) > 1:
            outRootQuery = os.path.join(outRoot,'index_{}'.format(qnum)) #affix outRoot with qnum suffix for some length
            if not os.path.isdir(outRootQuery):
                os.system('mkdir ' + outRootQuery)
   
        m_dfc['Scored'] = ['Y']*len(m_dfc)
        printq("Beginning mask scoring...")
        r_df = createReport(m_dfc,journalData0, probeJournalJoin, myIndex, myRefDir, mySysDir,args.rbin,args.sbin,args.eks, args.dks, args.ntdks, args.kernel, outRootQuery, html=args.html,verbose=reportq,precision=args.precision)

        #set where the rows are the same in the join
        pjoins = probeJournalJoin.query("ProbeFileID=={}".format(r_df.query('pMCC == -2')['ProbeFileID'].tolist()))[['JournalName','StartNodeID','EndNodeID','ProbeFileID']]
        pjoins['Foo'] = pd.Series([0]*len(pjoins)) #dummy variable to be deleted later
        p_idx = journalData0.reset_index().merge(pjoins,how='left',on=['ProbeFileID','JournalName','StartNodeID','EndNodeID']).set_index('index').dropna().drop('Foo',1).index
#        p_idx = pjoins.reset_index().merge(journalData0,how='left',on=['ProbeFileID','JournalName','StartNodeID','EndNodeID']).set_index('index').dropna().drop('Color',1).index
        djoins = probeJournalJoin.query("DonorFileID=={}".format(r_df.query('dMCC == -2')['DonorFileID'].tolist()))[['JournalName','StartNodeID','EndNodeID','DonorFileID']]
        djoins['Foo'] = pd.Series([0]*len(djoins)) #dummy variable to be deleted later
        d_idx = journalData0.reset_index().merge(djoins,how='left',on=['DonorFileID','JournalName','StartNodeID','EndNodeID']).set_index('index').dropna().drop('Foo',1).index
#        d_idx = djoins.reset_index().merge(journalData0,how='left',on=['DonorFileID','JournalName','StartNodeID','EndNodeID']).set_index('index').dropna().drop('Color',1).index

        journalData0.loc[p_idx,'ProbeEvaluated'] = 'N'
        journalData0.loc[d_idx,'DonorEvaluated'] = 'N'
        journalcols = ['ProbeFileID','DonorFileID']
        journalcols_else = list(journalData0)
        journalcols_else.remove('ProbeFileID')
        journalcols_else.remove('DonorFileID')
        journalcols.extend(journalcols_else)

#        journalData0 = pd.merge(journalData0,probeJournalJoin[['ProbeFileID','DonorFileID','JournalName','StartNodeID','EndNodeID']],how='right',on=['JournalName','StartNodeID','EndNodeID'])
        journalData0 = journalData0[journalcols]
        journalData0.to_csv(path_or_buf=os.path.join(outRootQuery,prefix + '-journalResults.csv'),sep="|",index=False)
        a_df = 0

        #TODO: averaging procedure starts here
        p_idx = r_df.query('pMCC == -2').index
        d_idx = r_df.query('dMCC == -2').index
        r_df.loc[p_idx,'pNMM'] = ''
        r_df.loc[p_idx,'pBWL1'] = ''
        r_df.loc[p_idx,'pGWL1'] = ''
        r_df.loc[d_idx,'dNMM'] = ''
        r_df.loc[d_idx,'dBWL1'] = ''
        r_df.loc[d_idx,'dGWL1'] = ''
        #reorder r_df's columns. Names first, then scores, then other metadata
        rcols = r_df.columns.tolist()
        firstcols = ['TaskID','ProbeFileID','ProbeFileName','ProbeMaskFileName','DonorFileID','DonorFileName','DonorMaskFileName','IsTarget','OutputProbeMaskFileName','OutputDonorMaskFileName','ConfidenceScore','pNMM','pMCC','pBWL1','pGWL1','dNMM','dMCC','dBWL1','dGWL1']
        metadata = [t for t in rcols if t not in firstcols]
        firstcols.extend(metadata)
        r_df = r_df[firstcols]
   
        #filter here
        a_df = 0
        #filter nan out of the below
        if len(r_df.query("(ProbeScored == 'Y') | (DonorScored == 'Y')").dropna()) == 0:
            #if nothing was scored, print a message and return
            print("None of the masks that we attempted to score for this run had regions to be scored. Further factor analysis is futile. This is not an error.")
        else:
            metrics = ['pNMM','pMCC','pBWL1','pGWL1','dNMM','dMCC','dBWL1','dGWL1']
            r_dfc = r_df.copy()
            r_dfc.loc[p_idx,'ProbeScored'] = 'N'
            r_dfc.loc[d_idx,'DonorScored'] = 'N'
            #p_dummyscores = r_dfc.query("pMCC > -2")[metrics].mean(axis=0)
            #d_dummyscores = r_dfc.query("dMCC > -2")[metrics].mean(axis=0)
    
            #substitute for other values that won't get counted in the average
            r_dfc.loc[p_idx,'pNMM'] = np.nan #p_dummyscores['pNMM']
            r_dfc.loc[p_idx,'pBWL1'] = np.nan #p_dummyscores['pBWL1']
            r_dfc.loc[p_idx,'pGWL1'] = np.nan#$p_dummyscores['pGWL1']
            r_dfc.loc[d_idx,'dNMM'] = np.nan #d_dummyscores['dNMM']
            r_dfc.loc[d_idx,'dBWL1'] = np.nan #d_dummyscores['dBWL1']
            r_dfc.loc[d_idx,'dGWL1'] = np.nan #d_dummyscores['dGWL1']
            r_dfc.loc[p_idx,'pMCC'] = np.nan #p_dummyscores['pMCC']
            r_dfc.loc[d_idx,'dMCC'] = np.nan #d_dummyscores['dMCC']
#            if args.queryManipulation:
#                my_partition = pt.Partition(r_dfc,q,factor_mode,metrics) #average over queries
#            else:
#                my_partition = pt.Partition(r_dfc,query,factor_mode,metrics) #average over queries
            my_partition = pt.Partition(r_dfc,q,factor_mode,metrics,verbose) #average over queries
            df_list = my_partition.render_table(metrics)
        
            if args.query and (len(df_list) > 0): #don't print anything if there's nothing to print
                #use Partition for OOP niceness and to identify file to be written. 
                #a_df get the headers of temp_df and tack entries on one after the other
                a_df = pd.DataFrame(columns=df_list[0].columns) 
                for i,temp_df in enumerate(df_list):
                    temp_df.to_csv(path_or_buf="{}_{}.csv".format(os.path.join(outRootQuery,prefix + '-mask_scores'),i),sep="|",index=False)
                    a_df = a_df.append(temp_df,ignore_index=True)
                #at the same time do an optOut filter where relevant and save that
                if args.optOut:
                    my_partition_o = pt.Partition(r_dfc.query("Scored=='Y'"),["({}) & (IsOptOut!='Y')".format(q) for q in query],factor_mode,metrics,verbose) #average over queries
                    df_list_o = my_partition_o.render_table(metrics)
                    if len(df_list_o) > 0:
                        for i,temp_df_o in enumerate(df_list_o):
                            temp_df_o.to_csv(path_or_buf="{}_{}.csv".format(os.path.join(outRootQuery,prefix + '-mask_scores_optout'),i),sep="|",index=False)
                            a_df = a_df.append(temp_df_o,ignore_index=True)
                    
            elif (args.queryPartition or (factor_mode == '') or (factor_mode == 'qm')) and (len(df_list) > 0):
                a_df = df_list[0]
                if len(a_df) > 0:
                    #add optOut scoring in addition to (not replacing) the averaging procedure
                    if args.optOut:
                        if q == '':
                            my_partition_o = pt.Partition(r_dfc.query("Scored=='Y'"),"IsOptOut!='Y'",factor_mode,metrics,verbose) #average over queries
                        else:
                            my_partition_o = pt.Partition(r_dfc.query("Scored=='Y'"),"({}) & (IsOptOut!='Y')".format(q),factor_mode,metrics,verbose) #average over queries
                        df_list_o = my_partition_o.render_table(metrics)
                        if len(df_list_o) > 0:
                            a_df = a_df.append(df_list_o[0],ignore_index=True)
                    a_df.to_csv(path_or_buf=os.path.join(outRootQuery,prefix + "-mask_score.csv"),sep="|",index=False)
                else:
                    a_df = 0
            #TODO: averaging procedure ends here

        #generate HTML table report
        df2html(r_df,a_df,outRootQuery,args.queryManipulation,q)
    
        prefix = os.path.basename(args.inSys).split('.')[0]
        r_df.loc[r_df.query('pMCC == -2').index,'pMCC'] = ''
        r_df.loc[r_df.query('pMCC == -2').index,'ProbeScored'] = 'N'
        r_df.loc[r_df.query('dMCC == -2').index,'dMCC'] = ''
        r_df.loc[r_df.query('dMCC == -2').index,'DonorScored'] = 'N'
        r_df.to_csv(path_or_buf=os.path.join(outRootQuery,prefix + '-mask_scores_perimage.csv'),sep="|",index=False)

printq("Ending the mask scoring report.")

#if verbose and (a_df is not 0): #to avoid complications of print formatting when not verbose
#    precision = args.precision
#    if args.task == 'manipulation':
#        myavgs = [a_df[mets][0] for mets in ['NMM','MCC','BWL1','GWL1']]
#    
#        allmets = "Avg NMM: {}, Avg MCC: {}, Avg BWL1: {}, Avg GWL1: {}".format(round(myavgs[0],precision),
#                                                                 round(myavgs[1],precision),
#                                                                 round(myavgs[2],precision),
#                                                                 round(myavgs[3],precision))
#        printq(allmets)
#    
#    elif args.task == 'splice':
#        pavgs  = [a_df[mets][0] for mets in ['pNMM','pMCC','pBWL1','pGWL1']]
#        davgs  = [a_df[mets][0] for mets in ['dNMM','dMCC','dBWL1','dGWL1']]
#        pallmets = "Avg pNMM: {}, Avg pMCC: {}, Avg pBWL1: {}, Avg pGWL1: {}".format(round(pavgs[0],precision),
#                                                                     round(pavgs[1],precision),
#                                                                     round(pavgs[2],precision),
#                                                                     round(pavgs[3],precision))
#        dallmets = "Avg dNMM: {}, Avg dMCC: {}, Avg dBWL1: {}, Avg dGWL1: {}".format(round(davgs[0],precision),
#                                                                     round(davgs[1],precision),
#                                                                     round(davgs[2],precision),
#                                                                     round(davgs[3],precision))
#        printq(pallmets)
#        printq(dallmets)
#    else:
#        printerr("ERROR: Task not recognized.")

