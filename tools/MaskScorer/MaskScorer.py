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
from abc import ABCMeta, abstractmethod

# loading scoring and reporting libraries
#lib_path = "../../lib"
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib")
sys.path.append(lib_path)
from metricRunner import maskMetricRunner
import Partition_mask as pt
import Render
#import masks
#execfile(os.path.join(lib_path,"masks.py"))
#execfile('maskreport.py')

########### Temporary Variable ############################################################
localOptOutColName = "ProbeStatus"
pastOptOutColName = "IsOptOut"

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
parser.add_argument('-oR','--outRoot',type=str,
help="Directory root plus prefix to save outputs.",metavar='character')
parser.add_argument('--outMeta',action='store_true',help='Save the CSV file with the system scores with minimal metadata')
parser.add_argument('--outAllmeta',action='store_true',help='Save the CSV file with the system scores with all metadata')

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
parser.add_argument('--color',action='store_true',help="Evaluate colorized referenced masks. Individual regions in the colorized masks are identifiable by region and do not intersect.")
parser.add_argument('--nspx',type=int,default=-1,
help="Set a pixel value for all system output masks to serve as a no-score region [0,255]. -1 indicates that no particular pixel value will be chosen to be the no-score zone. [default=-1]",metavar='integer')
parser.add_argument('-pppns','--perProbePixelNoScore',action='store_true',
help="Use the pixel values in the OptOutPixel column of the system output to designate no-score zones.")

#parser.add_argument('--avgOver',type=str,default='',
#help="A collection of features to average reports over, separated by commas.", metavar="character")
parser.add_argument('-v','--verbose',type=int,default=None,
help="Control print output. Select 1 to print all non-error print output and 0 to suppress all print output (bar argument-parsing errors).",metavar='0 or 1')
parser.add_argument('-p','--processors',type=int,default=1,
help="The number of processors to use in the computation. Choosing too many processors will cause the program to forcibly default to a smaller number. [default=1].",metavar='positive integer')
parser.add_argument('--precision',type=int,default=16,
help="The number of digits to round computed scores, [e.g. a score of 0.3333333333333... will round to 0.33333 for a precision of 5], [default=16].",metavar='positive integer')
parser.add_argument('-html',help="Output data to HTML files.",action="store_true")
parser.add_argument('--optOut',action='store_true',help="Evaluate algorithm performance on a select number of trials determined by the performer via values in the ProbeStatus column.")
parser.add_argument('--displayScoredOnly',action='store_true',help="Display only the data for which a localized score could be generated.")
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
    printerr("ERROR: Localization task type must be 'manipulation' or 'splice'.")
if args.refDir is None:
    printerr("ERROR: Test directory path must be supplied.")

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

#generate plotjson options
detpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../DetectionScorer/plotJsonFiles')
if not os.path.isdir(detpath):
    os.system(' '.join(['mkdir',detpath]))
Render.gen_default_plot_options(path=os.path.join(detpath,'plot_options.json'))

#assume outRoot exists
if args.outRoot in [None,'']:
    printerr("ERROR: the folder name and prefix for outputs must be supplied.")

outdir=os.path.dirname(args.outRoot)
outpfx=os.path.basename(args.outRoot)

if not os.path.isdir(outdir):
    os.system(' '.join(['mkdir',outdir]))

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
mySys = pd.read_csv(mySysFile,sep="|",header=0,dtype=sys_dtype,na_filter=False)

ref_dtype = {}
myRefFile = os.path.join(myRefDir,args.inRef)
with open(myRefFile,'r') as ref:
    ref_dtype = {h:str for h in ref.readline().rstrip().split('|')} #treat it as string

myRef = pd.read_csv(myRefFile,sep="|",header=0,dtype=ref_dtype,na_filter=False)
#sub_ref = myRef[myRef['IsTarget']=="Y"].copy()
sub_ref = myRef
myIndex = pd.read_csv(os.path.join(myRefDir,args.inIndex),sep="|",header=0,dtype=index_dtype,na_filter=False)

param_pfx = ['Probe']
if args.task == 'splice':
    param_pfx = ['Probe','Donor']
param_ids = [''.join([e,'FileID']) for e in param_pfx]

m_df = pd.merge(sub_ref, mySys, how='left', on=param_ids)
# get rid of inf values from the merge and entries for which there is nothing to work with.
m_df = m_df.replace([np.inf,-np.inf],np.nan).dropna(subset=[''.join([e,'MaskFileName']) for e in param_pfx])

#for all columns unique to mySys except ConfidenceScore, replace np.nan with empty string
sysCols = list(mySys)
refCols = list(sub_ref)

if args.optOut and (not (localOptOutColName in sysCols) and not (pastOptOutColName in sysCols)):
    print("ERROR: No {} column detected. Filtration is meaningless.".format(localOptOutColName))
    exit(1)

sysCols = [c for c in sysCols if c not in refCols]
sysCols.remove('ConfidenceScore')

for c in sysCols:
    m_df.loc[pd.isnull(m_df[c]),c] = ''
if args.indexFilter:
    printq("Filtering the reference and system output by index file...")
    m_df = pd.merge(myIndex[param_ids + ['ProbeWidth']],m_df,how='left',on=param_ids).drop('ProbeWidth',1)

if len(m_df) == 0:
    print("ERROR: the system output data does not match with the index. Either one may be empty. Please validate again.")
    exit(1)

#apply to post-index filtering
totalTrials = len(m_df)
#NOTE: IsOptOut values can be any one of "Y", "N", "Detection", or "Localization"
#NOTE: ProbeStatus values can be any one of "Processed", "NonProcessed", "OptOutAll", "OptOutDetection", "OptOutLocalization"
optOutCol = localOptOutColName
if localOptOutColName in sysCols:
    undesirables = str(['NonProcessed','OptOutAll','OptOutLocalization'])
elif pastOptOutColName in sysCols:
    optOutCol = pastOptOutColName
    undesirables = str(['Y','Localization'])

optOutQuery = "==".join([optOutCol,undesirables])

totalOptOut = len(m_df.query(optOutQuery))
totalOptIn = totalTrials - totalOptOut
TRR = float(totalOptIn)/totalTrials
m_df = m_df.query("IsTarget=='Y'") #TODO: don't filter anymore, but see Jon first.

if args.perProbePixelNoScore and (('ProbeOptOutPixelValue' not in sysCols) or ((args.task == 'splice') and ('DonorOptOutPixelValue' not in sysCols))):
    if args.task == 'manipulation':
        print("ERROR: 'ProbeOptOutPixelValue' is not found in the columns of the system output.")
    elif args.task == 'splice':
        print("ERROR: 'ProbeOptOutPixelValue' or 'DonorOptOutPixelValue' is not found in the columns of the system output.")
    exit(1)

#opting out at the beginning
if args.optOut:
    m_df = m_df.query(" ".join(['not',optOutQuery]))

class loc_scoring_params:
    def __init__(self,
                 mode,
                 eks,
                 dks,
                 ntdks,
                 nspx,
                 pppns,
                 kernel,
                 verbose,
                 html,
                 precision,
                 processors):
        self.mode = mode
        self.eks = eks
        self.dks = dks
        self.ntdks = ntdks
        self.nspx = nspx
        self.pppns = pppns
        self.kernel = kernel
        self.verbose = verbose
        self.html = html
        self.precision = precision
        self.processors = processors

#define HTML functions here
df2html = lambda *a:None
if args.task == 'manipulation':
    def createReport(m_df, journalData, probeJournalJoin, index, refDir, sysDir, rbin, sbin,erodeKernSize, dilateKernSize,distractionKernSize, kern,outputRoot,html,color,verbose,precision):
        # if the confidence score are 'nan', replace the values with the mininum score
        #m_df[pd.isnull(m_df['ConfidenceScore'])] = m_df['ConfidenceScore'].min()
        # convert to the str type to the float type for computations
        #m_df['ConfidenceScore'] = m_df['ConfidenceScore'].astype(np.float)
    
        metricRunner = maskMetricRunner(m_df,args.refDir,mySysDir,args.rbin,args.sbin,journalData,probeJournalJoin,index,speedup=args.speedup,color=args.color)
        #revise this to outputRoot and loc_scoring_params
        params = loc_scoring_params(0,args.eks,args.dks,args.ntdks,args.nspx,args.perProbePixelNoScore,args.kernel,args.verbose,args.html,args.precision,args.processors)
        df = metricRunner.getMetricList(outputRoot,params)
#        df = metricRunner.getMetricList(args.eks,args.dks,args.ntdks,args.nspx,args.kernel,outputRoot,args.verbose,args.html,precision=args.precision,processors=args.processors)
        merged_df = pd.merge(m_df.drop('Scored',1),df,how='left',on='ProbeFileID')

        nonscore_df = merged_df.query("OptimumMCC == -2")
#        merged_df['Scored'] = pd.Series(['Y']*len(merged_df))
#        merged_df.loc[merged_df.query('MCC == -2').index,'Scored'] = 'N'
        midx = nonscore_df.index
        if len(midx) > 0:
            merged_df.loc[midx,'OptimumThreshold'] = np.nan
            merged_df.loc[midx,'OptimumNMM'] = np.nan
            merged_df.loc[midx,'OptimumBWL1'] = np.nan
            merged_df.loc[midx,'GWL1'] = np.nan
            merged_df.loc[midx,'AUC'] = np.nan
            merged_df.loc[midx,'EER'] = np.nan
            merged_df.loc[midx,'OptimumThreshold'] = np.nan
            merged_df.loc[midx,'OptimumPixelTP'] = np.nan
            merged_df.loc[midx,'OptimumPixelTN'] = np.nan
            merged_df.loc[midx,'OptimumPixelFP'] = np.nan
            merged_df.loc[midx,'OptimumPixelFN'] = np.nan
            merged_df.loc[midx,'PixelN'] = np.nan
            merged_df.loc[midx,'PixelBNS'] = np.nan
            merged_df.loc[midx,'PixelSNS'] = np.nan
            merged_df.loc[midx,'PixelPNS'] = np.nan
        #remove the rows that were not scored due to no region being present. We set those rows to have MCC == -2.
        if args.displayScoredOnly:
            #get the list of non-scored and delete them
            nonscore_df['ProbeFileID'].apply(lambda x: os.system('rm -rf {}'.format(os.path.join(outputRoot,x))))
#            nonscore_df['ProbeFileID'].apply(lambda x: os.system('echo {}'.format(os.path.join(outputRoot,x))))
            merged_df = merged_df.query('OptimumMCC > -2')
    
        #reorder merged_df's columns. Names first, then scores, then other metadata
        rcols = merged_df.columns.tolist()
        firstcols = ['TaskID','ProbeFileID','ProbeFileName','ProbeMaskFileName','IsTarget','OutputProbeMaskFileName','ConfidenceScore','OptimumThreshold','OptimumNMM','OptimumMCC','OptimumBWL1','GWL1','AUC','EER','Scored','PixelN','OptimumPixelTP','OptimumPixelTN','OptimumPixelFP','OptimumPixelFN','PixelBNS','PixelSNS','PixelPNS']
        if args.sbin >= 0:
            firstcols.extend(['MaximumThreshold','MaximumNMM','MaximumMCC','MaximumBWL1',
                              'MaximumPixelTP','MaximumPixelTN','MaximumPixelFP','MaximumPixelFN',
                              'ActualThreshold','ActualNMM','ActualMCC','ActualBWL1',
                              'ActualPixelTP','ActualPixelTN','ActualPixelFP','ActualPixelFN'])

        metadata = [t for t in rcols if t not in firstcols]
        firstcols.extend(metadata)
        merged_df = merged_df[firstcols]
    
        return merged_df

    def journalUpdate(probeJournalJoin,journalData,r_df):
        #get the manipulations that were not scored and set the same columns in journalData to 'N'
        pjoins = probeJournalJoin.query("ProbeFileID=={}".format(r_df.query('OptimumMCC == -2')['ProbeFileID'].tolist()))[['JournalName','StartNodeID','EndNodeID','ProbeFileID']]
        pjoins['Foo'] = pd.Series([0]*len(pjoins)) #dummy variable to be deleted later
#        p_idx = pjoins.reset_index().merge(journalData,how='left',on=['ProbeFileID','JournalName','StartNodeID','EndNodeID']).set_index('index').dropna().drop('Color',1).index
        p_idx = journalData.reset_index().merge(pjoins,how='left',on=['ProbeFileID','JournalName','StartNodeID','EndNodeID']).set_index('index').dropna().drop('Foo',1).index

        #set where the rows are the same in the join
        journalData.loc[p_idx,'Evaluated'] = 'N'
        journalcols_else = list(journalData)
        journalcols_else.remove('ProbeFileID')
#        journalcols = ['ProbeFileID','JournalName','StartNodeID','EndNodeID']
        #journalcols.extend(list(journalData))
        #journalData = pd.merge(journalData,probeJournalJoin[['ProbeFileID','JournalName','StartNodeID','EndNodeID']],how='right',on=['JournalName','StartNodeID','EndNodeID'])
        journalcols = ['ProbeFileID']
        journalcols.extend(journalcols_else)
        journalData = journalData[journalcols]
        journalData.to_csv(path_or_buf=os.path.join(outRootQuery,'_'.join([prefix,'journalResults.csv'])),sep="|",index=False)
        return 0

    #averaging procedure starts here.
    def averageByFactors(r_df,metrics,factor_mode,query): #TODO: next time pass in object of parameters instead of tacking on new ones every time
        if 'OptimumMCC' not in metrics:
            print("ERROR: OptimumMCC is not in the metrics provided.")
            return 1
        #filter nan out of the below
        metrics_to_be_scored = ['OptimumThreshold','OptimumMCC','OptimumNMM','OptimumBWL1','GWL1','AUC','EER']
        if args.sbin >= 0:
            metrics_to_be_scored.extend(['MaximumThreshold','MaximumMCC','MaximumNMM','MaximumBWL1',
                                         'ActualThreshold','ActualMCC','ActualNMM','ActualBWL1'])
        if r_df.query("Scored=='Y'")[metrics_to_be_scored].dropna().shape[0] == 0:
            #if nothing was scored, print a message and return
            print("None of the masks that we attempted to score for query {} had regions to be scored. Further factor analysis is futile.".format(query))
            return 0
        r_dfc = r_df.copy()
        r_idx = r_dfc.query('OptimumMCC == -2').index
        r_dfc.loc[r_idx,'Scored'] = 'N'
        r_dfc.loc[r_idx,'OptimumMCC'] = np.nan
        r_df_scored = r_dfc.query("Scored=='Y'")
        ScoreableTrials = len(r_df_scored)
        my_partition = pt.Partition(r_df_scored,query,factor_mode,metrics) #average over queries
        df_list = my_partition.render_table(metrics)
        if len(df_list) == 0:
            return 0
        
#        totalTrials = len(r_df)
#        totalOptOut = len(r_df.query("IsOptOut=='Y'"))
#        totalOptIn = totalTrials - totalOptOut
#        TRR = float(totalOptIn)/totalTrials

        a_df = 0
        if factor_mode == 'q': #don't print anything if there's nothing to print
            #use Partition for OOP niceness and to identify file to be written.
            #a_df get the headers of temp_df and tack entries on one after the other
            a_df = pd.DataFrame(columns=df_list[0].columns)
            for i,temp_df in enumerate(df_list):
                heads = list(temp_df)
                temp_df['TRR'] = TRR
                temp_df['totalTrials'] = totalTrials
                temp_df['ScoreableTrials'] = ScoreableTrials
                temp_df['totalOptIn'] = totalOptIn
                temp_df['totalOptOut'] = totalOptOut
                temp_df['optOutScoring'] = 'N'
                if args.optOut:
                    temp_df['optOutScoring'] = 'Y'

                temp_df['OptimumThreshold'] = temp_df['OptimumThreshold'].dropna().apply(lambda x: str(int(x)))
                if args.sbin >= 0:
                    temp_df['MaximumThreshold'] = temp_df['MaximumThreshold'].dropna().apply(lambda x: str(int(x)))
                    temp_df['ActualThreshold'] = temp_df['ActualThreshold'].dropna().apply(lambda x: str(int(x)))

                heads.extend(['TRR','totalTrials','ScoreableTrials','totalOptIn','totalOptOut','optOutScoring'])
                temp_df = temp_df[heads]
                temp_df.to_csv(path_or_buf="{}_{}.csv".format(os.path.join(outRootQuery,'_'.join([prefix,'mask_scores'])),i),sep="|",index=False)
                if temp_df is not 0:
                    temp_df['OptimumThreshold'] = temp_df['OptimumThreshold'].dropna().apply(lambda x: str(int(x)))
                    if args.sbin >= 0:
                        temp_df['MaximumThreshold'] = temp_df['MaximumThreshold'].dropna().apply(lambda x: str(int(x)))
                        temp_df['ActualThreshold'] = temp_df['ActualThreshold'].dropna().apply(lambda x: str(int(x)))
                a_df = a_df.append(temp_df,ignore_index=True)
                
            #at the same time do an optOut filter where relevant and save that
#            if args.optOut:
#                my_partition_o = pt.Partition(r_dfc.query("Scored=='Y'"),["({}) & (IsOptOut!='Y')".format(q) for q in query],factor_mode,metrics,verbose) #average over queries
#                df_list_o = my_partition_o.render_table(metrics)
#                if len(df_list_o) > 0:
#                    for i,temp_df_o in enumerate(df_list_o):
#                        heads = list(temp_df_o)
#                        temp_df_o['TRR'] = TRR
#                        temp_df_o['totalTrials'] = totalTrials
#                        temp_df_o['totalOptIn'] = totalOptIn
#                        temp_df_o['totalOptOut'] = totalOptOut
#                        heads.extend(['TRR','totalTrials','totalOptIn','totalOptOut'])
#                        temp_df_o = temp_df_o[heads]
#                        temp_df_o.to_csv(path_or_buf="{}_{}.csv".format(os.path.join(outRootQuery,prefix + '-mask_scores_optout'),i),sep="|",index=False)
#                        a_df = a_df.append(temp_df_o,ignore_index=True)
        elif (factor_mode == 'qp') or (factor_mode == '') or (factor_mode == 'qm'):
            a_df = df_list[0]
            if len(a_df) == 0:
                return 0
            #add optOut scoring in addition to (not replacing) the averaging procedure
#            if args.optOut:
#                if query == '':
#                    my_partition_o = pt.Partition(r_dfc.query("Scored=='Y'"),"IsOptOut!='Y'",factor_mode,metrics,verbose) #average over queries
#                else:
#                    my_partition_o = pt.Partition(r_dfc.query("(Scored=='Y') & (IsOptOut!='Y')"),"({}) & (IsOptOut!='Y')".format(query),factor_mode,metrics,verbose) #average over queries
#                df_list_o = my_partition_o.render_table(metrics)
#                if len(df_list_o) > 0:
#                    a_df = a_df.append(df_list_o[0],ignore_index=True)
            heads = list(a_df)
            a_df['TRR'] = TRR
            a_df['totalTrials'] = totalTrials
            a_df['ScoreableTrials'] = ScoreableTrials
            a_df['totalOptIn'] = totalOptIn
            a_df['totalOptOut'] = totalOptOut
            a_df['optOutScoring'] = 'N'
            if args.optOut:
                a_df['optOutScoring'] = 'Y'
            a_df['OptimumThreshold'] = a_df['OptimumThreshold'].dropna().apply(lambda x: str(int(x)))
            if args.sbin >= 0:
                a_df['MaximumThreshold'] = a_df['MaximumThreshold'].dropna().apply(lambda x: str(int(x)))
                a_df['ActualThreshold'] = a_df['ActualThreshold'].dropna().apply(lambda x: str(int(x)))
            heads.extend(['TRR','totalTrials','ScoreableTrials','totalOptIn','totalOptOut','optOutScoring'])
            a_df = a_df[heads]
            a_df.to_csv(path_or_buf=os.path.join(outRootQuery,"_".join([prefix,"mask_score.csv"])),sep="|",index=False)

        return a_df

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

            html_out = html_out.round({'OptimumNMM':3,'OptimumMCC':3,'OptimumBWL1':3,'GWL1':3})

            #final filtering
            html_out.loc[html_out.query("OptimumMCC == -2").index,'OptimumMCC'] = ''
            html_out.loc[html_out.query("OptimumMCC == 0 & Scored == 'N'").index,'Scored'] = 'Y'

            #write to index.html
            fname = os.path.join(outputRoot,'index.html')
            myf = open(fname,'w')

            #add other metrics where relevant
            if average_df is not 0:
                #write title and then average_df
                metriclist = {}
                for met in ['NMM','MCC','BWL1']:
                    metriclist[''.join(['Optimum',met])] = 3
                    if args.sbin >= 0:
                        metriclist[''.join(['Maximum',met])] = 3
                        metriclist[''.join(['Actual',met])] = 3
                for met in ['GWL1','AUC','EER']:
                    metriclist[met] = 3
                
                a_df_copy = average_df.copy().round(metriclist)
                myf.write('<h3>Average Scores</h3>\n')
                myf.write(a_df_copy.to_html().replace("text-align: right;","text-align: center;"))

            myf.write('<h3>Per Scored Trial Scores</h3>\n')
            myf.write(html_out.to_html(escape=False).replace("text-align: right;","text-align: center;").encode('utf-8'))
            myf.write('\n')
            #write the query if manipulated
            if queryManipulation:
                myf.write("\nFiltered by query: {}\n".format(query))

            myf.close()

elif args.task == 'splice':
    def createReport(m_df, journalData, probeJournalJoin, index, refDir, sysDir, rbin, sbin,erodeKernSize, dilateKernSize,distractionKernSize, kern,outputRoot,html,color,verbose,precision):
        #finds rows in index and sys which correspond to target reference
        #sub_index = index[sub_ref['ProbeFileID'].isin(index['ProbeFileID']) & sub_ref['DonorFileID'].isin(index['DonorFileID'])]
        #sub_sys = sys[sub_ref['ProbeFileID'].isin(sys['ProbeFileID']) & sub_ref['DonorFileID'].isin(sys['DonorFileID'])]
    
        # if the confidence score are 'nan', replace the values with the mininum score
        #m_df[pd.isnull(m_df['ConfidenceScore'])] = m_df['ConfidenceScore'].min()
        # convert to the str type to the float type for computations
        #m_df['ConfidenceScore'] = m_df['ConfidenceScore'].astype(np.float)
#        maskMetricRunner = mm.maskMetricList(m_df,refDir,sysDir,rbin,sbin,journalData,probeJournalJoin,index,mode=1)
        metricRunner = maskMetricRunner(m_df,args.refDir,mySysDir,args.rbin,args.sbin,journalData,probeJournalJoin,index,speedup=args.speedup,color=args.color)
#        probe_df = maskMetricRunner.getMetricList(erodeKernSize,dilateKernSize,0,kern,outputRoot,verbose,html,precision=precision)
        #TODO: temporary until we can evaluate color for the splice task
        params = loc_scoring_params(1,args.eks,args.dks,0,args.nspx,args.perProbePixelNoScore,args.kernel,args.verbose,args.html,args.precision,args.processors)
        probe_df = metricRunner.getMetricList(outputRoot,params)
#        probe_df = metricRunner.getMetricList(args.eks,args.dks,0,args.nspx,args.kernel,outputRoot,args.verbose,args.html,precision=args.precision,processors=args.processors)
    
#        maskMetricRunner = mm.maskMetricList(m_df,refDir,sysDir,rbin,sbin,journalData,probeJournalJoin,index,mode=2) #donor images
#        metricRunner = maskMetricRunner(m_df,args.refDir,mySysDir,args.rbin,args.sbin,journalData,probeJournalJoin,index,mode=2,speedup=args.speedup,color=args.color)
#        donor_df = maskMetricRunner.getMetricList(erodeKernSize,dilateKernSize,0,kern,outputRoot,verbose,html,precision=precision)
        params = loc_scoring_params(2,args.eks,args.dks,0,args.nspx,args.perProbePixelNoScore,args.kernel,args.verbose,args.html,args.precision,args.processors)
        donor_df = metricRunner.getMetricList(outputRoot,params)
#        donor_df = metricRunner.getMetricList(args.eks,args.dks,0,args.nspx,args.kernel,outputRoot,args.verbose,args.html,precision=args.precision,processors=args.processors)

        #make another dataframe here that's formatted distinctly from the first.
        stackp = probe_df.copy()
        stackd = donor_df.copy()
        stackp['DonorFileID'] = stackd['DonorFileID']
        stackd['ProbeFileID'] = stackp['ProbeFileID']
        stackp['ScoredMask'] = 'Probe'
        stackd['ScoredMask'] = 'Donor'
 
        stackdf = pd.concat([stackp,stackd],axis=0)
        stackmerge = pd.merge(stackdf,m_df.drop('Scored',1),how='left',on=['ProbeFileID','DonorFileID'])
        firstcols = ['TaskID','ProbeFileID','ProbeFileName','ProbeMaskFileName','DonorFileID','DonorFileName','DonorMaskFileName','IsTarget','OutputProbeMaskFileName','OutputDonorMaskFileName','ConfidenceScore','ScoredMask','OptimumThreshold','OptimumNMM','OptimumMCC','OptimumBWL1','GWL1','AUC','EER']
        if args.sbin >= 0:
            firstcols.extend(['MaximumThreshold','MaximumNMM','MaximumMCC','MaximumBWL1',
                              'ActualThreshold','ActualNMM','ActualMCC','ActualBWL1'])
        rcols = stackmerge.columns.tolist()
        metadata = [t for t in rcols if t not in firstcols]
        firstcols.extend(metadata)
        stackmerge = stackmerge[firstcols]

        sidx = stackmerge.query('OptimumMCC==-2').index
        stackmerge.loc[sidx,'Scored'] = 'N'
        stackmerge.loc[sidx,'OptimumThreshold'] = np.nan
        stackmerge.loc[sidx,'OptimumNMM'] = np.nan
        stackmerge.loc[sidx,'OptimumBWL1'] = np.nan
        stackmerge.loc[sidx,'GWL1'] = np.nan
        stackmerge.loc[sidx,'AUC'] = np.nan
        stackmerge.loc[sidx,'EER'] = np.nan
        stackmerge.loc[sidx,'OptimumMCC'] = np.nan
        
        #add other scores for case sbin >= 0
        probe_df.rename(index=str,columns={"OptimumNMM":"pOptimumNMM",
                                           "OptimumMCC":"pOptimumMCC",
                                           "OptimumBWL1":"pOptimumBWL1",
                                           "GWL1":"pGWL1",
                                           "AUC":"pAUC",
                                           "EER":"pEER",
                                           'OptimumThreshold':'pOptimumThreshold',
                                           'OptimumPixelTP':'pOptimumPixelTP',
                                           'OptimumPixelTN':'pOptimumPixelTN',
                                           'OptimumPixelFP':'pOptimumPixelFP',
                                           'OptimumPixelFN':'pOptimumPixelFN',
                                           "MaximumNMM":"pMaximumNMM",
                                           "MaximumMCC":"pMaximumMCC",
                                           "MaximumBWL1":"pMaximumBWL1",
                                           'MaximumThreshold':'pMaximumThreshold',
                                           'MaximumPixelTP':'pMaximumPixelTP',
                                           'MaximumPixelTN':'pMaximumPixelTN',
                                           'MaximumPixelFP':'pMaximumPixelFP',
                                           'MaximumPixelFN':'pMaximumPixelFN',
                                           "ActualNMM":"pActualNMM",
                                           "ActualMCC":"pActualMCC",
                                           "ActualBWL1":"pActualBWL1",
                                           'ActualThreshold':'pActualThreshold',
                                           'ActualPixelTP':'pActualPixelTP',
                                           'ActualPixelTN':'pActualPixelTN',
                                           'ActualPixelFP':'pActualPixelFP',
                                           'ActualPixelFN':'pActualPixelFN',
                                           'PixelAverageAUC':'pPixelAverageAUC',
                                           'MaskAverageAUC':'pMaskAverageAUC',
                                           'PixelN':'pPixelN',
                                           'PixelBNS':'pPixelBNS',
                                           'PixelSNS':'pPixelSNS',
                                           'PixelPNS':'pPixelPNS',
                                           "ColMaskFileName":"ProbeColMaskFileName",
                                           "AggMaskFileName":"ProbeAggMaskFileName",
                                           "Scored":"ProbeScored"},inplace=True)
    
        donor_df.rename(index=str,columns={"OptimumNMM":"dOptimumNMM",
                                           "OptimumMCC":"dOptimumMCC",
                                           "OptimumBWL1":"dOptimumBWL1",
                                           "GWL1":"dGWL1",
                                           "AUC":"dAUC",
                                           "EER":"dEER",
                                           'OptimumThreshold':'dOptimumThreshold',
                                           'OptimumPixelTP':'dOptimumPixelTP',
                                           'OptimumPixelTN':'dOptimumPixelTN',
                                           'OptimumPixelFP':'dOptimumPixelFP',
                                           'OptimumPixelFN':'dOptimumPixelFN',
                                           "MaximumNMM":"dMaximumNMM",
                                           "MaximumMCC":"dMaximumMCC",
                                           "MaximumBWL1":"dMaximumBWL1",
                                           'MaximumThreshold':'dMaximumThreshold',
                                           'MaximumPixelTP':'dMaximumPixelTP',
                                           'MaximumPixelTN':'dMaximumPixelTN',
                                           'MaximumPixelFP':'dMaximumPixelFP',
                                           'MaximumPixelFN':'dMaximumPixelFN',
                                           "ActualNMM":"dActualNMM",
                                           "ActualMCC":"dActualMCC",
                                           "ActualBWL1":"dActualBWL1",
                                           'ActualThreshold':'dActualThreshold',
                                           'ActualPixelTP':'dActualPixelTP',
                                           'ActualPixelTN':'dActualPixelTN',
                                           'ActualPixelFP':'dActualPixelFP',
                                           'ActualPixelFN':'dActualPixelFN',
                                           'PixelAverageAUC':'dPixelAverageAUC',
                                           'MaskAverageAUC':'dMaskAverageAUC',
                                           'PixelN':'dPixelN',
                                           'PixelBNS':'dPixelBNS',
                                           'PixelSNS':'dPixelSNS',
                                           'PixelPNS':'dPixelPNS',
                                           "ColMaskFileName":"DonorColMaskFileName",
                                           "AggMaskFileName":"DonorAggMaskFileName",
                                           "Scored":"DonorScored"},inplace=True)
    
        pd_df = pd.concat([probe_df,donor_df],axis=1)
        merged_df = pd.merge(m_df,pd_df,how='left',on=['ProbeFileID','DonorFileID']).drop('Scored',1)
        nonscore_df = merged_df.query('(pOptimumMCC == -2) and (dOptimumMCC == -2)')

        #reorder merged_df's columns. Names first, then scores, then other metadata
        p_idx = merged_df.query('pOptimumMCC == -2').index
        d_idx = merged_df.query('dOptimumMCC == -2').index
        rcols = merged_df.columns.tolist()
        firstcols = ['TaskID','ProbeFileID','ProbeFileName','ProbeMaskFileName','DonorFileID','DonorFileName','DonorMaskFileName','IsTarget','OutputProbeMaskFileName','OutputDonorMaskFileName','ConfidenceScore','pOptimumThreshold','pOptimumNMM','pOptimumMCC','pOptimumBWL1','pGWL1','pAUC','pEER','dOptimumThreshold','dOptimumNMM','dOptimumMCC','dOptimumBWL1','dGWL1','dAUC','dEER']
        if args.sbin >= 0:
            firstcols.extend(['pMaximumThreshold','pMaximumNMM','pMaximumMCC','pMaximumBWL1',
                              'pMaximumPixelTP','pMaximumPixelTN','pMaximumPixelFP','pMaximumPixelFN',
                              'dMaximumThreshold','dMaximumNMM','dMaximumMCC','dMaximumBWL1',
                              'dMaximumPixelTP','dMaximumPixelTN','dMaximumPixelFP','dMaximumPixelFN',
                              'pActualThreshold','pActualNMM','pActualMCC','pActualBWL1',
                              'pActualPixelTP','pActualPixelTN','pActualPixelFP','pActualPixelFN',
                              'dActualThreshold','dActualNMM','dActualMCC','dActualBWL1',
                              'dActualPixelTP','dActualPixelTN','dActualPixelFP','dActualPixelFN'])
        metadata = [t for t in rcols if t not in firstcols]
        firstcols.extend(metadata)
        merged_df = merged_df[firstcols]
  
        #account for other metrics
        if (len(p_idx) > 0) or (len(d_idx) > 0):
            metriclist = ['OptimumThreshold','OptimumNMM','OptimumMCC','OptimumBWL1','GWL1',
                          'OptimumPixelTP','OptimumPixelTN','OptimumPixelFP','OptimumPixelFN',
                          'PixelN','PixelBNS','PixelSNS','PixelPNS']
            if args.sbin >= 0:
                metriclist = metriclist + ['MaximumThreshold','MaximumNMM','MaximumMCC','MaximumBWL1',
                                           'MaximumPixelTP','MaximumPixelTN','MaximumPixelFP','MaximumPixelFN',
                                           'ActualThreshold','ActualNMM','ActualMCC','ActualBWL1',
                                           'ActualPixelTP','ActualPixelTN','ActualPixelFP','ActualPixelFN']
            
            for met in metriclist:
                merged_df.loc[p_idx,''.join(['p',met])] = np.nan
                merged_df.loc[d_idx,''.join(['d',met])] = np.nan

        if args.displayScoredOnly:
            #get the list of non-scored and delete them
            nonscore_df.apply(lambda x: os.system('rm -rf {}'.format(os.path.join(outputRoot,'_'.join(x['ProbeFileID'],x['DonorFileID'])))))
#            nonscore_df.apply(lambda x: os.system('echo {}'.format(os.path.join(outputRoot,'_'.join(x['ProbeFileID'],x['DonorFileID'])))))
            merged_df = merged_df.query('(pOptimumMCC > -2) and (dOptimumMCC > -2)')
        return merged_df,stackmerge

    def journalUpdate(probeJournalJoin,journalData,r_df):
        #set where the rows are the same in the join
        pjoins = probeJournalJoin.query("ProbeFileID=={}".format(r_df.query('pOptimumMCC == -2')['ProbeFileID'].tolist()))[['JournalName','StartNodeID','EndNodeID','ProbeFileID']]
        pjoins['Foo'] = pd.Series([0]*len(pjoins)) #dummy variable to be deleted later
        p_idx = journalData.reset_index().merge(pjoins,how='left',on=['ProbeFileID','JournalName','StartNodeID','EndNodeID']).set_index('index').dropna().drop('Foo',1).index
#        p_idx = pjoins.reset_index().merge(journalData,how='left',on=['ProbeFileID','JournalName','StartNodeID','EndNodeID']).set_index('index').dropna().drop('Color',1).index
        djoins = probeJournalJoin.query("DonorFileID=={}".format(r_df.query('dOptimumMCC == -2')['DonorFileID'].tolist()))[['JournalName','StartNodeID','EndNodeID','DonorFileID']]
        djoins['Foo'] = pd.Series([0]*len(djoins)) #dummy variable to be deleted later
        d_idx = journalData.reset_index().merge(djoins,how='left',on=['DonorFileID','JournalName','StartNodeID','EndNodeID']).set_index('index').dropna().drop('Foo',1).index
#        d_idx = djoins.reset_index().merge(journalData,how='left',on=['DonorFileID','JournalName','StartNodeID','EndNodeID']).set_index('index').dropna().drop('Color',1).index

        journalData.loc[p_idx,'ProbeEvaluated'] = 'N'
        journalData.loc[d_idx,'DonorEvaluated'] = 'N'
        journalcols = ['ProbeFileID','DonorFileID']
        journalcols_else = list(journalData)
        journalcols_else.remove('ProbeFileID')
        journalcols_else.remove('DonorFileID')
        journalcols.extend(journalcols_else)

#        journalData = pd.merge(journalData,probeJournalJoin[['ProbeFileID','DonorFileID','JournalName','StartNodeID','EndNodeID']],how='right',on=['JournalName','StartNodeID','EndNodeID'])
        journalData = journalData[journalcols]
        journalData.to_csv(path_or_buf=os.path.join(outRootQuery,'_'.join([prefix,'journalResults.csv'])),sep="|",index=False)
        return 0

    def averageByFactors(r_df,metrics,factor_mode,query):
        if ('pOptimumMCC' not in metrics) or ('dOptimumMCC' not in metrics):
            print("ERROR: pOptimumMCC or dOptimumMCC are not in the metrics.")
            return 1
        #filter nan out of the below
        metrics_to_be_scored = []
        for pfx in ['p','d']:
            for met in ['Threshold','MCC','NMM','BWL1']:
                metrics_to_be_scored.append(''.join([pfx,'Optimum',met]))
                if args.sbin >= 0:
                    metrics_to_be_scored.append(''.join([pfx,'Maximum',met]))
                    metrics_to_be_scored.append(''.join([pfx,'Actual',met]))
            for met in ['GWL1','AUC','EER']:
                 metrics_to_be_scored.append(''.join([pfx,met]))
        
        if r_df.query("(ProbeScored == 'Y') | (DonorScored == 'Y')")[metrics_to_be_scored].dropna().shape[0] == 0:
            #if nothing was scored, print a message and return
            print("None of the masks that we attempted to score for query {} had regions to be scored. Further factor analysis is futile.".format(query))
            return 0
        p_idx = r_df.query('pOptimumMCC == -2').index
        d_idx = r_df.query('dOptimumMCC == -2').index
        r_dfc = r_df.copy()
        r_dfc.loc[p_idx,'ProbeScored'] = 'N'
        r_dfc.loc[d_idx,'DonorScored'] = 'N'

        #substitute for other values that won't get counted in the average
        r_dfc.loc[p_idx,'pOptimumMCC'] = np.nan
        r_dfc.loc[d_idx,'dOptimumMCC'] = np.nan
        my_partition = pt.Partition(r_dfc.query("ProbeScored=='Y' | DonorScored=='Y'"),query,factor_mode,metrics) #average over queries
#            my_partition = pt.Partition(r_dfc,q,factor_mode,metrics,verbose) #average over queries
        df_list = my_partition.render_table(metrics)
        if len(df_list) == 0:
            return 0

#        totalTrials = len(r_df)
#        totalOptOut = len(r_df.query("IsOptOut=='Y'"))
#        totalOptIn = totalTrials - totalOptOut
#        TRR = float(totalOptIn)/totalTrials

        a_df = 0
        if factor_mode == 'q': #don't print anything if there's nothing to print
            #use Partition for OOP niceness and to identify file to be written. 
            #a_df get the headers of temp_df and tack entries on one after the other
            a_df = pd.DataFrame(columns=df_list[0].columns) 
            for i,temp_df in enumerate(df_list):
                heads = list(temp_df)
                temp_df['TRR'] = TRR
                temp_df['totalTrials'] = totalTrials
                temp_df['ScoreableProbeTrials'] = totalTrials - len(p_idx)
                temp_df['ScoreableDonorTrials'] = totalTrials - len(d_idx)
                temp_df['totalOptIn'] = totalOptIn
                temp_df['totalOptOut'] = totalOptOut
                temp_df['optOutScoring'] = 'N'
                if args.optOut:
                    temp_df['optOutScoring'] = 'Y'

                temp_df['pOptimumThreshold'] = temp_df['pOptimumThreshold'].dropna().apply(lambda x: str(int(x)))
                temp_df['dOptimumThreshold'] = temp_df['dOptimumThreshold'].dropna().apply(lambda x: str(int(x)))
                if args.sbin >= 0:
                    temp_df['pMaximumThreshold'] = temp_df['pMaximumThreshold'].dropna().apply(lambda x: str(int(x)))
                    temp_df['pActualThreshold'] = temp_df['pActualThreshold'].dropna().apply(lambda x: str(int(x)))
                    temp_df['dMaximumThreshold'] = temp_df['dMaximumThreshold'].dropna().apply(lambda x: str(int(x)))
                    temp_df['dActualThreshold'] = temp_df['dActualThreshold'].dropna().apply(lambda x: str(int(x)))

                heads.extend(['TRR','totalTrials','ScoreableProbeTrials','ScoreableDonorTrials','totalOptIn','totalOptOut','optOutScoring'])
                temp_df = temp_df[heads]
                temp_df.to_csv(path_or_buf="{}_{}.csv".format(os.path.join(outRootQuery,'_'.join([prefix,'mask_scores'])),i),sep="|",index=False)
                a_df = a_df.append(temp_df,ignore_index=True)
            #at the same time do an optOut filter where relevant and save that
#            if args.optOut:
#                my_partition_o = pt.Partition(r_dfc.query("ProbeScored=='Y' | DonorScored=='Y'"),["({}) & (IsOptOut!='Y')".format(q) for q in query],factor_mode,metrics,verbose) #average over queries
#                df_list_o = my_partition_o.render_table(metrics)
#                if len(df_list_o) > 0:
#                    for i,temp_df_o in enumerate(df_list_o):
#                        heads = list(temp_df_o)
#                        temp_df_o['TRR'] = TRR
#                        temp_df_o['totalTrials'] = totalTrials
#                        temp_df_o['totalOptIn'] = totalOptIn
#                        temp_df_o['totalOptOut'] = totalOptOut
#                        heads.extend(['TRR','totalTrials','totalOptIn','totalOptOut'])
#                        temp_df_o = temp_df_o[heads]
#                        temp_df_o.to_csv(path_or_buf="{}_{}.csv".format(os.path.join(outRootQuery,prefix + '-mask_scores_optout'),i),sep="|",index=False)
#                        a_df = a_df.append(temp_df_o,ignore_index=True)
                
        elif (factor_mode == 'qp') or (factor_mode == '') or (factor_mode == 'qm'):
            a_df = df_list[0]
            if len(a_df) == 0:
                return 0
            #add optOut scoring in addition to (not replacing) the averaging procedure
#            if args.optOut:
#                if query == '':
#                    my_partition_o = pt.Partition(r_dfc.query("ProbeScored=='Y' | DonorScored=='Y'"),"IsOptOut!='Y'",factor_mode,metrics,verbose) #average over queries
#                else:
#                    my_partition_o = pt.Partition(r_dfc.query("(ProbeScored=='Y' | DonorScored=='Y') & (IsOptOut!='Y')"),"({}) & (IsOptOut!='Y')".format(query),factor_mode,metrics,verbose) #average over queries
#                df_list_o = my_partition_o.render_table(metrics)
#                if len(df_list_o) > 0:
#                    a_df = a_df.append(df_list_o[0],ignore_index=True)
            heads = list(a_df)
            a_df['TRR'] = TRR
            a_df['totalTrials'] = totalTrials
            a_df['ScoreableProbeTrials'] = totalTrials - len(p_idx)
            a_df['ScoreableDonorTrials'] = totalTrials - len(d_idx)
            a_df['totalOptIn'] = totalOptIn
            a_df['totalOptOut'] = totalOptOut
            a_df['optOutScoring'] = 'N'
            if args.optOut:
                a_df['optOutScoring'] = 'Y'
            heads.extend(['TRR','totalTrials','ScoreableProbeTrials','ScoreableDonorTrials','totalOptIn','totalOptOut','optOutScoring'])
            a_df = a_df[heads]
            if a_df is not 0:
                a_df['pOptimumThreshold'] = a_df['pOptimumThreshold'].dropna().apply(lambda x: str(int(x)))
                a_df['dOptimumThreshold'] = a_df['dOptimumThreshold'].dropna().apply(lambda x: str(int(x)))
                if args.sbin >= 0:
                    a_df['pMaximumThreshold'] = a_df['pMaximumThreshold'].dropna().apply(lambda x: str(int(x)))
                    a_df['pActualThreshold'] = a_df['pActualThreshold'].dropna().apply(lambda x: str(int(x)))
                    a_df['dMaximumThreshold'] = a_df['dMaximumThreshold'].dropna().apply(lambda x: str(int(x)))
                    a_df['dActualThreshold'] = a_df['dActualThreshold'].dropna().apply(lambda x: str(int(x)))
            a_df.to_csv(path_or_buf=os.path.join(outRootQuery,'_'.join([prefix,"mask_score.csv"])),sep="|",index=False)

        return a_df

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

            html_out = html_out.round({'pOptimumNMM':3,'pOptimumMCC':3,'pOptimumBWL1':3,'pGWL1':3,'dOptimumNMM':3,'dOptimumMCC':3,'dOptimumBWL1':3,'dGWL1':3})

            html_out.loc[html_out.query("pOptimumMCC == -2").index,'pOptimumMCC'] = ''
            html_out.loc[html_out.query("dOptimumMCC == -2").index,'dOptimumMCC'] = ''
            html_out.loc[html_out.query("pOptimumMCC == 0 & ProbeScored == 'N'").index,'ProbeScored'] = 'Y'
            html_out.loc[html_out.query("dOptimumMCC == 0 & DonorScored == 'N'").index,'DonorScored'] = 'Y'
            #write to index.html
            fname = os.path.join(outputRoot,'index.html')
            myf = open(fname,'w')

            if average_df is not 0:
                #write title and then average_df
                metriclist = {}
                for pfx in ['p','d']:
                    for met in ['NMM','MCC','BWL1']:
                        metriclist[''.join([pfx,'Optimum',met])] = 3
                        if args.sbin >= 0:
                            metriclist[''.join([pfx,'Maximum',met])] = 3
                            metriclist[''.join([pfx,'Actual',met])] = 3
                    for met in ['GWL1','AUC','EER']:
                        metriclist[''.join([pfx,met])] = 3

                a_df_copy = average_df.copy().round(metriclist)
                myf.write('<h3>Average Scores</h3>\n')
                myf.write(a_df_copy.to_html().replace("text-align: right;","text-align: center;"))

            myf.write('<h3>Per Scored Trial Scores</h3>\n')
            myf.write(html_out.to_html(escape=False).encode('utf-8'))
            myf.write('\n')
            #write the query if manipulated
            if queryManipulation:
                myf.write("\nFiltered by query: {}\n".format(query))

            myf.close()

#TODO: basic data init-ing starts here
factor_mode = ''
query = '' #in a similar format to queryManipulation elements, since partition treats them similarly
if args.query:
    factor_mode = 'q'
    query = args.query #is a list of items
elif args.queryPartition:
    factor_mode = 'qp'
    query = args.queryPartition #is a singleton, so keep it as such
elif args.queryManipulation:
    factor_mode = 'qm'
    query = args.queryManipulation #is a list of items

## if the confidence score are 'nan', replace the values with the mininum score
#mySys[pd.isnull(mySys['ConfidenceScore'])] = mySys['ConfidenceScore'].min()
## convert to the str type to the float type for computations
#mySys['ConfidenceScore'] = mySys['ConfidenceScore'].astype(np.float)

outRoot = outdir
prefix = outpfx#os.path.basename(args.inSys).split('.')[0]

reportq = 0
if args.verbose:
    reportq = 1

if args.precision < 1:
    printq("Precision should not be less than 1 for scores to be meaningful. Defaulting to 16 digits.")
    args.precision=16

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
    
#TODO: basic data init-ing ends here

# Merge the reference and system output
if args.task == 'manipulation':
    #TODO: basic data cleanup
    # if the confidence score are 'nan', replace the values with the mininum score
    m_df.loc[pd.isnull(m_df['ConfidenceScore']),'ConfidenceScore'] = mySys['ConfidenceScore'].min()
    # convert to the str type to the float type for computations
    m_df['ConfidenceScore'] = m_df['ConfidenceScore'].astype(np.float)

#    journalData0 = pd.merge(probeJournalJoin[['ProbeFileID','JournalName']].drop_duplicates(),journalMask,how='left',on=['JournalName']).drop_duplicates()
    journalData0 = pd.merge(probeJournalJoin,journalMask,how='left',on=['JournalName','StartNodeID','EndNodeID'])
    n_journals = len(journalData0)
    journalData0.index = range(n_journals)
    #TODO: basic data cleanup ends here

    if factor_mode == 'qm':
        queryM = query
    else:
        queryM = ['']

    for qnum,q in enumerate(queryM):
        #journalData0 = journalMask.copy() #pd.merge(probeJournalJoin,journalMask,how='left',on=['JournalName','StartNodeID','EndNodeID'])
        journalData_df = pd.merge(probeJournalJoin,journalMask,how='left',on=['JournalName','StartNodeID','EndNodeID'])

        m_dfc = m_df.copy()
        if factor_mode == 'qm':
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
#            journalData_df = journalData_df.merge(big_df[['ProbeFileID','JournalName','StartNodeID','EndNodeID']],how='left',on=['ProbeFileID','JournalName','StartNodeID','EndNodeID'])
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
                os.system(' '.join(['mkdir',outRootQuery]))
        m_dfc['Scored'] = ['Y']*len(m_dfc)

        printq("Beginning mask scoring...")
        r_df = createReport(m_dfc,journalData0, probeJournalJoin, myIndex, myRefDir, mySysDir,args.rbin,args.sbin,args.eks, args.dks, args.ntdks, args.kernel, outRootQuery, html=args.html,color=args.color,verbose=reportq,precision=args.precision)

        #get the manipulations that were not scored and set the same columns in journalData0 to 'N'
        journalUpdate(probeJournalJoin,journalData0,r_df)
        
        metrics = ['OptimumThreshold','OptimumNMM','OptimumMCC','OptimumBWL1','GWL1','AUC','EER','PixelAverageAUC','MaskAverageAUC']
        if args.sbin >= 0:
            metrics.extend(['MaximumThreshold','MaximumNMM','MaximumMCC','MaximumBWL1',
                            'ActualThreshold','ActualNMM','ActualMCC','ActualBWL1'])

        a_df = 0
        if factor_mode == 'qm':
            a_df = averageByFactors(r_df,metrics,factor_mode,q)
        else:
            a_df = averageByFactors(r_df,metrics,factor_mode,query)

        # tack on PixelAverageAUC and MaskAverageAUC to a_df and remove from r_df
        r_df = r_df.drop(['PixelAverageAUC','MaskAverageAUC'],1)
        if args.sbin >= 0:
            r_df = r_df.drop(['MaximumThreshold','ActualThreshold'],1)

#        if a_df is not 0:
#            a_df['OptimumThreshold'] = a_df['OptimumThreshold'].dropna().apply(lambda x: str(int(x)))
#            if args.sbin >= 0:
#                a_df['MaximumThreshold'] = a_df['MaximumThreshold'].dropna().apply(lambda x: str(int(x)))
#                a_df['ActualThreshold'] = a_df['ActualThreshold'].dropna().apply(lambda x: str(int(x)))

        r_idx = r_df.query('OptimumMCC == -2').index
        if len(r_idx) > 0:
            r_df.loc[r_idx,'Scored'] = 'N'
            r_df.loc[r_idx,'OptimumNMM'] = ''
            r_df.loc[r_idx,'OptimumBWL1'] = ''
            r_df.loc[r_idx,'GWL1'] = ''

        #generate HTML table report
        df2html(r_df,a_df,outRootQuery,args.queryManipulation,q)

        r_df.loc[r_idx,'OptimumMCC'] = ''
        prefix = outpfx#os.path.basename(args.inSys).split('.')[0]

        #convert all pixel values to decimal-less strings
        pix2ints = ['OptimumThreshold','OptimumPixelTP','OptimumPixelFP','OptimumPixelTN','OptimumPixelFN',
                    'PixelN','PixelBNS','PixelSNS','PixelPNS']
        if args.sbin >= 0:
            pix2ints.extend(['MaximumPixelTP','MaximumPixelFP','MaximumPixelTN','MaximumPixelFN',
                             'ActualPixelTP','ActualPixelFP','ActualPixelTN','ActualPixelFN'])

        for pix in pix2ints:
            r_df[pix] = r_df[pix].dropna().apply(lambda x: str(int(x)))

        if args.outMeta:
            roM_df = r_df[['TaskID','ProbeFileID','ProbeFileName','OutputProbeMaskFileName','IsTarget','ConfidenceScore',optOutCol,'OptimumNMM','OptimumMCC','OptimumBWL1','GWL1']]
            roM_df.to_csv(path_or_buf=os.path.join(outRootQuery,'_'.join([prefix,'perimage-outMeta.csv'])),sep="|",index=False)
        if args.outAllmeta:
            #left join with index file and journal data
            rAM_df = pd.merge(r_df,myIndex,how='left',on=['TaskID','ProbeFileID','ProbeFileName'])
            rAM_df = pd.merge(rAM_df,journalData0,how='left',on=['ProbeFileID','JournalName'])
            rAM_df.to_csv(path_or_buf=os.path.join(outRootQuery,'_'.join([prefix,'perimage-allMeta.csv'])),sep="|",index=False)

        r_df.to_csv(path_or_buf=os.path.join(outRootQuery,'_'.join([prefix,'mask_scores_perimage.csv'])),sep="|",index=False)
    
elif args.task == 'splice':
    #TODO: basic data cleanup
    # if the confidence score are 'nan', replace the values with the mininum score
    m_df.loc[pd.isnull(m_df['ConfidenceScore']),'ConfidenceScore'] = mySys['ConfidenceScore'].min()
    # convert to the str type to the float type for computations
    m_df['ConfidenceScore'] = m_df['ConfidenceScore'].astype(np.float)

    joinfields = param_ids+['JournalName']
    journalData0 = pd.merge(probeJournalJoin[joinfields].drop_duplicates(),journalMask,how='left',on=['JournalName']).drop_duplicates()
    n_journals = len(journalData0)
    journalData0.index = range(n_journals)
    #TODO: basic data cleanup ends here

    if factor_mode == 'qm':
        queryM = query
    else:
        queryM = ['']

    eval_pfx = param_pfx[:]
    for qnum,q in enumerate(queryM):
        m_dfc = m_df.copy()

        for param in eval_pfx:
            if factor_mode == 'qm':
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

        m_dfc.index = range(m_dfc.shape[0])
            #journalData.index = range(0,len(journalData))

        #if no (ProbeFileID,DonorFileID) pairs match between the two, there is nothing to be scored.
        if len(pd.merge(m_df,journalData_df,how='left',on=param_ids)) == 0:
            print("The query '{}' yielded no journal data over which computation may take place.".format(q))
            continue

        outRootQuery = outRoot
        if len(queryM) > 1:
            outRootQuery = os.path.join(outRoot,'index_{}'.format(qnum)) #affix outRoot with qnum suffix for some length
            if not os.path.isdir(outRootQuery):
                os.system(' '.join(['mkdir',outRootQuery]))
   
        m_dfc['Scored'] = ['Y']*m_dfc.shape[0]

        printq("Beginning mask scoring...")
        r_df,stackdf = createReport(m_dfc,journalData0, probeJournalJoin, myIndex, myRefDir, mySysDir,args.rbin,args.sbin,args.eks, args.dks, args.ntdks, args.kernel, outRootQuery, html=args.html,color=args.color,verbose=reportq,precision=args.precision)
        journalUpdate(probeJournalJoin,journalData0,r_df)

        #filter here
        metrics = ['pOptimumThreshold','pOptimumNMM','pOptimumMCC','pOptimumBWL1','pGWL1','pAUC','pEER','pPixelAverageAUC','pMaskAverageAUC',
                   'dOptimumThreshold','dOptimumNMM','dOptimumMCC','dOptimumBWL1','dGWL1','dAUC','dEER','dPixelAverageAUC','dMaskAverageAUC']
        if args.sbin >= 0:
            metrics.extend(['pMaximumThreshold','pMaximumNMM','pMaximumMCC','pMaximumBWL1',
                            'dMaximumThreshold','dMaximumNMM','dMaximumMCC','dMaximumBWL1',
                            'pActualThreshold','pActualNMM','pActualMCC','pActualBWL1',
                            'dActualThreshold','dActualNMM','dActualMCC','dActualBWL1'])
        a_df = 0
        if factor_mode == 'qm':
            a_df = averageByFactors(r_df,metrics,factor_mode,q)
        else:
            a_df = averageByFactors(r_df,metrics,factor_mode,query)

        r_df = r_df.drop(['pPixelAverageAUC','pMaskAverageAUC','dPixelAverageAUC','dMaskAverageAUC'],1)
        if args.sbin >= 0:
            r_df = r_df.drop(['pMaximumThreshold','pActualThreshold','dMaximumThreshold','dActualThreshold'],1)

        #convert all to ints.
#        if a_df is not 0:
#            a_df['pOptimumThreshold'] = a_df['pOptimumThreshold'].dropna().apply(lambda x: str(int(x)))
#            a_df['dOptimumThreshold'] = a_df['dOptimumThreshold'].dropna().apply(lambda x: str(int(x)))
#            if args.sbin >= 0:
#                a_df['pMaximumThreshold'] = a_df['pMaximumThreshold'].dropna().apply(lambda x: str(int(x)))
#                a_df['pActualThreshold'] = a_df['pActualThreshold'].dropna().apply(lambda x: str(int(x)))
#                a_df['dMaximumThreshold'] = a_df['dMaximumThreshold'].dropna().apply(lambda x: str(int(x)))
#                a_df['dActualThreshold'] = a_df['dActualThreshold'].dropna().apply(lambda x: str(int(x)))

        r_df.loc[r_df.query('pOptimumMCC == -2').index,'ProbeScored'] = 'N'
        r_df.loc[r_df.query('pOptimumMCC == -2').index,'pOptimumNMM'] = ''
        r_df.loc[r_df.query('pOptimumMCC == -2').index,'pOptimumBWL1'] = ''
        r_df.loc[r_df.query('pOptimumMCC == -2').index,'pGWL1'] = ''
        r_df.loc[r_df.query('dOptimumMCC == -2').index,'DonorScored'] = 'N'
        r_df.loc[r_df.query('dOptimumMCC == -2').index,'dOptimumNMM'] = ''
        r_df.loc[r_df.query('dOptimumMCC == -2').index,'dOptimumBWL1'] = ''
        r_df.loc[r_df.query('dOptimumMCC == -2').index,'dGWL1'] = ''

        #generate HTML table report
        df2html(r_df,a_df,outRootQuery,args.queryManipulation,q)
    
        r_df.loc[r_df.query('pOptimumMCC == -2').index,'pOptimumMCC'] = ''
        r_df.loc[r_df.query('dOptimumMCC == -2').index,'dOptimumMCC'] = ''

        prefix = outpfx#os.path.basename(args.inSys).split('.')[0]

        #convert all pixel values to decimal-less strings
        pix2ints = ['pOptimumThreshold','pOptimumPixelTP','pOptimumPixelFP','pOptimumPixelTN','pOptimumPixelFN',
                    'pPixelN','pPixelBNS','pPixelSNS','pPixelPNS',
                    'dOptimumThreshold','dOptimumPixelTP','dOptimumPixelFP','dOptimumPixelTN','dOptimumPixelFN',
                    'dPixelN','dPixelBNS','dPixelSNS','dPixelPNS']
        if args.sbin >= 0:
            pix2ints.extend(['pMaximumPixelTP','pMaximumPixelFP','pMaximumPixelTN','pMaximumPixelFN',
                             'pActualPixelTP','pActualPixelFP','pActualPixelTN','pActualPixelFN',
                             'dMaximumPixelTP','dMaximumPixelFP','dMaximumPixelTN','dMaximumPixelFN',
                             'dActualPixelTP','dActualPixelFP','dActualPixelTN','dActualPixelFN'])

        for pix in pix2ints:
            r_df[pix] = r_df[pix].dropna().apply(lambda x: str(int(x)))

        #other reports of varying
        if args.outMeta:
            roM_df = stackdf[['TaskID','ProbeFileID','ProbeFileName','DonorFileID','DonorFileName','OutputProbeMaskFileName','OutputDonorMaskFileName','ScoredMask','IsTarget','ConfidenceScore',optOutCol,'OptimumNMM','OptimumMCC','OptimumBWL1','GWL1']]
            roM_df.to_csv(path_or_buf=os.path.join(outRootQuery,'_'.join([prefix,'perimage-outMeta.csv'])),sep="|",index=False)
        if args.outAllmeta:
            #left join with index file and journal data
            rAM_df = pd.merge(stackdf.copy(),myIndex,how='left',on=['TaskID','ProbeFileID','ProbeFileName','DonorFileID','DonorFileName'])
            rAM_df = pd.merge(rAM_df,journalData0,how='left',on=['ProbeFileID','DonorFileID','JournalName'])
            rAM_df.to_csv(path_or_buf=os.path.join(outRootQuery,'_'.join([prefix,'perimage-allMeta.csv'])),sep="|",index=False)

        r_df.to_csv(path_or_buf=os.path.join(outRootQuery,'_'.join([prefix,'mask_scores_perimage.csv'])),sep="|",index=False)

printq("Ending the mask scoring report.")
exit(0)

