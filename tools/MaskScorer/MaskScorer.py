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
help="Binarize the system output mask to black and white with a numeric threshold in the interval [0,255]. Pick -1 to choose the threshold for the mask at the maximal absolute MCC value. [default=-1]",metavar='integer')
#parser.add_argument('--avgOver',type=str,default='',
#help="A collection of features to average reports over, separated by commas.", metavar="character")
parser.add_argument('-v','--verbose',type=int,default=None,
help="Control print output. Select 1 to print all non-error print output and 0 to suppress all print output (bar argument-parsing errors).",metavar='0 or 1')
parser.add_argument('--precision',type=int,default=16,
help="The number of digits to round computed scores, [e.g. a score of 0.3333333333333... will round to 0.33333 for a precision of 5], [default=16].",metavar='positive integer')
parser.add_argument('-html',help="Output data to HTML files.",action="store_true")

#TODO: add option for OptOut

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

#TODO: write merge on multi-index for two dataframes

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

#define HTML functions here
if args.html:
    if args.task == 'manipulation':
        def df2html(df,average_df,outputRoot,queryManipulation,query):
            html_out = df.copy()
    
            #os.path.join doesn't seem to work with Pandas Series so just do a manual string addition
            if outputRoot[-1] == '/':
                outputRoot = outputRoot[:-1]
    
            #set links around the system output data frame files for images that are not NaN
            #html_out.ix[~pd.isnull(html_out['OutputProbeMaskFileName']),'ProbeFileName'] = '<a href="' + outputRoot + '/' + html_out.ix[~pd.isnull(html_out['OutputProbeMaskFileName']),'ProbeFileName'].str.split('/').str.get(-1).str.split('.').str.get(0) + '.html">' + html_out['ProbeFileName'] + '</a>'
            pd.set_option('display.max_colwidth',-1)
            html_out.loc[~pd.isnull(html_out['OutputProbeMaskFileName']) & (html_out['Scored'] == 'Y'),'ProbeFileName'] = '<a href="' + html_out.ix[~pd.isnull(html_out['OutputProbeMaskFileName']),'ProbeFileID'] + '/' + html_out.ix[~pd.isnull(html_out['OutputProbeMaskFileName']),'ProbeFileName'].str.split('/').str.get(-1).str.split('.').str.get(0) + '.html">' + html_out.ix[~pd.isnull(html_out['OutputProbeMaskFileName']),'ProbeFileName'].str.split('/').str.get(-1) + '</a>'
            html_out = html_out.round({'NMM':3,'MCC':3,'BWL1':3,'GWL1':3})

            #write to index.html
            fname = os.path.join(outputRoot,'index.html')
            myf = open(fname,'w')
            myf.write(html_out.to_html(escape=False).replace("text-align: right;","text-align: center;"))
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
        def df2html(df,average_df,outputRoot,queryManipulation,query):
            html_out = df.copy()
    
            #os.path.join doesn't seem to work with Pandas Series so just do a manual string addition
            if outputRoot[-1] == '/':
                outputRoot = outputRoot[:-1]
    
            #set links around the system output data frame files for images that are not NaN
            #html_out.ix[~pd.isnull(html_out['OutputProbeMaskFileName']),'ProbeFileName'] = '<a href="' + outputRoot + '/' + html_out.ix[~pd.isnull(html_out['OutputProbeMaskFileName']),'ProbeFileName'].str.split('/').str.get(-1).str.split('.').str.get(0) + '.html">' + html_out['ProbeFileName'] + '</a>'
            #html_out.ix[~pd.isnull(html_out['OutputDonorMaskFileName']),'DonorFileName'] = '<a href="' + outputRoot + '/' + html_out.ix[~pd.isnull(html_out['OutputDonorMaskFileName']),'DonorFileName'].str.split('/').str.get(-1).str.split('.').str.get(0) + '.html">' + html_out['DonorFileName'] + '</a>'
            pd.set_option('display.max_colwidth',-1)
            html_out.loc[~pd.isnull(html_out['OutputProbeMaskFileName']),'ProbeFileName'] = '<a href="' + html_out.ix[~pd.isnull(html_out['OutputProbeMaskFileName']),'ProbeFileID'] + '_' + html_out.ix[~pd.isnull(html_out['OutputProbeMaskFileName']),'DonorFileID'] + '/probe/' + html_out.ix[~pd.isnull(html_out['OutputProbeMaskFileName']),'ProbeFileName'].str.split('/').str.get(-1).str.split('.').str.get(0) + '.html">' + html_out.ix[~pd.isnull(html_out['OutputProbeMaskFileName']),'ProbeFileName'].str.split('/').str.get(-1) + '</a>'
            html_out.loc[~pd.isnull(html_out['OutputDonorMaskFileName']),'DonorFileName'] = '<a href="' + html_out.ix[~pd.isnull(html_out['OutputProbeMaskFileName']),'ProbeFileID'] + '_' + html_out.ix[~pd.isnull(html_out['OutputProbeMaskFileName']),'DonorFileID'] + '/donor/' + html_out.ix[~pd.isnull(html_out['OutputDonorMaskFileName']),'DonorFileName'].str.split('/').str.get(-1).str.split('.').str.get(0) + '.html">' + html_out.ix[~pd.isnull(html_out['OutputDonorMaskFileName']),'DonorFileName'].str.split('/').str.get(-1) + '</a>'
            html_out = html_out.round({'pNMM':3,'pMCC':3,'pBWL1':3,'pGWL1':3,'dNMM':3,'dMCC':3,'dBWL1':3,'dGWL1':3})

            #write to index.html
            fname = os.path.join(outputRoot,'index.html')
            myf = open(fname,'w')
            myf.write(html_out.to_html(escape=False))
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
    else:
        df2html = lambda *a:None
else:
    df2html = lambda *a:None

printq("Beginning the mask scoring report...")

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
myRefFile = os.path.join(myRefDir,args.inRef)

ref_dtype = {}
with open(myRefFile,'r') as ref:
    ref_dtype = {h:str for h in ref.readline().rstrip().split('|')} #treat it as string

myRef = pd.read_csv(myRefFile,sep='|',header=0,dtype=ref_dtype,na_filter=False)
mySys = pd.read_csv(mySysFile,sep='|',header=0,dtype=sys_dtype,na_filter=False)
myIndex = pd.read_csv(os.path.join(myRefDir,args.inIndex),sep='|',header=0,dtype=index_dtype,na_filter=False)

factor_mode = ''
query = ['']
if args.query:
    factor_mode = 'q'
    query = args.query
elif args.queryPartition:
    factor_mode = 'qp'
    query = [args.queryPartition]
elif args.queryManipulation:
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

sub_ref = myRef[myRef['IsTarget']=="Y"].copy()

#update accordingly along with ProbeJournalJoin and JournalMask csv's in refDir
refpfx = os.path.join(myRefDir,args.inRef.split('.')[0])
#try/catch this
try:
    probeJournalJoin = pd.read_csv(refpfx + '-probejournaljoin.csv',sep='|',header=0,na_filter=False)
except IOError:
    print("No probeJournalJoin file is present. This run will terminate.")
    exit(1)

try:
    journalMask = pd.read_csv(refpfx + '-journalmask.csv',sep='|',header=0,na_filter=False)
except IOError:
    print("No journalMask file is present. This run will terminate.")
    exit(1)
    
# Merge the reference and system output for SSD/DSD reports
if args.task == 'manipulation':
    m_df = pd.merge(sub_ref, mySys, how='left', on='ProbeFileID')
    # get rid of inf values from the merge and entries for which there is nothing to work with.
    m_df = m_df.replace([np.inf,-np.inf],np.nan).dropna(subset=['OutputProbeMaskFileName'])

    # if the confidence score are 'nan', replace the values with the mininum score
    m_df.loc[pd.isnull(m_df['ConfidenceScore']),'ConfidenceScore'] = mySys['ConfidenceScore'].min()
    # convert to the str type to the float type for computations
    m_df['ConfidenceScore'] = m_df['ConfidenceScore'].astype(np.float)

    journalData0 = pd.merge(probeJournalJoin[['ProbeFileID','DonorFileID','JournalID']],journalMask,how='left',on=['JournalID']).drop_duplicates()
    n_journals = len(journalData0)

    if args.queryManipulation:
        queryM = query
    else:
        queryM = ['']

    for qnum,q in enumerate(queryM):
        #journalData0 = journalMask.copy() #pd.merge(probeJournalJoin,journalMask,how='left',on=['JournalID','StartNodeID','EndNodeID'])
        journalData_df = pd.merge(probeJournalJoin,journalMask,how='left',on=['JournalID','StartNodeID','EndNodeID'])
        n_journals = len(journalData0)

        m_dfc = m_df.copy()
        if args.queryManipulation:
            journalData0['Evaluated'] = pd.Series(['N']*n_journals)
        else:
            journalData0['Evaluated'] = pd.Series(['Y']*n_journals) #add column for Evaluated: 'Y'/'N'

        #journalData = journalData0.copy()
        #use big_df to filter from the list as a temporary thing
        if q is not '':
            #exit if query does not match
            try:
                big_df = pd.merge(m_df,journalData_df,how='left',on=['ProbeFileID','JournalID']).query(q)
            except pd.computation.ops.UndefinedVariableError:
                print("The query '{}' doesn't seem to refer to a valid key. Please correct the query and try again.".format(q))
                exit(1)

            m_dfc = m_dfc.query("ProbeFileID=={}".format(np.unique(big_df.ProbeFileID).tolist()))
            #journalData = journalData.query("ProbeFileID=={}".format(list(big_df.ProbeFileID)))
            journalData_df = journalData_df.query("ProbeFileID=={}".format(list(big_df.ProbeFileID)))
            journalData0.loc[journalData0.reset_index().merge(big_df[['JournalID','StartNodeID','EndNodeID','ProbeFileID','ProbeMaskFileName']],\
                             how='left',on=['JournalID','StartNodeID','EndNodeID']).set_index('index').dropna().drop('ProbeMaskFileName',1).index,'Evaluated'] = 'Y'
            m_dfc.index = range(0,len(m_dfc))
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
    
        r_df = mr.createReportSSD(m_dfc,journalData0, probeJournalJoin, myIndex, myRefDir, mySysDir,args.rbin,args.sbin,args.eks, args.dks, args.ntdks, args.kernel, outRootQuery, html=args.html,verbose=reportq,precision=args.precision)
        #get the manipulations that were not scored and set the same columns in journalData0 to 'N'
        pjoins = probeJournalJoin.query("ProbeFileID=={}".format(r_df.query('MCC == -2')['ProbeFileID'].tolist()))[['JournalID','StartNodeID','EndNodeID','ProbeFileID','ProbeMaskFileName']]
        p_idx = journalData0.reset_index().merge(pjoins,how='left',on=['ProbeFileID','JournalID','StartNodeID','EndNodeID']).set_index('index').dropna().drop('ProbeMaskFileName',1).index

        #set where the rows are the same in the join
        journalData0.loc[p_idx,'Evaluated'] = 'N'
        journalcols = ['ProbeFileID','JournalID','StartNodeID','EndNodeID']
        #journalcols.extend(list(journalData0))
        #journalData0 = pd.merge(journalData0,probeJournalJoin[['ProbeFileID','JournalID','StartNodeID','EndNodeID']],how='right',on=['JournalID','StartNodeID','EndNodeID'])
        journalData0 = journalData0[journalcols]
        journalData0.to_csv(path_or_buf=os.path.join(outRootQuery,prefix + '-journalResults.csv'),index=False)
    
        r_df['Scored'] = pd.Series(['Y']*len(r_df))
        r_df.loc[r_df.query('MCC == -2').index,'Scored'] = 'N'
        r_df.loc[r_df.query('MCC == -2').index,'NMM'] = ''
        r_df.loc[r_df.query('MCC == -2').index,'BWL1'] = ''
        r_df.loc[r_df.query('MCC == -2').index,'GWL1'] = ''
        r_df.loc[r_df.query('MCC == -2').index,'MCC'] = ''
        #remove the rows that were not scored due to no region being present. We set those rows to have MCC == -2.
    
        #reorder r_df's columns. Names first, then scores, then other metadata
        rcols = r_df.columns.tolist()
        firstcols = ['TaskID','ProbeFileID','ProbeFileName','ProbeMaskFileName','IsTarget','OutputProbeMaskFileName','ConfidenceScore','NMM','MCC','BWL1','GWL1','Scored']
        metadata = [t for t in rcols if t not in firstcols]
        firstcols.extend(metadata)
        r_df = r_df[firstcols]
    
        a_df = 0
        if len(r_df.query("Scored=='Y'")) == 0:
            #if nothing was scored, print a message and return
            print("None of the masks that we attempted to score for this run had regions to be scored. Further factor analysis is futile. This is not an error.")
        else:
            metrics = ['NMM','MCC','BWL1','GWL1']
            my_partition = pt.Partition(r_df.query("Scored=='Y'"),query,factor_mode,metrics) #average over queries
            df_list = my_partition.render_table(metrics)
         
            if args.query and (len(df_list) > 0): #don't print anything if there's nothing to print
                #use Partition for OOP niceness and to identify file to be written.
                #a_df get the headers of temp_df and tack entries on one after the other
                a_df = pd.DataFrame(columns=df_list[0].columns) 
                for i,temp_df in enumerate(df_list):
                    temp_df.to_csv(path_or_buf="{}_{}.csv".format(os.path.join(outRootQuery,prefix + '-mask_scores'),i),index=False)
                    a_df = a_df.append(temp_df,ignore_index=True)
                    
            elif (args.queryPartition or (factor_mode == '')) and (len(df_list) > 0):
                a_df = df_list[0]
                if len(a_df) > 0:
                    a_df.to_csv(path_or_buf=os.path.join(outRootQuery,prefix + "-mask_score.csv"),index=False)
                else:
                    a_df = 0
    
        #generate HTML table report
        df2html(r_df,a_df,outRootQuery,args.queryManipulation,q)

        prefix = os.path.basename(args.inSys).split('.')[0]
        r_df.to_csv(path_or_buf=os.path.join(outRootQuery,prefix + '-mask_scores_perimage.csv'),index=False)
    
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
    m_df = pd.merge(sub_ref, mySys, how='left', on=['ProbeFileID','DonorFileID'])

    # get rid of inf values from the merge
    m_df = m_df.replace([np.inf,-np.inf],np.nan).dropna(subset=['ProbeMaskFileName',
                                                                'DonorMaskFileName'])
    # if the confidence score are 'nan', replace the values with the mininum score
    m_df.loc[pd.isnull(m_df['ConfidenceScore']),'ConfidenceScore'] = mySys['ConfidenceScore'].min()
    # convert to the str type to the float type for computations
    m_df['ConfidenceScore'] = m_df['ConfidenceScore'].astype(np.float)

    journalData0 = pd.merge(probeJournalJoin[['ProbeFileID','DonorFileID','JournalID']],journalMask,how='left',on=['JournalID']).drop_duplicates()
    n_journals = len(journalData0)

    if args.queryManipulation:
        queryM = query
    else:
        queryM = ['']

    for qnum,q in enumerate(queryM):
        m_dfc = m_df.copy()
        if args.queryManipulation:
            journalData0['ProbeEvaluated'] = pd.Series(['N']*n_journals)
            journalData0['DonorEvaluated'] = pd.Series(['N']*n_journals)
        else:
            journalData0['ProbeEvaluated'] = pd.Series(['Y']*n_journals) #add column for Evaluated: 'Y'/'N'
            journalData0['DonorEvaluated'] = pd.Series(['Y']*n_journals) #add column for Evaluated: 'Y'/'N'

        #use big_df to filter from the list as a temporary thing
        journalData_df = pd.merge(probeJournalJoin,journalMask,how='left',on=['JournalID','StartNodeID','EndNodeID'])
        #journalData = journalData0.copy()

        if q is not '':
            #exit if query does not match
            try:
                big_df = pd.merge(m_df,journalData_df,how='left',on=['ProbeFileID','DonorFileID']).query(q)
            except pd.computation.ops.UndefinedVariableError:
                print("The query '{}' doesn't seem to refer to a valid key. Please correct the query and try again.".format(q))
                exit(1)

            #do a join with the big dataframe and filter out the stuff that doesn't show up by pairs
            m_dfc = pd.merge(m_dfc,big_df[['ProbeFileID','DonorFileID']],how='left',on=['ProbeFileID','DonorFileID','JournalID']).dropna().drop('JournalID',1)
            #journalData = pd.merge(journalData0,big_df[['ProbeFileID','DonorFileID','JournalID']],how='left',on=['ProbeFileID','DonorFileID','JournalID'])
            journalData0.loc[journalData0.reset_index().merge(big_df[['JournalID','StartNodeID','EndNodeID','ProbeFileID','DonorFileID','ProbeMaskFileName']],\
                             how='left',on=['JournalID','StartNodeID','EndNodeID']).set_index('index').dropna().drop('ProbeMaskFileName',1).index,'ProbeEvaluated'] = 'Y'
            journalData0.loc[journalData0.reset_index().merge(big_df[['JournalID','StartNodeID','EndNodeID','ProbeFileID','DonorFileID','DonorMaskFileName']],\
                             how='left',on=['JournalID','StartNodeID','EndNodeID']).set_index('index').dropna().drop('DonorMaskFileName',1).index,'DonorEvaluated'] = 'Y'

            m_dfc.index = range(0,len(m_dfc))
            #journalData.index = range(0,len(journalData))

        #if no (ProbeFileID,DonorFileID) pairs match between the two, there is nothing to be scored.
        if len(pd.merge(m_df,journalData_df,how='left',on=['ProbeFileID','DonorFileID'])) == 0:
            print("The query '{}' yielded no journal data over which computation may take place.".format(q))
            continue

        outRootQuery = outRoot
        if len(queryM) > 1:
            outRootQuery = os.path.join(outRoot,'index_{}'.format(qnum)) #affix outRoot with qnum suffix for some length
            if not os.path.isdir(outRootQuery):
                os.system('mkdir ' + outRootQuery)
   
        r_df = mr.createReportDSD(m_dfc,journalData0,probeJournalJoin, myIndex, myRefDir, mySysDir,args.rbin,args.sbin,args.eks, args.dks, args.ntdks, args.kernel, outRootQuery, html=args.html,verbose=reportq,precision=args.precision)

        #set where the rows are the same in the join
        pjoins = probeJournalJoin.query("ProbeFileID=={}".format(r_df.query('pMCC == -2')['ProbeFileID'].tolist()))[['JournalID','StartNodeID','EndNodeID','ProbeFileID','ProbeMaskFileName']]
        djoins = probeJournalJoin.query("DonorFileID=={}".format(r_df.query('dMCC == -2')['DonorFileID'].tolist()))[['JournalID','StartNodeID','EndNodeID','DonorFileID','DonorMaskFileName']]
        p_idx = journalData0.reset_index().merge(pjoins,how='left',on=['ProbeFileID','JournalID','StartNodeID','EndNodeID']).set_index('index').dropna().drop('ProbeMaskFileName',1).index
        d_idx = journalData0.reset_index().merge(djoins,how='left',on=['DonorFileID','JournalID','StartNodeID','EndNodeID']).set_index('index').dropna().drop('DonorMaskFileName',1).index

        journalData0.loc[p_idx,'ProbeEvaluated'] = 'N'
        journalData0.loc[d_idx,'DonorEvaluated'] = 'N'
        journalcols = ['ProbeFileID','DonorFileID']
        journalcols.extend(list(journalData0))

#        journalData0 = pd.merge(journalData0,probeJournalJoin[['ProbeFileID','DonorFileID','JournalID','StartNodeID','EndNodeID']],how='right',on=['JournalID','StartNodeID','EndNodeID'])
        journalData0 = journalData0[journalcols]
        journalData0.to_csv(path_or_buf=os.path.join(outRootQuery,prefix + '-journalResults.csv'),index=False)
        a_df = 0
    
        #reorder r_df's columns. Names first, then scores, then other metadata
        rcols = r_df.columns.tolist()
        firstcols = ['TaskID','ProbeFileID','ProbeFileName','ProbeMaskFileName','DonorFileID','DonorFileName','DonorMaskFileName','IsTarget','OutputProbeMaskFileName','OutputDonorMaskFileName','ConfidenceScore','pNMM','pMCC','pBWL1','pGWL1','dNMM','dMCC','dBWL1','dGWL1']
        metadata = [t for t in rcols if t not in firstcols]
        firstcols.extend(metadata)
        r_df = r_df[firstcols]
    
        metrics = ['pNMM','pMCC','pBWL1','pGWL1','dNMM','dMCC','dBWL1','dGWL1']
        my_partition = pt.Partition(r_df,query,factor_mode,metrics) #average over queries
        df_list = my_partition.render_table(metrics)
    
        if args.query and (len(df_list) > 0): #don't print anything if there's nothing to print
            #use Partition for OOP niceness and to identify file to be written. 
            #a_df get the headers of temp_df and tack entries on one after the other
            a_df = pd.DataFrame(columns=df_list[0].columns) 
            for i,temp_df in enumerate(df_list):
                temp_df.to_csv(path_or_buf="{}_{}.csv".format(os.path.join(outRootQuery,prefix + '-mask_scores'),i),index=False)
                a_df = a_df.append(temp_df,ignore_index=True)
                
        elif (args.queryPartition or (factor_mode == '')) and (len(df_list) > 0):
            a_df = df_list[0]
            if len(a_df) > 0:
                a_df.to_csv(path_or_buf=os.path.join(outRootQuery,prefix + "-mask_score.csv"),index=False)
            else:
                a_df = 0
    
        #generate HTML table report
        df2html(r_df,a_df,outRootQuery,args.queryManipulation,q)
    
        prefix = os.path.basename(args.inSys).split('.')[0]
        r_df.to_csv(path_or_buf=os.path.join(outRootQuery,prefix + '-mask_scores_perimage.csv'),index=False)


if verbose and (a_df is not 0): #to avoid complications of print formatting when not verbose
    precision = args.precision
    if args.task in ['manipulation']:
        myavgs = [a_df[mets][0] for mets in ['NMM','MCC','BWL1','GWL1']]
    
        allmets = "Avg NMM: {}, Avg MCC: {}, Avg BWL1: {}, Avg GWL1: {}".format(round(myavgs[0],precision),
                                                                 round(myavgs[1],precision),
                                                                 round(myavgs[2],precision),
                                                                 round(myavgs[3],precision))
        printq(allmets)
    
    elif args.task == 'splice':
        pavgs  = [a_df[mets][0] for mets in ['pNMM','pMCC','pBWL1','pGWL1']]
        davgs  = [a_df[mets][0] for mets in ['dNMM','dMCC','dBWL1','dGWL1']]
        pallmets = "Avg pNMM: {}, Avg pMCC: {}, Avg pBWL1: {}, Avg pGWL1: {}".format(round(pavgs[0],precision),
                                                                     round(pavgs[1],precision),
                                                                     round(pavgs[2],precision),
                                                                     round(pavgs[3],precision))
        dallmets = "Avg dNMM: {}, Avg dMCC: {}, Avg dBWL1: {}, Avg dGWL1: {}".format(round(davgs[0],precision),
                                                                     round(davgs[1],precision),
                                                                     round(davgs[2],precision),
                                                                     round(davgs[3],precision))
        printq(pallmets)
        printq(dallmets)
    else:
        printerr("ERROR: Task not recognized.")

