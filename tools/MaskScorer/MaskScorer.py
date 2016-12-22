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

# loading scoring and reporting libraries
lib_path = "../../lib"
#import masks
#import maskreport
import Partition_mask as f
execfile(os.path.join(lib_path,"masks.py")) #EDIT: find better way to import?
execfile('maskreport.py')


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
help="Directory root + file name to save outputs.",metavar='character')

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
if outRoot[-1]=='/':
    outRoot = outRoot[:-1]

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
    m_df = m_df.replace([np.inf,-np.inf],np.nan).dropna(subset=['ProbeMaskFileName'])

    # if the confidence score are 'nan', replace the values with the mininum score
    m_df.ix[pd.isnull(m_df['ConfidenceScore']),'ConfidenceScore'] = mySys['ConfidenceScore'].min()
    # convert to the str type to the float type for computations
    m_df['ConfidenceScore'] = m_df['ConfidenceScore'].astype(np.float)

    journalData = pd.merge(probeJournalJoin,journalMask,how='left',on=['JournalID','StartNodeID','EndNodeID'])

    if (args.targetManiType != 'all')
        journalData = journalData.query('Purpose=={}'.format(args.targetManiType.split(','))) #filter by targetManiType

    #m_df = pd.merge(m_df,probeJournalJoin,how='left',on='ProbeFileID')
    #m_df = pd.merge(journalMask,m_df,how='left',on='JournalID')

    #partition query here and filter further
    #selection = f.Partition(m_df,query,factor_mode)
    #DM_List = selection.part_dm_list
    #table_df = selection.render_table()
   
    if args.factor:
        #divided into separate tables for each query. A separate report for each.

        #TODO: generate list of temp dataframes here
        my_partition = pt.Partition(m_df,query,'f')

        temp_df_list = my_partition.render_table()

        for i,temp_df in enumerate(temp_df_list):
            # remember, default eks 15, dks 9

            #TODO: use Partition for OOP niceness and to identify file to be written. Existing partition customized for detection scorer, so customize.

            #Mask Scorer needs own Partition object. Should be in lib or maskreport? See Yooyoung.

            #TODO: get avgdict from Partition
            r_df = createReportSSD(temp_df,journalData, myRefDir, mySysDir,args.rbin,args.sbin,args.targetManiType,args.eks, args.dks, args.outRoot, html=args.html,verbose=reportq,precision=args.precision)
            a_df = avg_scores_by_factors_SSD(temp_df,args.task,avgdict,precision=args.precision)
            r_df.to_csv(path_or_buf='{}-perimage-{}.csv'.format(outRoot,query[i]),index=False)
            a_df.to_csv(path_or_buf="{}-{}.csv".format(outRoot,query[i]),index=False)
            
    elif args.factorp:

        #TODO: filter first

        #TODO: then use groupby

        r_df_fin = pd.read_csv('../../data/test_suite/maskScorerTests/ref_maskreport_3-perimage.csv') #read in the csv first so we can delete the rows later
        r_df_fin = r_df.drop(r_df.index[0:2])
        a_df_fin = pd.read_csv('../../data/test_suite/maskScorerTests/ref_maskreport_3.csv') #read in the csv first so we can delete the rows later
        a_df_fin = r_df.drop(r_df.index[0])

        #filter m_df and then use groupby to iterate
        m_df = m_df.query(args.factorp)

        #TODO: use a very general form of partition for string parsing. String parsed very well. Customize it to be more general though. Just like parsing the query string. And then combine with groupby for analysis.
        m_headers = list(m_df)
        parselist = args.factorp.replace(' ','').split('&') #remove spaces and parse by
        
        #TODO: instead just use simple parsing and the header list from the reference file?

        #TODO: and then append to both r_df_fin and a_df_fin. 
        
        r_df = createReportSSD(m_df,journalData myRefDir, mySysDir,args.rbin,args.sbin,args.targetManiType,args.eks, args.dks, args.outRoot, html=args.html,verbose=reportq,precision=args.precision) # default eks 15, dks 9
        avglist = avglist.replace(' ','') #delete extra spaces
        avglist = query.split('&')  #TODO: get from factor by query
        if avglist == ['']:
            avglist = []

        a_df = avg_scores_by_factors_SSD(r_df,args.task,avglist,precision=args.precision)

    else:
        #neither factors
        r_df = createReportSSD(m_df, myRefDir, mySysDir,args.rbin,args.sbin,args.targetManiType,args.eks, args.dks, args.outRoot, html=args.html,verbose=reportq,precision=args.precision)
        a_df = avg_scores_by_factors_SSD(m_df,args.task,avglist,precision=args.precision)

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
    r_df = createReportDSD(m_df, myRefDir, mySysDir,args.rbin,args.sbin,args.eks, args.dks, args.kern, args.outRoot, html=args.html,verbose=reportq,precision=args.precision)
    a_df = avg_scores_by_factors_DSD(r_df,args.task,avglist,precision=args.precision)

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

outRoot = args.outRoot
if outRoot[-1]=='/':
    outRoot = outRoot[:-1]

prefix = os.path.basename(args.inSys)
r_df.to_csv(path_or_buf=os.path.join(outRoot,prefix + '-score_perimage.csv'),index=False)
a_df.to_csv(path_or_buf=os.path.join(outRoot,prefix + "-score.csv"),index=False)
