#!/usr/bin/env python2
"""
* File: MaskScorer.py
* Date: 03/09/2018
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
import shutil
import cv2
import pandas as pd
import argparse
import numpy as np
#import maskreport as mr
#import pdb #debug purposes
#from abc import ABCMeta, abstractmethod
import configparser

this_dir = os.path.dirname(os.path.abspath(__file__))
# loading scoring and reporting libraries
#lib_path = "../../lib"
lib_path = os.path.join(this_dir, "../../lib")
sys.path.append(lib_path)
from myround import myround
import Partition_mask as pt

parser = argparse.ArgumentParser(description='Compute scores for the masks and generate a report.')
parser.add_argument('-t','--task',type=str,default='manipulation',
    help='Two different types of tasks: [manipulation] and [splice]',metavar='character')
parser.add_argument('--refDir',type=str,
    help='Dataset directory path: [e.g., ../../data/test_suite/maskscorertests]',metavar='character')
parser.add_argument('--sysDir',type=str,default='.',
    help='System output directory path: [e.g., ../../data/NC2016_Test]',metavar='character')
parser.add_argument('-r','--inRef',type=str,
    help='Reference csv file name: [e.g., reference/manipulation/NC2016-manipulation-ref.csv]',metavar='character')
parser.add_argument('-s','--inSys',type=str,
    help='System output csv file name: [e.g., ~/expid/system_output.csv]',metavar='character')
parser.add_argument('-x','--inIndex',type=str,
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
factor_group.add_argument('-qm', '--queryManipulation', nargs='*',default=[''],
    help="Filter the data by given queries before evaluation. Each query will result in a separate evaluation run.", metavar='character')

parser.add_argument('--optOut',action='store_true',help="Evaluate algorithm performance on a select number of trials determined by the performer via values in the ProbeStatus column.")
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
parser.add_argument('--sbin',type=int,default=-10,
    help="Binarize the system output mask to black and white with a numeric threshold in the interval [-1,255]. -1 can be chosen to binarize the entire mask to white. -10 indicates that the threshold for the mask will be chosen at the maximal absolute MCC value. [default=-10]",metavar='integer')
parser.add_argument('--jpeg2000',action='store_true',help="Evaluate JPEG2000 reference masks. Individual regions in the JPEG2000 masks may interserct; each pixel may contain multiple manipulations.")
parser.add_argument('--nspx',type=int,default=-1,
    help="Set a pixel value for all system output masks to serve as a no-score region [0,255]. -1 indicates that no particular pixel value will be chosen to be the no-score zone. [default=-1]",metavar='integer')
parser.add_argument('-pppns','--perProbePixelNoScore',action='store_true',
    help="Use the pixel values in the ProbeOptOutPixelValue column (DonorOptOutPixelValue as well for the splice task) of the system output to designate no-score zones. This value will override the value set for the global no-score pixel.")

#parser.add_argument('--avgOver',type=str,default='',
#    help="A collection of features to average reports over, separated by commas.", metavar="character")
parser.add_argument('-v','--verbose',type=int,default=None,
    help="Control print output. Select 1 to print all non-error print output and 0 to suppress all print output (bar argument-parsing errors).",metavar='0 or 1')
parser.add_argument('-p','--processors',type=int,default=1,
    help="The number of processors to use in the computation. Choosing too many processors will cause the program to forcibly default to a smaller number. [default=1].",metavar='positive integer')
parser.add_argument('--precision',type=int,default=16,
    help="The number of digits to round computed scores. Note that rounding is not absolute, but is by significant digits (e.g. a score of 0.003333333333333... will round to 0.0033333 for a precision of 5). (default = 16).",metavar='positive integer')
parser.add_argument('--truncate',action='store_true',
    help="Truncate rather than round the figures to the specified precision. If no number is specified for precision, the default 16 will be used.")
parser.add_argument('-html',help="Output data to HTML files.",action="store_true")
parser.add_argument('--displayScoredOnly',action='store_true',help="Display only the data for which a localized score could be generated.")
parser.add_argument('-xF','--indexFilter',action='store_true',help="Filter scoring to only files that are present in the index file. This option permits scoring to select smaller index files for the purpose of testing.")
parser.add_argument('--speedup',action='store_true',help="Run mask evaluation with a sped-up evaluator.")
parser.add_argument('--debug_off',action='store_false',help="Continue running localization scorer on the next probe even when encountering errors. The errors will still be printed, but not raised.")
#parser.add_argument('--cache_dir',type=str,default=None,
#help="The directory to cache reference mask data for future use. Subdirectories will be created according to specific details related to the task.",metavar='valid file directory')
#parser.add_argument('--cache_flush',action='store_true',help="Flush the cache directory before starting computation. This is especially crucial when the queryManipulation options are used in conjunction with --cache_dir.")
#TODO: add perimage file directory option? Skip perimage straight to averaging. Potentially applicable for optOut.
parser.add_argument('--scoreDir',type=str,
    help="The directory to the perimage file. Used when the perimage file has already been computed, but an average over a different averaging (non-qm) query needs to be computed, such as in the --optOut option.",metavar='valid file path')
#TODO: make the above option an alternate to system output options


args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    exit(0)

verbose = args.verbose

if verbose:
    def printq(string):
        print(string)
else:
    printq = lambda *a:None

def printerr(errmsg,verbose=None,exitcode=1):
    if verbose is not 0:
        parser.print_help()
        print(errmsg)
    exit(exitcode)

def mkdir(dirname):
    if not os.path.isdir(dirname):
        os.system('mkdir {}'.format(dirname))

#scoring scalar parameters dictionary
class params:
    def __init__(self,**kwds):
        self.__dict__.update(kwds)

perimage_params = params(optOut = args.optOut,
                         eks = args.eks,
                         dks = args.dks, 
                         ntdks = args.ntdks, 
                         kernel = args.kernel, 
                         sbin = args.sbin, 
                         jpeg2000 = args.jpeg2000,
                         indexFilter = args.indexFilter, 
                         nspx = args.nspx, 
                         pppns = args.perProbePixelNoScore, 
                         verbose = args.verbose, 
                         processors = args.processors,
                         precision = args.precision, 
                         truncate = args.truncate, 
                         speedup = args.speedup, 
                         debug_off = args.debug_off 
                         )

module_path = os.path.join(this_dir,'modules')
sys.path.append(module_path)
from perimage_report import localization_perimage_runner
from average_report import average_report
if args.html:
    from html_report import html_generator

outRoot = os.path.dirname(args.outRoot)
mkdir(outRoot)

#preprocessing
if args.task not in ['manipulation','splice']:
    printerr("ERROR: Localization task type must be 'manipulation' or 'splice'.")

for f_pair in [[args.refDir,'Dataset directory path'],[args.inRef,'Reference file path relative to test directory'],
               [args.inSys,'System output file path'],[args.inIndex,'Index file path'],
               [args.outRoot,'The folder name and file prefix for the output']]:
    f_name = f_pair[0]
    f_description = f_pair[1]
    if f_name in [None,'']:
        printerr("ERROR: {} must be supplied.".format(f_description),verbose)

sys_dir = os.path.join(args.sysDir,os.path.dirname(args.inSys))
ref_dir = args.refDir
task = args.task

if task == 'manipulation':
    index_dtype = {'TaskID':str,
             'ProbeFileID':str,
             'ProbeFileName':str,
             'ProbeWidth':np.int64,
             'ProbeHeight':np.int64}

elif task == 'splice':
    index_dtype = {'TaskID':str,
             'ProbeFileID':str,
             'ProbeFileName':str,
             'ProbeWidth':np.int64,
             'ProbeHeight':np.int64,
             'DonorFileID':str,
             'DonorFileName':str,
             'DonorWidth':np.int64,
             'DonorHeight':np.int64}

printq("Beginning the mask scoring report for submission {}...".format(args.inSys))

#TODO: if scoreDir is provided with a valid perimage file, skip all unnecessary file reads and preprocessing
mySysFile = os.path.join(args.sysDir,args.inSys)
sys_df = pd.read_csv(mySysFile,sep="|",header=0,na_filter=False)

sys_df.loc[pd.isnull(sys_df['ConfidenceScore']),'ConfidenceScore'] = sys_df['ConfidenceScore'].min()
sys_df['ConfidenceScore'] = sys_df['ConfidenceScore'].astype(np.float)

ref_df_name = os.path.join(ref_dir,args.inRef)
ref_df = pd.read_csv(ref_df_name,sep="|",header=0,na_filter=False)
ref_pfx = ref_df_name[:-4]
pjj_df_name = '-'.join([ref_pfx,'probejournaljoin.csv'])
journal_df_name = '-'.join([ref_pfx,'journalmask.csv'])
try:
    pjj_df = pd.read_csv(pjj_df_name,sep="|",header=0,na_filter=False)
except:
    print("Error: Expected probeJournalJoin file {}. File is not present. This run will terminate.".format(pjj_df_name))
    exit(1)
try:
    journal_df = pd.read_csv(journal_df_name,sep="|",header=0,na_filter=False)
except:
    print("Error: Expected journalMask file {}. File is not present. This run will terminate.".format(journal_df_name))
    exit(1)

index_df = pd.read_csv(os.path.join(ref_dir,args.inIndex),sep="|",header=0,dtype=index_dtype,na_filter=False)

#preprocessing starts here
manip_queries = args.queryManipulation

has_qm = manip_queries != ['']
is_multi_qm = len(manip_queries) > 1

#join stage
if task == 'manipulation':
    param_ids = ['ProbeFileID']
elif task == 'splice':
    param_ids = ['ProbeFileID','DonorFileID']
#    journaljoinfields = ['JournalName']
journaljoinfields = ['JournalName','StartNodeID','EndNodeID']

ref_and_sys = pd.merge(ref_df,sys_df,on=param_ids)
full_journal_df = pd.merge(pjj_df,journal_df,how='left',on=journaljoinfields)

eval_default = 'Y'
if has_qm:
    eval_default = 'N'

if task == 'splice':
    full_journal_df['ProbeEvaluated'] = eval_default
    full_journal_df['DonorEvaluated'] = eval_default
else:
    full_journal_df['Evaluated'] = eval_default

#filtration and cleanup
if args.indexFilter:
    printq("Filtering the reference and system output by index file...")
    ref_and_sys = pd.merge(index_df[param_ids + ['ProbeWidth']],ref_and_sys,how='left',on=param_ids).drop('ProbeWidth',1)

    printq("Filtering the journal data by index file...")
    id_field = 'ProbeFileID'
    if task == 'splice':
        id_field = 'ProbeDonorID'
        index_df[id_field] = index_df[param_ids].apply(lambda x: ':'.join(x),axis=1)
        full_journal_df[id_field] = full_journal_df[param_ids].apply(lambda x: ':'.join(x),axis=1)

#    index_df = index_df.query("{}=={}".format(id_field,full_journal_df[id_field].unique().tolist()))
    index_df = pd.merge(full_journal_df[[id_field,'JournalName']],index_df,how='left',on=id_field).drop('JournalName',1)
    full_journal_df = pd.merge(index_df[[id_field,'ProbeWidth']],full_journal_df,how='left',on=id_field).drop('ProbeWidth',1)

    if task == 'splice':
        index_df = index_df.drop(id_field,1)
        full_journal_df = full_journal_df.drop(id_field,1)

#check for optout columns
sys_cols = list(sys_df)
nc17_oo_name = 'IsOptOut'
mfc18_oo_name = 'ProbeStatus'
if not ((nc17_oo_name in sys_cols) or (mfc18_oo_name in sys_cols)):
    print("Error: Expected {} or {}. Neither column is found.".format(nc17_oo_name,mfc18_oo_name))
    exit(1)
oo_name = ''
if nc17_oo_name in sys_cols:
    oo_name = nc17_oo_name
    undesirables = str(['Y','Localization'])
    all_statuses = {'Y','N','Detection','Localization','FailedValidation'}
elif mfc18_oo_name in sys_cols:
    oo_name = mfc18_oo_name
    undesirables = str(['OptOutAll','OptOutLocalization'])
    all_statuses = {'Processed','NonProcessed','OptOutAll','OptOutDetection','OptOutLocalization','FailedValidation'}
probeStatuses = set(list(sys_df[oo_name].unique()))
if probeStatuses - all_statuses > set():
    print("ERROR: Status {} is not recognized for column {}.".format(probeStatuses - all_statuses,oo_name))
    exit(1)

if (task == 'splice') and (oo_name == mfc18_oo_name):
    donorStatuses = set(list(sys_df['DonorStatus'].unique()))
    all_donor_statuses = {'Processed','NonProcessed','OptOutLocalization','FailedValidation'}
    donor_diff_statuses = donorStatuses - all_donor_statuses
    if donor_diff_statuses != set():
        print("ERROR: Status {} is not recognized for column DonorStatus.".format(donor_diff_statuses))
        exit(1)

#prefilter the dataframe for simpler combination
if args.optOut:
    if nc17_oo_name in sys_cols:
        ref_and_sys = ref_and_sys.query("IsOptOut!={}".format(undesirables))
    elif mfc18_oo_name in sys_cols:
        if task == 'manipulation':
            ref_and_sys = ref_and_sys.query("ProbeStatus!={}".format(undesirables))
        elif task == 'splice':
            ref_and_sys = ref_and_sys.query("not ((ProbeStatus=={}) & (DonorStatus=={}))".format(undesirables,undesirables))

n_journals = full_journal_df.shape[0]
full_journal_df.index = range(n_journals)

output_directory = outRoot
output_prefix = os.path.basename(args.outRoot)

ref_and_sys = ref_and_sys.query("IsTarget=='Y'")

#assign fields for formatting outMeta output
header_fields = ['TaskID','ProbeFileID','ProbeFileName','OutputProbeMaskFileName']
detection_fields = ['IsTarget','ConfidenceScore']
metric_fields = [
'OptimumThreshold','OptimumNMM','OptimumMCC','OptimumBWL1',
'GWL1','AUC','EER',
'OptimumPixelTP','OptimumPixelTN','OptimumPixelFP','OptimumPixelFN',
'MaximumThreshold','MaximumNMM','MaximumMCC','MaximumBWL1',
'MaximumPixelTP','MaximumPixelTN','MaximumPixelFP','MaximumPixelFN',
'ActualThreshold','ActualNMM','ActualMCC','ActualBWL1',
'ActualPixelTP','ActualPixelTN','ActualPixelFP','ActualPixelFN',
'PixelN','PixelBNS','PixelSNS','PixelPNS'
]
if oo_name == nc17_oo_name:
    optout_fields = ['IsOptOut']
elif oo_name == mfc18_oo_name:
    optout_fields = ['ProbeStatus']
if task == 'splice':
    header_fields.extend(['DonorFileID','DonorFileName','OutputDonorMaskFileName'])
    if oo_name == mfc18_oo_name:
        optout_fields.append('DonorStatus')

avg_metric_fields = [
'OptimumThreshold','OptimumNMM','OptimumMCC','OptimumBWL1',
'GWL1','AUC','EER',
'MaximumNMM','MaximumMCC','MaximumBWL1',
'ActualNMM','ActualMCC','ActualBWL1'
]
avg_constant_metric_fields = ['MaximumThreshold','ActualThreshold',
                              'PixelAverageAUC','MaskAverageAUC']
if task == 'splice':
    avg_metric_fields = [ 'p%s' % m for m in avg_metric_fields ] + [ 'd%s' % m for m in avg_metric_fields ]
    avg_constant_metric_fields = [ 'p%s' % m for m in avg_constant_metric_fields ] + [ 'd%s' % m for m in avg_constant_metric_fields ]

factor_mode = ''
avg_queries = ''
if args.query:
    factor_mode = 'q'
    avg_queries = args.query
elif args.queryPartition:
    factor_mode = 'qp'
    avg_queries = args.queryPartition
elif args.queryManipulation:
    factor_mode = 'qm'

round_modes = ['sd']
if args.truncate:
    round_modes.append('t')

scorequery = "Scored=='Y'"
stackquery = scorequery
if task == 'splice':
    scorequery = "(ProbeScored=='Y') | (DonorScored=='Y')"

#scoring begins
metric_runner = localization_perimage_runner(task,ref_and_sys,ref_dir,sys_dir,args.rbin,args.sbin,full_journal_df,pjj_df,index_df,speedup=args.speedup,color=args.jpeg2000)
for qnum,q in enumerate(manip_queries):
    if is_multi_qm:
        output_directory = os.path.join(outRoot,'index_%d' % qnum)
        mkdir(output_directory)

    if args.scoreDir is None:
        #TODO: cache_directory goes here when needed to generate.

        #perimage run.
        scored_df,stack_df = metric_runner.score_all_masks(output_directory,q,perimage_params)
        if args.displayScoredOnly:
            scored_df = scored_df.query(scorequery)
        scored_df.to_csv(path_or_buf=os.path.join(output_directory,'%s_mask_scores_perimage.csv' % output_prefix),sep="|",index=False)
        #for bookkeeping purposes
        metric_runner.journal_display.to_csv(path_or_buf=os.path.join(output_directory,'%s_journalResults.csv' % output_prefix),sep="|",index=False)
        if stack_df is not 0:
            if args.displayScoredOnly:
                stack_df = stack_df.query(stackquery)
            if args.outMeta:
                outmeta_df = stack_df[header_fields + optout_fields + detection_fields + metric_fields]
                outmeta_df.to_csv(path_or_buf=os.path.join(output_directory,'%s_perimage-outMeta.csv' % output_prefix),sep="|",index=False)
            if args.outAllmeta:
                outallmeta_df = pd.merge(stack_df,index_df,how='left',on=['TaskID'] + [h for h in header_fields if 'Output' not in h ])
                outallmeta_df = pd.merge(outallmeta_df,metric_runner.journal_display,how='left',on=param_ids + ['JournalName'])
                outallmeta_df.to_csv(path_or_buf=os.path.join(output_directory,'%s_perimage-outAllmeta.csv' % output_prefix),sep="|",index=False)
    else:
        #TODO: write it in
        print "reading in score directory for output now"
        score_df = 0
        
    #average run. Output has already been covered.
    a_df = average_report(task,scored_df,avg_metric_fields,avg_constant_metric_fields,factor_mode,avg_queries,os.path.join(output_directory,output_prefix),optout=args.optOut,precision=args.precision,round_modes=round_modes)
     
    #if applicable, HTML generation
    if args.html:
        journal_data_name = os.path.join(output_directory,'%s_journalResults.csv' % output_prefix)
        journal_df = pd.read_csv(journal_data_name,sep="|",na_filter=False,header=0)
        visual_report_generator = html_generator(task,scored_df,a_df,journal_df,index_df,ref_dir,sys_dir,os.path.join(output_directory,output_prefix),query=q,overwrite=True,usejpeg2000=args.jpeg2000)
        visual_report_generator.gen_report(perimage_params)

