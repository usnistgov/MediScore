#!/usr/bin/env python2
"""
* File: VideoSpatialLocalizationScorer.py
* Date: 10/31/2018
* Written by Daniel Zhou, under the guidance of Jonathan G. Fiscus
* Status: In Progress

* Description: This calculates performance scores for localizing mainpulated areas
               between reference masks and system output masks for videos.

* Requirements: This code requires the following packages:

    - h5py
    - opencv
    - pandas
    - numpy

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
import os
import shutil
import pandas as pd
import numpy as np
#import pdb #debug purposes
#from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import json
import configparser

this_dir = os.path.dirname(os.path.abspath(__file__))
# loading scoring and reporting libraries
lib_path = os.path.join(this_dir, "../../lib")
sys.path.append(lib_path)
from myround import myround

module_path = os.path.join(this_dir,'modules')
sys.path.append(module_path)
from perprobe_module import perprobe_module

img_scorer_path = os.path.join(this_dir,"../MaskScorer")
img_scorer_module_path = os.path.join(this_dir,"../MaskScorer/modules")
sys.path.append(img_scorer_path)
from MaskScorer import gen_average_fields
sys.path.append(img_scorer_module_path)
from average_report import average_report

def mkdir(dirname):
    if not os.path.isdir(dirname):
        os.system('mkdir {}'.format(dirname))

def read_csv(csv_name,sep="|",dtype=None):
    try:
        return pd.read_csv(csv_name,sep=sep,header=0,index_col=False,na_filter=False)
    except:
        print("Error: Expected file {}. File is not found. This run will terminate.".format(csv_name))
        exit(1)

def read_index(index_name,task):
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
    
    index_df = read_csv(index_name,dtype=index_dtype)
    return index_df

def mkdir(dirname):
    if not os.path.isdir(dirname):
        os.system('mkdir {}'.format(dirname))

#scoring scalar parameters dictionary
class params:
    def __init__(self,**kwds):
        self.__dict__.update(kwds)

if __name__ == '__main__':
    arg_list = ['task','refDir','sysDir','inRef','inSys','inIndex','outRoot','outMeta','outAllmeta',
                'query','queryPartition','queryManipulation','optOut','eks','dks','ntdks','kernel',
                'rbin','sbin','nspx','perProbePixelNoScore',
                'verbose','processors','precision','truncate_figures','displayScoredOnly','indexFilter','speedup','debug_off',
                'truncate','temporal_gt_only','temporal_scoring_only','collars','log']
    import argparse
    parser = argparse.ArgumentParser(description='Compute scores for the masks and generate a report.')
    parser.add_argument('-t','--task',type=str,default='manipulation',
        help='The task name. May add additional tasks in the future. Default: [manipulation]',metavar='character')
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

    parser.add_argument('--nspx',type=int,default=-1,
        help="Set a pixel value for all system output masks to serve as a no-score region [0,255]. -1 indicates that no particular pixel value will be chosen to be the no-score zone. [default=-1]",metavar='integer')
    parser.add_argument('-pppns','--perProbePixelNoScore',action='store_true',
        help="Use the pixel values in the ProbeOptOutPixelValue column (DonorOptOutPixelValue as well for the splice task) of the system output to designate no-score zones. This value will override the value set for the global no-score pixel.")
    
    #parser.add_argument('--avgOver',type=str,default='',
    #    help="A collection of features to average reports over, separated by commas.", metavar="character")
    parser.add_argument('-v','--verbose',type=int,default=None,
        help="Control print output. Select 1 to print all non-error print output and 0 to suppress all print output (bar argument-parsing errors).",metavar='0 or 1')
    parser.add_argument('-procs','--processors',type=int,default=1,
        help="The number of processors to use in the computation. Choosing too many processors will cause the program to forcibly default to a smaller number. [default=1].",metavar='positive integer')

    #TODO: need a better number rounding control interface. Or just deprecate or remove --truncate_figures.
    parser.add_argument('--precision',type=int,default=16,
        help="The number of digits to round computed scores. Note that rounding is not absolute, but is by significant digits (e.g. a score of 0.003333333333333... will round to 0.0033333 for a precision of 5). (default = 16).",metavar='positive integer')
    parser.add_argument('--truncate_figures',action='store_true',
        help="Truncate rather than round the figures to the specified precision. If no number is specified for precision, the default 16 will be used.")

#    parser.add_argument('-html',help="Output data to HTML files.",action="store_true")
    parser.add_argument('--displayScoredOnly',action='store_true',help="Display only the data for which a localized score could be generated.")
    parser.add_argument('-xF','--indexFilter',action='store_true',help="Filter scoring to only files that are present in the index file. This option permits scoring to select smaller index files for the purpose of testing.")
    parser.add_argument('--speedup',action='store_true',help="Run mask evaluation with a sped-up evaluator.")
    parser.add_argument('--debug_off',action='store_false',help="Continue running localization scorer on the next probe even when encountering errors. The errors will still be printed, but not raised.")

    #options from the VideoTemporalLocalizationScorer
    parser.add_argument('--truncate', help="Truncate any system intervals that goes beyond the video reference framecount (to the framecount value)", action='store_true')
    parser.add_argument('--temporal_gt_only',help="Score only on frames where there is temporal localization manipulation.")
    parser.add_argument('--temporal_scoring_only',help="Generate only the temporal localization metrics. ()")

    #TODO: to develop the options below
    parser.add_argument('-c', '--collars', help='collar value to add to each side of the reference intervals', default=None, type=int)
    parser.add_argument('--video_opt_out', help="Score taking in account the VideoFrameOptOutSegments field", action='store_true')
    parser.add_argument('-l','--log',type=str,default=None,help="Output to a log file. If not specified, will default to the directory of args.outRoot.")

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
    
    out_dir = os.path.dirname(args.outRoot)
    mkdir(out_dir)
    if not args.log:
        args.log = out_dir
    command_parameters = OrderedDict()
    command_parameters['command'] = " ".join(["python{}".format(sys.version[0])," ".join(sys.argv[:])]) #NOTE: artificial python command to ensure version is correct
    command_parameters["query"] = ""
    arg_dict = vars(args)
    for a in arg_list:
        command_parameters[a] = arg_dict[a]

    verbose = args.verbose
    if verbose:
        def printq(string):
            print(string)
    else:
        printq = lambda *a:None

    exit_status = 0

    #process query mode
    query_mode = ''
    avg_queries = '' 
    if args.queryManipulation != ['']:
        query_mode = 'qm'
    elif args.query:
        query_mode = 'q'
        avg_queries = args.query
    elif args.queryPartition:
        query_mode = 'qp'
        avg_queries = args.queryPartition

    is_multi_qm = len(args.queryManipulation) > 1
    
    #read in data
    sys_dir = os.path.join(args.sysDir,os.path.dirname(args.inSys))
    ref_dir = args.refDir
    task = args.task
    
    mySysFile = os.path.join(args.sysDir,args.inSys)
    sys_df = read_csv(mySysFile)
    
    index_df = read_index(os.path.join(ref_dir,args.inIndex),task=task)

    ref_df_name = os.path.join(ref_dir,args.inRef)
    ref_df = read_csv(ref_df_name)
    ref_pfx = ref_df_name[:-4]
    pjj_df_name = '-'.join([ref_pfx,'probejournaljoin.csv'])
    pjj_df = read_csv(pjj_df_name)
    journal_df_name = '-'.join([ref_pfx,'journalmask.csv'])
    journal_df = read_csv(journal_df_name)
    
    #initialize the scoring module
    scoring_module = perprobe_module(args.task,ref_df,pjj_df,journal_df,index_df,sys_df,ref_dir,sys_dir,args.rbin,args.sbin)
    out_pfx = os.path.basename(args.outRoot)
    avg_metric_fields,avg_constant_metric_fields = gen_average_fields('{}-video'.format(task))

    for i,q in enumerate(args.queryManipulation):
        output_directory = out_dir
        if query_mode == 'qm':
            if len(args.queryManipulation) > 1:
                output_directory = os.path.join(out_dir,"index_%d" % i)
            printq("Running video spatial localization scorer with query: {}".format(q))
        output_prefix = os.path.join(output_directory,out_pfx)
        mkdir(output_directory)

        command_parameters["query"] = q
        with open(os.path.join(output_directory,"parameters.json"),'w') as log_file:
            json.dump(command_parameters,log_file,indent=4)
            
        score_df,exit_status_run = scoring_module.score_all_masks(out_dir,
                                                  query=q,
                                                  query_mode = query_mode,
                                                  opt_out = args.optOut,
                                                  video_opt_out = args.video_opt_out,
                                                  truncate=args.truncate,
                                                  collars=args.collars,
                                                  temporal_gt_only=args.temporal_gt_only,
                                                  temporal_scoring_only=args.temporal_scoring_only,
                                                  eks=args.eks,
                                                  dks=args.dks,
                                                  ntdks=args.ntdks,
                                                  nspx=args.nspx,
                                                  pppns=args.perProbePixelNoScore,
                                                  kernel=args.kernel,
                                                  precision=args.precision,
                                                  verbose=args.verbose,
                                                  processors=args.processors
                                                 )
        exit_status |= exit_status_run

        score_df.to_csv("_".join([output_prefix,"pervideo.csv"]),sep="|",index=False)
        scoring_module.journal_join_df.to_csv("_".join([output_prefix,"journalResults.csv"]),sep="|",index=False)

        #average here with the relevant fields
        a_df = average_report(task,score_df,sys_df,avg_metric_fields,avg_constant_metric_fields,query_mode,avg_queries,output_prefix,optout=args.optOut,precision=args.precision,round_modes=['sd'])
#        a_df.to_csv("_".join([output_prefix,'.csv']),sep="|",index=False)

exit(exit_status)
