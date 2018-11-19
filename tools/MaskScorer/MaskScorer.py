#!/usr/bin/env python2
"""
* File: MaskScorer.py
* Date: 11/7/2018
* Written by Daniel Zhou, under the guidance of Jonathan G. Fiscus
* Status: In Progress

* Description: This calculates performance scores for localizing mainpulated areas
               between reference masks and system output masks for images.

* Requirements: This code requires the following packages:

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
import pandas as pd
import numpy as np
#import pdb #debug purposes
#from abc import ABCMeta, abstractmethod
import json
from collections import OrderedDict
import configparser

this_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(this_dir,'modules')
sys.path.append(module_path)
from perimage_report import localization_perimage_runner,get_ordered_df_headers
from average_report import average_report
from html_report import html_generator

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

invariant_columns = ["TaskID","ProbeFileID","DonorFileID","IsTarget","ConfidenceScore",
                     "ProbeFileName","BinaryProbeMaskFileName","OutputProbeMaskFileName",
                     "DonorFileName","DonorMaskFileName","OutputDonorMaskFileName"]
metric_columns = ["OptimumThreshold","OptimumMCC","OptimumNMM","OptimumBWL1",
                  "GWL1","AUC","EER","PixelAverageAUC","MaskAverageAUC",
                  "OptimumPixelTP","OptimumPixelTN","OptimumPixelFP","OptimumPixelFN",
                  "PixelN","PixelBNS","PixelSNS","PixelPNS",
                  "MaximumThreshold","MaximumMCC","MaximumNMM","MaximumBWL1",
                  "MaximumPixelTP","MaximumPixelTN","MaximumPixelFP","MaximumPixelFN",
                  "ActualThreshold","ActualMCC","ActualNMM","ActualBWL1",
                  "ActualPixelTP","ActualPixelTN","ActualPixelFP","ActualPixelFN"]

def gen_average_fields(task):
    avg_metric_fields = [
        'OptimumThreshold','OptimumNMM','OptimumMCC','OptimumBWL1',
        'GWL1','AUC','EER',
        'MaximumNMM','MaximumMCC','MaximumBWL1',
        'ActualNMM','ActualMCC','ActualBWL1'
        ]
    avg_constant_metric_fields = ['MaximumThreshold','ActualThreshold']
    if task == 'splice':
        avg_metric_fields = [ 'p%s' % m for m in avg_metric_fields ] + [ 'd%s' % m for m in avg_metric_fields ]
        avg_constant_metric_fields = [ 'p%s' % m for m in avg_constant_metric_fields ] + [ 'd%s' % m for m in avg_constant_metric_fields ]
    return avg_metric_fields,avg_constant_metric_fields

def stack_splice_perimage(df,invariant_columns=['ProbeFileID','DonorFileID',"IsTarget"],variant_columns=[]):
    """
    *Description: translates the dataframe into a stacked splice dataframe for horizontal compactification.
      - invariant columns: columns that do not vary for probe or donor
      - variant columns: columns that do vary for probe or donor. Generally various metric columns, without the probe or donor prefix.
    """
    df_cols = df.columns.values.tolist()
    df_cols_variant = [c for c in df_cols if c not in invariant_columns]

    probe_variants = ["p{}".format(c) for c in variant_columns if "p{}".format(c) in df_cols_variant]
    probe_variants_other = ["Probe{}".format(c) for c in variant_columns if "Probe{}".format(c) in df_cols_variant]
    probe_variants_key = [c for c in df_cols if ("Probe" in c) and (c not in invariant_columns + probe_variants_other)]
    donor_variants = ["d{}".format(c) for c in variant_columns if "d{}".format(c) in df_cols_variant]
    donor_variants_other = ["Donor{}".format(c) for c in variant_columns if "Donor{}".format(c) in df_cols_variant]
    donor_variants_key = [c for c in df_cols if ("Donor" in c) and (c not in invariant_columns + donor_variants_other)]
    remaining_cols = [c for c in df_cols if c not in invariant_columns + probe_variants_key + probe_variants + probe_variants_other + donor_variants_key + donor_variants + donor_variants_other]

    probe_df = df[invariant_columns + probe_variants_key + probe_variants + probe_variants_other + remaining_cols].copy()
    probe_df.rename(columns={"p{}".format(c):c for c in variant_columns},inplace=True)
    probe_df.rename(columns={c:c.replace("Probe","") for c in probe_variants_other + probe_variants_key},inplace=True)
    probe_df["ScoredMask"] = "Probe"
    donor_df = df[invariant_columns + donor_variants_key + donor_variants + donor_variants_other + remaining_cols].copy()
    donor_df.rename(columns={"d{}".format(c):c for c in variant_columns},inplace=True)
    probe_df.rename(columns={c:c.replace("Donor","") for c in probe_variants_other + probe_variants_key},inplace=True)
    donor_df["ScoredMask"] = "Donor"

    nc17_oo_col = "IsOptOut"
    oo_cols = [nc17_oo_col if nc17_oo_col in df_cols else "Status"]
    
    df = pd.concat([probe_df,donor_df])
    ordered_headers = get_ordered_df_headers('splice:stack',df.columns.values.tolist(),oo_cols,variant_columns,[])
    df = df.sort_values(["ProbeFileID","DonorFileID","ScoredMask"],ascending=[True,True,False])[ordered_headers]
    return df

if __name__ == '__main__':
    arg_list = ['task','refDir','sysDir','inRef','inSys','inIndex','outRoot','outMeta','outAllmeta','jpeg2000',
                'query','queryPartition','queryManipulation','optOut','eks','dks','ntdks','kernel',
                'rbin','sbin','nspx','perProbePixelNoScore',
                'verbose','processors','precision','truncate_figures','displayScoredOnly','indexFilter','speedup','debug_off','html','log']

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
    
    parser.add_argument('--jpeg2000',action='store_true',help="Evaluate JPEG2000 reference masks. Individual regions in the JPEG2000 masks may interserct; each pixel may contain multiple manipulations.")
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

    parser.add_argument('-html',help="Output data to HTML files.",action="store_true")
    parser.add_argument('-l','--log',type=str,default=None,help="Output to a log file. If not specified, will default to the directory of args.outRoot.")

    #parser.add_argument('--cache_dir',type=str,default=None,
    #help="The directory to cache reference mask data for future use. Subdirectories will be created according to specific details related to the task.",metavar='valid file directory')
    #parser.add_argument('--cache_flush',action='store_true',help="Flush the cache directory before starting computation. This is especially crucial when the queryManipulation options are used in conjunction with --cache_dir.")
    #TODO: add perimage file directory option? Skip perimage straight to averaging. Potentially applicable for optOut.
#    parser.add_argument('--scoreDir',type=str,
#        help="The directory to the perimage file. Used when the perimage file has already been computed, but an average over a different averaging (non-qm) query needs to be computed, such as in the --optOut option.",metavar='valid file path')
    
    #TODO: make the above option an alternate to system output options
    
    args = parser.parse_args()
    
    out_dir = os.path.abspath(os.path.dirname(args.outRoot))
    mkdir(out_dir)
    out_pfx = os.path.basename(args.outRoot)

    command_parameters = OrderedDict()
    command_parameters['command'] = " ".join(["python{}".format(sys.version[0])," ".join(sys.argv[:])]) #NOTE: artificial python command to ensure version is correct
    command_parameters['query'] = ""
    arg_dict = vars(args)
    for a in arg_list:
        command_parameters[a] = arg_dict[a]

    if len(sys.argv) < 2:
        parser.print_help()
        exit(0)
    
    verbose = args.verbose
    if verbose:
        def printq(string):
            print(string)
    else:
        printq = lambda *a:None

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

    avg_metric_fields,avg_constant_metric_fields = gen_average_fields(task)
    
    #initialize the scoring module
    scoring_module = localization_perimage_runner(task,ref_df,pjj_df,journal_df,index_df,sys_df,ref_dir,sys_dir,args.rbin,args.sbin,debug_mode=args.debug_off)

    for i,q in enumerate(args.queryManipulation):
        output_directory = out_dir
        output_prefix = args.outRoot
        if is_multi_qm:
            output_directory = os.path.join(out_dir,"index_%d" % i)
            output_prefix = os.path.join(output_directory,out_pfx)
            printq("Running image localization scorer with query: {}".format(q))
        mkdir(output_directory)

        if not args.log:
            log_dir = os.path.join(output_directory,"parameters.json")
        command_parameters["query"] = q
        with open(log_dir,'w') as log_file:
            json.dump(command_parameters,log_file,indent=4)
        score_df = scoring_module.score_all_masks(output_directory,
                                                  query=q,
                                                  query_mode = query_mode,
                                                  opt_out = args.optOut,
                                                  usejpeg2000 = args.jpeg2000,
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

        if (task == 'splice') and args.outMeta:
            printq("Additionally producing stacked splice csv for ease of viewing.")
            #produce stacked splice output. Write a function to do this.
            stack_df = stack_splice_perimage(score_df.drop("ProbeMaskFileName",axis=1),invariant_columns=invariant_columns,variant_columns=metric_columns)
            stack_df.to_csv("_".join([output_prefix,"perimage_outMeta.csv"]),sep="|",index=False)

        perimage_filename = "_".join([output_prefix,"mask_scores_perimage.csv"])
        score_df.to_csv(perimage_filename,sep="|",index=False)
        scoring_module.output_journal_join_df("_".join([output_prefix,"journalResults.csv"]))

        #average run. Output has already been covered. #TODO: move output to here instead? Not realizable for args.query.
        a_df = average_report(task,score_df,sys_df,avg_metric_fields,avg_constant_metric_fields,query_mode,avg_queries,output_prefix,optout=args.optOut,precision=args.precision,round_modes=['sd'])
         
        #if applicable, HTML generation
        if args.html:
            #TODO: cache_directory most relevant here
            journal_data_name = '%s_journalResults.csv' % output_prefix
            journal_df = pd.read_csv(journal_data_name,sep="|",na_filter=False,header=0)
#            os.system("python2 modules/html_report.py -t {} -pi {} -avg {}_mask_score.csv -j {} --refDir {} -x {} --sysDir {} -oR {} --overwrite".format(task,
#                                                                                                                                                         perimage_filename,
#                                                                                                                                                         output_prefix,
#                                                                                                                                                         journal_data_name,
#                                                                                                                                                         ref_dir,
#                                                                                                                                                         args.inIndex,
#                                                                                                                                                         sys_dir,
#                                                                                                                                                         output_prefix
#                                                                                                                                                         ))

            visual_report_generator = html_generator(task,score_df,a_df,journal_df,index_df,ref_dir,sys_dir,output_prefix,query=q,overwrite=True,usejpeg2000=args.jpeg2000)
            #expand the perimage params
            visual_report_generator.gen_report(eks=args.eks,
                                               dks=args.dks,
                                               ntdks=args.ntdks,
                                               nspx=args.nspx,
                                               pppns=args.perProbePixelNoScore,
                                               kernel=args.kernel,
                                               processors=args.processors)

