"""
 *File: perimage_report.py
 *Date: 04/26/2018
 *Author: Daniel Zhou
 *Status: Complete

 *Description: this code contains the runner that runs the metrics over
               each pair of masks.

 *Disclaimer:
 This software was developed at the National Institute of Standards
 and Technology (NIST) by employees of the Federal Government in the
 course of their official duties. Pursuant to Title 17 Section 105
 of the United States Code, this software is not subject to copyright
 protection and is in the public domain. NIST assumes no responsibility
 whatsoever for use by other parties of its source code or open source
 server, and makes no guarantees, expressed or implied, about its quality,
 reliability, or any other characteristic."
"""

import cv2
import glymur
import math
import copy
import numpy as np
import pandas as pd
import os
import sys
import random
import multiprocessing
from collections import OrderedDict
from decimal import Decimal
from numpngw import write_apng
from string import Template
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../lib')
sys.path.append(lib_path)
import masks
from detMetrics import Metrics as dmets
from maskMetrics18 import maskMetrics as maskMetrics1 #TODO: temporary file
#from maskMetrics_old import maskMetrics as maskMetrics2
from myround import myround
from plotROC import plotROC,detPackage
from constants import *

debug_mode = True
pd.options.mode.chained_assignment = None

def perimage_per_proc(args):
    #External function to parallelize object method.
    return localization_perimage_runner.apply_score_masks(*args) 

def fields2ints(df,fields):
    for f in fields:
        df[f] = df[f].dropna().apply(lambda x: str(int(x)))
    return df

def get_ordered_df_headers(taskmode,all_cols,opt_out_cols,base_metric_cols,base_last_cols):
    """
    Return the ordered dataframe headers. Also useful for generating empty data frames.
    """
    task,mode = taskmode.split(":")
    metric_cols = base_metric_cols
    last_cols = base_last_cols
    if task == 'manipulation':
        param_ids = ['ProbeFileID']
        first_refs = ['TaskID','ProbeFileID','IsTarget','ConfidenceScore','ProbeFileName','ProbeMaskFileName']
        if 'ProbeBitPlaneMaskFileName' in all_cols:
            first_refs.append('ProbeBitPlaneMaskFileName')
        first_cols = first_refs + ['OutputProbeMaskFileName','Scored']
        first_cols.extend(opt_out_cols)
    elif task == 'splice':
        param_ids = ['ProbeFileID','DonorFileID']
        file_cols_template = ['%sFileName','%sMaskFileName','Output%sMaskFileName']
        if mode == 'base':
            metric_cols = [ 'p%s' % m for m in base_metric_cols ] + [ 'd%s' % m for m in base_metric_cols ]
            file_cols_template.append('%sScored')
            last_cols = [ 'Probe%s' % m for m in base_last_cols ] + [ 'Donor%s' % m for m in base_last_cols ]

        file_cols = [ m % 'Probe' for m in file_cols_template ] + [ m % 'Donor' for m in file_cols_template ]
        if mode == 'stack':
            file_cols.append('Scored')
        
        first_cols = ['TaskID'] + param_ids + ['IsTarget','ConfidenceScore'] + file_cols + opt_out_cols
        if 'JournalName' in all_cols:
            first_cols.append('JournalName') 

    remaining_cols = list(set(all_cols) - set(first_cols) - set(metric_cols) - set(last_cols))
    return first_cols + metric_cols + remaining_cols + last_cols

class localization_perimage_runner():
    def __init__(self,
                 task,
                 mergedf,
                 refD,
                 sysD,
                 refBin,
                 sysBin,
                 journaldf,
                 joindf,
                 index,
                 speedup=False,
                 color=False,
                 colordict={'red':[0,0,255],'blue':[255,51,51],'yellow':[0,255,255],'green':[0,207,0],'pink':[193,182,255],'purple':[211,0,148],'white':[255,255,255],'gray':[127,127,127]}):
        """
        Constructor

        Attributes:
        - mergedf: the joined dataframe of the reference file joined with the
                   index and system output file
        - refD: the directory containing the reference masks
        - sysD: the directory containing the system output masks
        - refBin: the threshold to binarize the reference mask files. Setting it to
                  -1 with a journal dataframe will enable a no-score region to be
                  selected over a set of colored regions matching certain tasks
        - sysBin: the threshold to binarize the system output mask files. Setting it to
                  -1 will compute the metrics over a set of distinct thresholds, with the
                  threshold yielding the maximum MCC picked.
        - journaldf: the journal dataframe to be saved. Contains information matching
                     the color of the manipulated region to the task in question
        - joindf: the dataframe joining information between the reference file and journal
        - index: the index file dataframe to be saved. Used solely for dimensionality validation of reference mask.
        - colordict: the dictionary of colors to use for the HTML output, in BGR array format,
                     to be used as reference
        - speedup: determines the mask metric computation method to be used
        - color: whether to use 3-channel color assessment (dated to the NC17 evaluation)
        """
        self.task = task
        self.ref_and_sys = mergedf
        self.refDir = refD
        self.sysDir = sysD
        self.rbin = refBin
        self.sbin = sysBin
        self.journal_ref = journaldf #for recordkeeping and resetting
        self.join_data = joindf
        self.index = index
        self.speedup=speedup
        self.usejpeg2000=color
        if self.task == 'splice':
            self.usejpeg2000 = False
        self.colordict=colordict

    def score_all_masks(self,out_root,query,params):
        """
        * Description: gets metrics for each pair of reference and system masks
        * Inputs:
        *     outputRoot: the directory for outputs to be written
        *     params: an object containing additional parameters for scoring, with the following variables:
        *         mode: determines the data to access. 0 denotes the default 'manipulation' task. 1 denotes the 'splice' task
                         with the probe image, 2 denotes the 'splice' task with the donor image.
        *         eks: length of the erosion kernel matrix
        *         dks: length of the dilation kernel matrix
        *         ntdks: length of the dilation kernel matrix for the unselected no-score zones.
                                       0 means nothing will be scored
        *         nspx: pixel value in the mask to treat as custom no-score region.
        *         pppns: whether or not to use the pixel value in the ProbeOptOutPixelValue column as a no-score zone for the mask
                         (DonorOptOutPixelValue for splice)
        *         kernel: kernel shape to be used
        *         verbose: permit printout from metrics
        *         html: whether or not to generate an HTML report
        *         precision: the number of digits to round the computed metrics to.
        *         processors: the number of processors to use to score the maskss.
        * Output:
        *     df: a dataframe of the computed metrics
        """

        ref_and_sys_cols = self.ref_and_sys.columns.values.tolist()

        #saving parameters from param object
        self.out_root = out_root
        self.optOut = params.optOut
        opt_out_mode = 0
        if 'IsOptOut' in ref_and_sys_cols:
            if self.optOut:
                opt_out_mode = 1
            self.opt_out_column = "IsOptOut"
            self.undesirables = ['Y','Localization']
            opt_out_cols = ['IsOptOut'] #for column ordering later
        elif 'ProbeStatus' in ref_and_sys_cols:
            if self.optOut:
                opt_out_mode = 2
            self.opt_out_column = "ProbeStatus"
            self.undesirables = ['OptOutAll','OptOutLocalization']
            if self.task == 'splice':
                opt_out_cols = ['ProbeStatus','DonorStatus']
            else:
                opt_out_cols = ['ProbeStatus']

        if params.speedup:
            self.mask_metrics = maskMetrics1
        else:
#            self.mask_metrics = maskMetrics2
            self.mask_metrics = maskMetrics1

        self.kern = params.kernel
        self.erodeKernSize = params.eks
        self.dilateKernSize = params.dks
        self.distractionKernSize = params.ntdks
        self.noScorePixel = params.nspx
        self.perProbePixelNoScore = params.pppns

        self.verbose = params.verbose
        self.precision = params.precision
        processors = params.processors
        global debug_mode
        debug_mode = params.debug_off

        self.round_modes = ['sd']
        if params.truncate:
            self.round_modes.append('t')
        self.cache_dir = False #in case we need it in a future implementation

        #reflist and syslist should come from the same dataframe, so length checking is not required
        binpfx = ''
        evalcol='Evaluated'
        mymode='Probe'
        if self.task == 'splice':
            evalcol='ProbeEvaluated'
        self.evalcol = evalcol
        self.mymode = mymode
        self.manip_file_id_col = '%sFileID' % mymode

        if self.task == 'manipulation':
            param_ids = ['ProbeFileID']
        elif self.task == 'splice':
            param_ids = ['ProbeFileID','DonorFileID']

        df = self.ref_and_sys.copy()

        all_cols = df.columns.values.tolist()
        base_metric_cols = ['OptimumThreshold','OptimumNMM','OptimumMCC','OptimumBWL1',
                            'GWL1','AUC','EER',
                             'PixelAverageAUC','MaskAverageAUC',
                            'OptimumPixelTP','OptimumPixelTN','OptimumPixelFP','OptimumPixelFN',
                            'PixelN','PixelBNS','PixelSNS','PixelPNS',
                            'MaximumThreshold','MaximumNMM','MaximumMCC','MaximumBWL1',
                            'MaximumPixelTP','MaximumPixelTN','MaximumPixelFP','MaximumPixelFN',
                            'ActualThreshold','ActualNMM','ActualMCC','ActualBWL1',
                            'ActualPixelTP','ActualPixelTN','ActualPixelFP','ActualPixelFN' 
                           ]
        base_last_cols = ['ColMaskFileName','AggMaskFileName'] #can be updated later when html module is run

        #filter journal here with a localization-relevant query
        self.journal_display = self.journal_ref.copy() #NOTE: reset the display for this scoring run
        if query != '':
            #determine whether to join on JournalName for backwards compatibility
            big_df_join_fields = param_ids
            if 'JournalName' in ref_and_sys_cols:
                big_df_join_fields = big_df_join_fields + ['JournalName']

            try:
                big_df = pd.merge(self.ref_and_sys,self.journal_ref,how='left',on=big_df_join_fields).query(query)
            except query_exception:
                print("The query '{}' doesn't seem to refer to a valid key. Please correct the query and try again.".format(query))
                exit(1)
#            df = pd.merge(df,big_df[big_df_join_fields],how='left',on=big_df_join_fields).dropna()
            df = pd.merge(df,big_df[big_df_join_fields + ['StartNodeID']],how='inner',on=big_df_join_fields).dropna().drop("StartNodeID",1).drop_duplicates()

            journal_join_fields_all = ['JournalName','StartNodeID','EndNodeID']
            journal_bigdf_join_fields = journal_join_fields_all + param_ids
        
            if self.task == 'manipulation':
                mask_file_name = "ProbeMaskFileName" #NOTE: only grab the ones with masks (i.e. has localizable manipulation)
                target_manips = self.journal_ref.reset_index().merge(big_df[journal_bigdf_join_fields + [mask_file_name]],how='left',on=journal_join_fields_all).set_index('index').dropna().drop(mask_file_name,1).index
                self.journal_display.loc[target_manips,"Evaluated"] = 'Y'
                journal_filter = self.journal_display.iloc[target_manips]
            elif self.task == 'splice':
                for pfx in ['Probe','Donor']:
                    mask_file_name = "%sMaskFileName" % pfx
                    target_manips = self.journal_ref.reset_index().merge(big_df[journal_bigdf_join_fields + [mask_file_name]],how='left',on=journal_join_fields_all).set_index('index').dropna().drop(mask_file_name,1).index
                    self.journal_display.loc[target_manips,"%sEvaluated" % pfx] = 'Y'
                journal_filter = self.journal_display.query("(ProbeEvaluated == 'Y') or (DonorEvaluated == 'Y')")

            if pd.merge(df,journal_filter,how='left',on=big_df_join_fields).dropna().shape[0] == 0:
                print("The query '{}' yielded no journal data over which image scoring may take place.".format(query))
                #return empty dataframe based on task and type
                df_and_stack = []
                for mode in ['base','stack']:
                    df_and_stack.append(pd.DataFrame(columns=get_ordered_df_headers("{}:{}".format(self.task,mode),all_cols,opt_out_cols,base_metric_cols,base_last_cols)))
                
                return df_and_stack[0],df_and_stack[1]

        #initialize null metrics
        met_cols = ['OptimumNMM','OptimumMCC','OptimumBWL1','GWL1','AUC','EER',
                    'ActualNMM','ActualMCC','ActualBWL1',
                    'OptimumPixelTP','OptimumPixelTN','OptimumPixelFP','OptimumPixelFN','OptimumThreshold',
                    'ActualPixelTP','ActualPixelTN','ActualPixelFP','ActualPixelFN','ActualThreshold',
                    'PixelN','PixelBNS','PixelSNS','PixelPNS']

        nrow = df.shape[0]
        met_mat = np.empty((nrow,len(met_cols)))
        met_mat.fill(np.nan)
        metrics_init = pd.DataFrame(met_mat,columns=met_cols,index=df.index)

        #shared object with multiprocessing manager
        manager = multiprocessing.Manager()
        self.thresscores = manager.dict()
        self.thresholds = manager.list()
        self.errlist = manager.list()
        self.msg_queue = manager.Queue()

        self.probe_id_field = 'ProbeFileID'
        self.probe_w_field = 'ProbeWidth'
        self.probe_h_field = 'ProbeHeight'
        self.sys_mask_field = 'OutputProbeMaskFileName'
        self.probe_oopx_field = 'ProbeOptOutPixelValue'

        pix2ints = ['OptimumThreshold','OptimumPixelTP','OptimumPixelFP','OptimumPixelTN','OptimumPixelFN',
                    'PixelN','PixelBNS','PixelSNS','PixelPNS']
        if self.sbin >= -1:
            pix2ints.extend(['MaximumPixelTP','MaximumPixelFP','MaximumPixelTN','MaximumPixelFN',
                             'ActualPixelTP','ActualPixelFP','ActualPixelTN','ActualPixelFN'])

        stack_df = 0
        if self.task == 'manipulation':
            #TODO: wrap each of these runs in its own function
            self.mode = 0
            self.probe_mask_field = 'ProbeMaskFileName'
            if self.usejpeg2000:
                self.probe_mask_field = 'ProbeBitPlaneMaskFileName'

            df = pd.concat([df,metrics_init],axis=1)
            df['Scored'] = 'Y'
            df['ColMaskFileName'] = ''
            df['AggMaskFileName'] = ''

            #postprocessing columns for ordering
            first_refs = ['TaskID','ProbeFileID','IsTarget','ConfidenceScore','ProbeFileName','ProbeMaskFileName']
            if 'ProbeBitPlaneMaskFileName' in ref_and_sys_cols:
                first_refs.append('ProbeBitPlaneMaskFileName')

            first_cols = first_refs + ['OutputProbeMaskFileName','Scored',self.opt_out_column]
            metric_cols = base_metric_cols
            last_cols = base_last_cols
            remaining_cols = list(set(all_cols) - set(first_cols) - set(metric_cols) - set(last_cols))
             
            df = self.score_perimage_run(df,processors)
            self.genROCs(df,out_root,self.task)
            #maximum metrics scoring
            #merge all the thresholds for score_max_metrics

            #TODO: define this via a new class
            df = self.score_max_metrics(df)

        elif self.task == 'splice':
            #TODO: wrap each of these runs in its own function
            self.mode = 1 #probe first
            self.probe_mask_field = 'BinaryProbeMaskFileName'

            df = pd.concat([df,metrics_init],axis=1)
            df['Scored'] = 'Y'
            df['ColMaskFileName'] = ''
            df['AggMaskFileName'] = ''
            df = self.score_perimage_run(df,processors)
            #generate ROC here
            self.genROCs(df,out_root,self.task)

            #TODO: define this via a new class
            df = self.score_max_metrics(df)
            #empty out threshold cache when done to make room for donor
            self.thresscores = manager.dict()
            self.thresholds = manager.list()
            self.errlist = manager.list()
            #setup stack dataframe
            stack_cols = get_ordered_df_headers("{}:{}".format(self.task,'stack'),all_cols,opt_out_cols,base_metric_cols,base_last_cols)
            stack_df = df[stack_cols]
            stack_df['ScoredMask'] = 'Probe'

            #prefix all metrics with 'p'
            df.rename(index=str,columns={"OptimumNMM":"pOptimumNMM",
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

            self.mode = 2 #then donor
            self.probe_id_field = 'DonorFileID'
            self.probe_w_field = 'DonorWidth'
            self.probe_h_field = 'DonorHeight'
            self.probe_mask_field = 'DonorMaskFileName'
            self.sys_mask_field = 'OutputDonorMaskFileName'
            self.probe_oopx_field = 'DonorOptOutPixelValue'
            self.evalcol = "DonorEvaluated"
            if opt_out_mode == 2:
                self.opt_out_column = "DonorStatus"
                self.undesirables = ['OptOutLocalization']
            metrics_init.index = df.index
            df = pd.concat([df,metrics_init],axis=1)
            df['Scored'] = 'Y'
            df['ColMaskFileName'] = ''
            df['AggMaskFileName'] = ''
            df = self.score_perimage_run(df,processors)
            #generate ROC here
            self.genROCs(df,out_root,self.task)

            #TODO: define this via a new class
            df = self.score_max_metrics(df)
            #setup stack dataframe
            stack_df_donor = df[stack_cols]
            stack_df_donor['ScoredMask'] = 'Donor'
            
            stack_df = pd.concat([stack_df,stack_df_donor],axis=0) #stack the two dfs
            stack_df = stack_df.reset_index(drop=True)
            
            #prefix all metrics with 'd'
            df.rename(index=str,columns={"OptimumNMM":"dOptimumNMM",
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

            pix2ints = ['p%s' % m for m in pix2ints] + ['d%s' % m for m in pix2ints]

        df = fields2ints(df,pix2ints)
        #optout filtering
        if opt_out_mode == 1:
            outquery = "IsOptOut != ['Y','Localization']"
        elif opt_out_mode == 2:
            if self.mode == 0:
                outquery = "ProbeStatus!=['OptOutLocalization','OptOutAll']"
            else:
                outquery = "not ((ProbeStatus==['OptOutLocalization','OptOutAll']) and (DonorStatus==['OptOutLocalization']))"
        if opt_out_mode > 0:
            df = df.query(outquery)
            if stack_df is not 0:
                stack_df = stack_df.query(outquery)

        #more postprocessing
        if self.task == 'manipulation':
            no_manip_idx = df.query("OptimumMCC==-2").index
            df.at[no_manip_idx,'Scored'] = 'N'
            df.at[no_manip_idx,'OptimumMCC'] = np.nan
        elif self.task == 'splice':
            no_manip_idx_p = df.query("pOptimumMCC==-2").index
            no_manip_idx_d = df.query("dOptimumMCC==-2").index
            df.at[no_manip_idx_p,'ProbeScored'] = 'N'
            df.at[no_manip_idx_p,'pOptimumMCC'] = np.nan
            df.at[no_manip_idx_d,'DonorScored'] = 'N'
            df.at[no_manip_idx_d,'dOptimumMCC'] = np.nan

        if stack_df is not 0:
            no_manip_idx = stack_df.query("OptimumMCC==-2").index
            stack_df.at[no_manip_idx,'Scored'] = 'N'
            stack_df.at[no_manip_idx,'OptimumMCC'] = np.nan

        df_ordered_headers = get_ordered_df_headers("{}:{}".format(self.task,'base'),all_cols,opt_out_cols,base_metric_cols,base_last_cols)
#        df = df[first_cols + metric_cols + remaining_cols + last_cols]
        df = df[df_ordered_headers]
        if self.task == 'manipulation':
            stack_df = df

        #print everything from queue if verbose
        if self.verbose:
            while not self.msg_queue.empty():
                msg = self.msg_queue.get()
                print("="*30)
                print(msg)

        return df,stack_df

    def score_probes(self,df):
        """
        To be used to score probes or donors. Whichever goes first
        """
        #TODO: do one
        return df

    def apply_score_masks(self,df):
        return df.apply(self.score_one_mask,axis=1,reduce=False)

    def score_perimage_run(self,df,processors):
        maxprocs = max(multiprocessing.cpu_count() - 2,1)
        #if more, print warning message and use max processors
        nrow = df.shape[0]
        if (processors > nrow) and (nrow > 0):
            print("Warning: number of processors ({}) is greater than number of rows ({}) in the data. Defaulting to rows in data ({}).".format(processors,nrow,nrow))
            processors = nrow
        if processors > maxprocs:
            print("Warning: the machine does not have {} processors available. Defaulting to max ({}).".format(processors,maxprocs))
            processors = maxprocs

        if processors == 1:
            #case for one processor for efficient debugging and to eliminate overhead when running
            df = df.apply(self.score_one_mask,axis=1,reduce=False)
        else:
            #split df into array of dataframes based on number of processors (and rows in the file)
            chunksize = nrow//processors
            dfS = [[self,df[i:(i+chunksize)]] for i in range(0,nrow,chunksize)]

            p = multiprocessing.Pool(processes=processors)
            dfS = p.map(perimage_per_proc,dfS)
            p.close()
            p.join()

            #re-merge in the order found and return
            df = pd.concat(dfS)

        if isinstance(df,pd.Series):
            df = df.to_frame().transpose()

        if df.query("OptimumMCC==-2").shape[0] > 0:
            self.journal_display.loc[self.journal_display.query("{}=={}".format(self.probe_id_field,df.query("OptimumMCC==-2")[self.probe_id_field].tolist())).index,self.evalcol] = 'N'

        return df

    def score_one_mask(self,loc_row):
        if self.optOut:
            if loc_row[self.opt_out_column] in self.undesirables:
                return loc_row
       
        #use for concatenating messages 
        printbuffer = []
        try:
            #generate the output directory
            if self.task == 'manipulation':
                param_ids = [loc_row['ProbeFileID']]
            elif self.task == 'splice':
                param_ids = [loc_row['ProbeFileID'],loc_row['DonorFileID']]
            probe_id = loc_row[self.probe_id_field]
            
            output_dir = self.get_sub_outroot(self.out_root,param_ids)
            #read in the reference and system masks
            idx_row = self.index.query("{}=='{}'".format(self.probe_id_field,probe_id))
            if idx_row.shape[0] == 0:
                printbuffer.append("The probe '{}' is not in the index file. Skipping.".format(probe_id))
                msg = '\n'.join(printbuffer)
                self.msg_queue.put(msg)
                return loc_row
            idx_row = idx_row.iloc[0]
            probe_width = idx_row[self.probe_w_field]
            probe_height = idx_row[self.probe_h_field]
    
            ref_mask_name = loc_row[self.probe_mask_field]
            sys_mask_name = loc_row[self.sys_mask_field]

            rImg,sImg = self.read_masks(ref_mask_name,sys_mask_name,probe_id,probe_width,probe_height,output_dir,printbuffer)
            if (rImg is 0) and (sImg is 0):
                msg = '\n'.join(printbuffer)
                self.msg_queue.put(msg)
                #placeholder value for determining when a mask has no region to be scored and therefore has no score.
                loc_row['OptimumMCC'] = -2
                return loc_row
            elif (rImg is not 0) and (sImg is 0):
                msg = '\n'.join(printbuffer)
                self.msg_queue.put(msg)
                #set to minimum values
                loc_row['OptimumMCC'] = 0
                loc_row['OptimumNMM'] = -1
                loc_row['OptimumBWL1'] = 1
                loc_row['OptimumGWL1'] = 1
                return loc_row
            
            #generate no-score zones
            printbuffer.append("Generating no-score zones...")

            pppnspx = self.noScorePixel #TODO: have this be separate from pppns?
            if self.perProbePixelNoScore:
                pppnspx = loc_row[self.probe_oopx_field]

            wts,bns,sns,pns = self.get_no_scores(rImg,probe_id,self.erodeKernSize,self.dilateKernSize,self.distractionKernSize,self.kern,sImg,pppnspx,printbuffer)
#            wts,bns,sns = rImg.aggregateNoScore(erodeKernSize,dilateKernSize,distractionKernSize,kern)
            #save the no-score zones separately for potential future use for 
            save_params = [16,0]
            cv2.imwrite(os.path.join(output_dir,'{}_bns.png'.format(probe_id)),255*bns,save_params)
            cv2.imwrite(os.path.join(output_dir,'{}_sns.png'.format(probe_id)),255*sns,save_params)
            cv2.imwrite(os.path.join(output_dir,'{}_pns.png'.format(probe_id)),255*pns,save_params)
            #save erode and dilate of images for extensive recordkeeping.
            emat = masks.dilate(rImg.bwmat,self.kern,self.erodeKernSize)
            dmat = masks.erode(rImg.bwmat,self.kern,self.dilateKernSize)
            cv2.imwrite(os.path.join(output_dir,'{}_erode.png'.format(probe_id)),emat,save_params)
            cv2.imwrite(os.path.join(output_dir,'{}_dilate.png'.format(probe_id)),dmat,save_params)

            rbin_name = os.path.join(output_dir,'-'.join([rImg.name.split('/')[-1][:-4],'bin.png']))
            rImg.save_color_ns(rbin_name,bns,sns,pns)
            #TODO: save rImg full and rImg partial

            #do a 3-channel combine with bns and sns for their colors before saving
            if wts.sum() == 0:
                printbuffer.append("Warning: No-score region covers all of {} {}. Skipping it.".format(self.probe_id_field,probe_id))
                loc_row['Scored'] = 'Y'
    
            #score the reference and system masks
            #TODO: pass in objects but don't make new ones?
            metric_runner = self.mask_metrics(rImg,sImg,wts)
            all_metrics,threshold_metrics = metric_runner.get_all_metrics(self.sbin,bns,sns,pns,self.erodeKernSize,self.dilateKernSize,self.distractionKernSize,self.kern,precision=self.precision,round_modes=self.round_modes,myprintbuffer=printbuffer)
            #save the masks
            optbin_name = os.path.join(output_dir,'{}-bin.png'.format(sImg.name.split('/')[-1][:-4]))
            optT = all_metrics['OptimumThreshold']
            if np.isnan(optT):
                sImg.bwmat = 255*np.ones(sImg.get_dims(),dtype=np.uint8)
                sImg.save(optbin_name)
            else:
                sImg.save(optbin_name,th=optT)
            if self.sbin >= -1:
                sImg.save('{}-actual_bin.png'.format(os.path.join(output_dir,os.path.basename(sImg.name)[:-4])),th=all_metrics['OptimumThreshold'])

            #assign scores to relevant entries in the row
            for m in ['GWL1','AUC','EER']:
                loc_row[m] = myround(all_metrics[m],self.precision,self.round_modes)

            for m in ['Threshold','NMM','MCC','BWL1']:
                printbuffer.append("Setting value for {}...".format(m))
                opt_met = 'Optimum%s' % m
                set_threshold = m == 'Threshold'
                if set_threshold:
                    loc_row[opt_met] = all_metrics[opt_met]
                else:
                    loc_row[opt_met] = myround(all_metrics[opt_met],self.precision,self.round_modes)
                if self.sbin >= -1:
                    act_met = 'Actual%s' % m
                    if set_threshold:
                        loc_row[act_met] = all_metrics[act_met]
                    else:
                        loc_row[act_met] = myround(all_metrics[act_met],self.precision,self.round_modes)

            for m in ['TP','TN','FP','FN']:#,'BNS','SNS','PNS']:
                printbuffer.append("Setting value for {}...".format(m))
                opt_met = 'OptimumPixel%s' % m
                loc_row[opt_met] = all_metrics[opt_met]
                if self.sbin >= -1:
                    act_met = 'ActualPixel%s' % m
                    loc_row[act_met] = all_metrics[act_met]

            for m in ['BNS','SNS','PNS','N']:
                pix_met = 'Pixel%s' % m
                loc_row[pix_met] = all_metrics[pix_met]

            #save threshold_metrics to global dictionary and to dictionary
            self.thresholds.extend(threshold_metrics['Threshold'].tolist())
            self.thresscores[probe_id] = threshold_metrics
            threshold_metrics.to_csv(os.path.join(output_dir,'thresMets.csv'),sep="|",index=False)

            #TODO: save this for another function
            #plot the ROC curve
            non_null_roc_rows = threshold_metrics.query("(TP + FN > 0) and (FP + TN > 0)")
            n_non_null_rows = non_null_roc_rows.shape[0]
#            if n_non_null_rows > 0:
            if False:
                if n_non_null_rows < threshold_metrics.shape[0]:
                    threshold_metrics.at[non_null_roc_rows.index,'TPR'] = non_null_roc_rows['TP']/(non_null_roc_rows['TP'] + non_null_roc_rows['FN'])
                    threshold_metrics.at[non_null_roc_rows.index,'FPR'] = non_null_roc_rows['FP']/(non_null_roc_rows['FP'] + non_null_roc_rows['TN']) 
                rocvalues = threshold_metrics[['TPR','FPR']]
                rocvalues = rocvalues.sort_values(by=['FPR','TPR'],ascending=[True,True]).reset_index(drop=True)
                mydets = detPackage(rocvalues['TPR'],
                                    rocvalues['FPR'],
                                    1,
                                    0,
                                    all_metrics['AUC'],
                                    all_metrics['OptimumPixelTP'] + all_metrics['OptimumPixelFN'],
                                    all_metrics['OptimumPixelFP'] + all_metrics['OptimumPixelTN'])

                myroc = plotROC(mydets,'roc','ROC of %s' % probe_id,output_dir) 
 
            msg = '\n'.join(printbuffer)
            self.msg_queue.put(msg)
            return loc_row
        except:
            exc_type,exc_obj,exc_tb = sys.exc_info()
            printbuffer.append("Scoring run for {} {} encountered exception {} at line {}.".format(self.probe_id_field,probe_id,exc_type,exc_tb.tb_lineno))
            msg = '\n'.join(printbuffer)
            self.msg_queue.put(msg)
            if debug_mode:
                while not self.msg_queue.empty():
                    msg = self.msg_queue.get()
                    print("*"*30) #TODO: tentative
                    print(msg)
                raise
            return loc_row

    def genROCs(self,scoredf,outroot,task):
        #generates the ROC for each row
        scoredf.apply(self.genROC,axis=1,reduce=False,outroot=outroot,task=task,mode=self.mode)
        return scoredf

    def genROC(self,scorerow,outroot,task,mode):
        if mode == 2:
            id_field = 'DonorFileID'
            mymode = 'Donor'
#            t_field = 'dOptimumThreshold'
        else:
            id_field = 'ProbeFileID'
            mymode = 'Probe'
#            if mode == 1:
#                t_field = 'pOptimumThreshold'
#            else:
#                t_field = 'OptimumThreshold'
        t_field = 'OptimumThreshold'

        if mode > 0:
            param_ids = [scorerow['ProbeFileID'],scorerow['DonorFileID']]
        else:
            param_ids = [scorerow['ProbeFileID']]

        output_dir = self.get_sub_outroot(outroot,param_ids)

        file_id = scorerow[id_field]
        if np.isnan(scorerow[t_field]):
            return scorerow

        try:
            threshold_metrics = self.thresscores[file_id]
            tmets = threshold_metrics.iloc[0]
        except:
            return scorerow

        #plot the ROC curve
        non_null_roc_rows = threshold_metrics.query("(TP + FN > 0) and (FP + TN > 0)")
        n_non_null_rows = non_null_roc_rows.shape[0]
        if n_non_null_rows > 0:
            if n_non_null_rows < threshold_metrics.shape[0]:
                threshold_metrics.at[non_null_roc_rows.index,'TPR'] = non_null_roc_rows['TP']/(non_null_roc_rows['TP'] + non_null_roc_rows['FN'])
                threshold_metrics.at[non_null_roc_rows.index,'FPR'] = non_null_roc_rows['FP']/(non_null_roc_rows['FP'] + non_null_roc_rows['TN']) 
            rocvalues = threshold_metrics[['TPR','FPR']]
            rocvalues = rocvalues.sort_values(by=['FPR','TPR'],ascending=[True,True]).reset_index(drop=True)
            mydets = detPackage(rocvalues['TPR'],
                                rocvalues['FPR'],
                                1,
                                0,
                                scorerow['AUC'],
                                tmets['TP'] + tmets['FN'],
                                tmets['FP'] + tmets['TN'])

            myroc = plotROC(mydets,'roc','ROC of %s' % file_id,output_dir) 
        return scorerow

    def get_sub_outroot(self,output_root,param_ids):
        """
        * Description: generates subdirectories in the output root where relevant
        * Inputs:
        *     output_root: the directory where the output of the scorer is saved
        *     param_ids: a list containing the ProbeFileID, or the ProbeFileID and DonorFileID for splice
        * Outputs:
        *     sub_outroot: the directory for files to be saved on this iteration of getting the metrics
        """

        #save all images in their own directories instead, rather than pool it all in one subdirectory.
        #depending on whether manipulation or splice (see taskID), make the relevant subdir_name
        subdir_name = "_".join(param_ids)
        sub_outroot = os.path.join(output_root,subdir_name)
        if not os.path.isdir(sub_outroot):
            os.system('mkdir {}'.format(sub_outroot))
        #further subdirectories for the splice task
        if self.mode == 1:
            sub_outroot = os.path.join(sub_outroot,'probe')
        elif self.mode == 2:
            sub_outroot = os.path.join(sub_outroot,'donor')
        #second check for additional subdirectories 
        if not os.path.isdir(sub_outroot):
            os.system('mkdir {}'.format(sub_outroot))
        return sub_outroot 

    def read_masks(self,refMaskFName,sysMaskFName,probeID,probeWidth,probeHeight,outRoot,myprintbuffer):
        """
        * Description: reads both the reference and system output masks and caches the binarized image
                       into the reference mask. If the journal dataframe is provided, the color and purpose
                       of select mask regions will also be added to the reference mask
        * Inputs:
        *     refMaskFName: the name of the reference mask to be parsed
        *     sysMaskFName: the name of the system output mask to be parsed
        *     probeID: the ProbeFileID corresponding to the reference mask
        *     outRoot: the directory where files are saved. Only relevant where sysMaskFName is blank,
                       requiring a default white mask to be generated.
        *     myprintbuffer: buffer to append printout for atomic printout
        * Outputs:
        *     rImg: the reference mask object
        *     sImg: the system output mask object
        """
        myprintbuffer.append("Reference Mask: {}, System Mask: {}".format(refMaskFName,sysMaskFName))

        #read in the system mask
        if sysMaskFName in [None,'',np.nan]:
            sysMaskName = os.path.join(outRoot,'whitemask.png')
            #generate the whitemask
            whitemask = 255*np.ones((probeHeight,probeWidth),dtype=np.uint8)
            cv2.imwrite(sysMaskName,whitemask,[16,0])
        else:
            sysMaskName = os.path.join(self.sysDir,sysMaskFName)
        sImg = masks.mask(sysMaskName)

        #read in the reference mask
        rImg = 0
        color_purpose = 0
        #create white mask if not exists
        if refMaskFName in [None,'',np.nan]:
            if self.usejpeg2000:
                refMaskName = os.path.join(outRoot,'whitemask_ref.jp2')
                whitemask = 255*np.ones((probeHeight,probeWidth),dtype=np.uint8)
                cv2.imwrite(refMaskName,whitemask)
            else:
                refMaskName = os.path.join(outRoot,'whitemask_ref.png')
                whitemask = np.zeros((probeHeight,probeWidth),dtype=np.uint8)
                glymur.Jp2k(refMaskName,whitemask)
        else:
            refMaskName = os.path.join(self.refDir,refMaskFName)

        if self.rbin >= 0:
            rImg = masks.refmask_color(refMaskName)
            rImg.binarize(self.rbin)
        else:
            #only need colors if selectively scoring
            myprintbuffer.append("Fetching {} {} from mask data...".format(self.probe_id_field,probeID))
            if self.mode != 2: #NOTE: temporary measure until we get splice sorted out.
                color_purpose = self.journal_display.query("{}=='{}'".format(self.probe_id_field,probeID))
            if not self.usejpeg2000:
                #NOTE: temporary measure for splice task
                if self.mode == 1:
                    color_purpose = 0
                rImg = masks.refmask_color(refMaskName,jData=color_purpose,mode=self.mode)
            else:
                rImg = masks.refmask(refMaskName,jData=color_purpose,mode=self.mode)
#                myprintbuffer.append("Initializing reference mask {} with colors {}.".format(refMaskName,rImg.colors))

        myprintbuffer.append("Initializing reference mask {}.".format(refMaskName))
        #additional checks
        no_ref_matrix = False
        no_sys_matrix = False
        if rImg.matrix is None:
            myprintbuffer.append("Error: Reference mask file {} is unreadable. Also check if the mask is present.".format(rImg.name))
            no_ref_matrix = True

        if sImg.matrix is None:
            myprintbuffer.append("Error: System mask file {} is unreadable. Also check if the mask is present.".format(sImg.name))
            no_sys_matrix = True

        if no_ref_matrix:
            exit(1)

        if no_sys_matrix:
            myprintbuffer.append("Defaulting to minimum metric values for the system mask.")
            return 1,0

        rImg.binarize(254)

        if not rImg.regionIsPresent():
            myprintbuffer.append("The region you are looking for is not in reference mask {}. Scoring neglected.".format(refMaskFName))
            return 0,0

        #check if the dimensions of ref match that of the sys
        rdims = rImg.get_dims()
        sdims = sImg.get_dims()
        if rdims != sdims:
            myprintbuffer.append("Error: Reference and system mask dimensions do not match for probe {}. Reference has dimensions: {}. System has dimensions: {}.".format(probe_id,rdims,sdims))
            exit(1)

        #check if rdims matches with the dimensions in the index file
        if rdims != [probeHeight,probeWidth]:
            myprintbuffer.append("Error: Reference mask does not match the dimensions in the index file. Expected: ({},{}). Got: ({},{})".format(rdims[0],rdims[1],probeHeight,probeWidth))
            exit(1)
         
        return rImg,sImg

    def get_no_scores(self,rImg,probeID,erodeKernSize,dilateKernSize,distractionKernSize,kern,sImg,pppnspx,myprintbuffer):
        wts,bns,sns = 0,0,0

        if self.cache_dir:
            bns_dir = os.path.join(self.cache_dir,'%s_bns.npy' % probeID)
            sns_dir = os.path.join(self.cache_dir,'%s_sns.npy' % probeID)
            #if it's found in the cache, read it in
            if os.path.isfile(bns_dir) and os.path.isfile(sns_dir):
#                 bns = cv2.imread(bns_dir,0)
#                 sns = cv2.imread(sns_dir,0)
                 bns = np.load(bns_dir)
                 sns = np.load(sns_dir)
                 wts = bns & sns
                 return wts,bns,sns

        wts,bns,sns = rImg.aggregateNoScore(erodeKernSize,dilateKernSize,distractionKernSize,kern)
        if self.cache_dir:
            #save the files in the cache
#            save_params = [16,0]
#            cv2.imwrite(bns_dir,bns,save_params)
#            cv2.imwrite(sns_dir,sns,save_params)
            np.save(bns_dir,bns)
            np.save(sns_dir,sns)
            
        pns=sImg.pixelNoScore(pppnspx)
        if pns is 1:
            myprintbuffer.append("{} {} is not recognized.".format(self.probe_oopx_field,pppnspx))
            exit(1)
        wts = cv2.bitwise_and(wts,pns)

        if (cv2.bitwise_and(wts,rImg.bwmat)).sum() == 0:
            myprintbuffer.append("Warning: No region in the mask {} is score-able.".format(rImg.name))

        return wts,bns,sns,pns

    #TODO: maximum metrics computation starts here. Store in a separate object.
    #TODO: make this the __init__ method
    #TODO: pass in task,top-level outRoot, dictionary of threshold scores, an array of thresholds, and a binarization threshold (default -10)
    #TODO; parallel read them all in from the list of rows in score_df
    def preprocess_threshold_metrics(self):
        probe_thres_mets = self.thresscores
        probe_thres_mets_new = {}
        all_thresholds = np.array(self.thresholds)
#        ['Threshold','NMM','MCC','BWL1','TP','TN','FP','FN','BNS','SNS','PNS','N']
        for p in self.probelist:
            thres_mets_df = probe_thres_mets[p]
            if thres_mets_df.shape[0] == 0: #a safeguard
                continue
            partial_thresholds = thres_mets_df['Threshold']
            sample_row = thres_mets_df.iloc[0]
            filled_index = np.digitize(all_thresholds,partial_thresholds,right=False) - 1
            black_rows = filled_index == -1
            black_thresholds = all_thresholds[black_rows]

            gt_pos = sample_row['TP'] + sample_row['FN']
            gt_neg = sample_row['FP'] + sample_row['TN']

            thres_mets_new_df = thres_mets_df.iloc[filled_index]
            thres_mets_new_df.at[black_thresholds,['TP','FP','TN','FN']] = gt_pos,gt_neg,0,0
            if sample_row['N'] == 0:
                thres_mets_new_df.at[:,['NMM','MCC','BWL1']] = np.nan
            else:
                if gt_pos == 0:
                    thres_mets_new_df.at[black_thresholds,['NMM','MCC','BWL1']] = np.nan,0,float(gt_neg)/sample_row['N']
                else:
                    thres_mets_new_df.at[black_thresholds,['NMM','MCC','BWL1']] = max([float(gt_pos - gt_neg)/gt_pos,-1]),0,float(gt_neg)/sample_row['N']
            
            if gt_pos == 0:
                thres_mets_new_df['TPR'] = np.nan
            if gt_neg == 0:
                thres_mets_new_df['FPR'] = np.nan

            #reassign the thresholds and index to all_thresholds
            thres_mets_new_df.index = all_thresholds
            probe_thres_mets_new[p] = thres_mets_new_df

        return probe_thres_mets_new

    def compute_pixel_probe_ROC(self,roc_values):
        aucs = {}
        for pfx in ['Pixel','Probe']:
            tpr_name = ''.join([pfx,'TPR'])
            fpr_name = ''.join([pfx,'FPR'])
            roc_pfx = pfx
            if pfx == 'Probe':
                roc_pfx = 'Mask'
            if (roc_values[tpr_name].count() > 0) and (roc_values[fpr_name].count() > 0):
                p_roc_values = roc_values[[fpr_name,tpr_name]]
                p_roc_values = p_roc_values.append(pd.DataFrame([[0,0],[1,1]],columns=[fpr_name,tpr_name]),ignore_index=True)
                p_roc = p_roc_values.sort_values(by=[fpr_name,tpr_name],ascending=[True,True]).reset_index(drop=True)
                fpr = p_roc[fpr_name]
                tpr = p_roc[tpr_name]
                myauc = dmets.compute_auc(fpr,tpr)
                aucs[''.join([roc_pfx,'AverageAUC'])] = myauc #store in scoredf to tack onto average dataframe later
        
                #compute confusion measures by using the totals across all probes
#                confsum = scoredf[['OptimumPixelTP','OptimumPixelTN','OptimumPixelFP','OptimumPixelFN']].sum(axis=0)
                confsum = roc_values[['TP','TN','FP','FN']].iloc[0]
                mydets = detPackage(tpr,
                                    fpr,
                                    1,
                                    0,
                                    myauc,
                                    confsum['TP'] + confsum['FN'],
                                    confsum['FP'] + confsum['TN'])
#                                    confsum['OptimumPixelTP'] + confsum['OptimumPixelFN'],
#                                    confsum['OptimumPixelFP'] + confsum['OptimumPixelTN'])
            
                if self.task == 'manipulation':
                    plot_name = '_'.join([roc_pfx.lower(),'average_roc'])
                    plot_title = ' '.join([roc_pfx,'Average ROC'])
                elif self.task == 'splice':
                    if self.mode == 1:
                        plot_name = '_'.join([roc_pfx.lower(),'average_roc_probe'])
                        plot_title = ' '.join(['Probe',roc_pfx,'Average ROC'])
                    if self.mode == 2:
                        plot_name = '_'.join([roc_pfx.lower(),'average_roc_donor'])
                        plot_title = ' '.join(['Donor',roc_pfx,'Average ROC'])
                plotROC(mydets,plot_name,plot_title,self.out_root)
            else:
                aucs[''.join([roc_pfx,'AverageAUC'])] = np.nan
        return aucs

    def score_max_metrics(self,scoredf):
        """
        * Description: the top-level function that scores the maximum metrics.
        """
        templist = self.thresholds
        self.thresholds = list(sorted(set(templist)))
        max_cols = ['MaximumNMM','MaximumMCC','MaximumBWL1','MaximumPixelTP','MaximumPixelTN','MaximumPixelFP','MaximumPixelFN','MaximumThreshold']
        probe_id_field = self.probe_id_field
        #if there's nothing to score in scoredf, return it
        if scoredf.query("OptimumMCC > -2").count()['OptimumMCC'] == 0:
            auc_cols = ['PixelAverageAUC','MaskAverageAUC']
            all_cols = max_cols + auc_cols
            for col in all_cols:
                scoredf[col] = np.nan
            return scoredf

        maxThreshold = -10
        scoredf['PixelAverageAUC'] = np.nan
        scoredf['MaskAverageAUC'] = np.nan
        self.probelist = self.thresscores.keys()
        
        #preprocess and then proceed to compute 
        probe_thres_mets_preprocess = self.preprocess_threshold_metrics()
        probe_thres_mets_agg = pd.concat(probe_thres_mets_preprocess.values(),keys=probe_thres_mets_preprocess.keys(),names=[probe_id_field,'Threshold'])
        thres_mets_sum = probe_thres_mets_agg.sum(level=[1])
        thres_mets_sum['PixelTPR'] = thres_mets_sum['TP']/(thres_mets_sum['TP'] + thres_mets_sum['FN'])
        thres_mets_sum['PixelFPR'] = thres_mets_sum['FP']/(thres_mets_sum['FP'] + thres_mets_sum['TN'])
        thres_mets_sum[['ProbeTPR','ProbeFPR']] = probe_thres_mets_agg[['TPR','FPR']].mean(level=[1])
        maxThreshold = thres_mets_sum['MCC'].idxmax()

#        roc_values = self.parallelize(roc_values,self.runROCvals,scoreAvgROCPerProc,1,top_procs=top_procs,top_procs_apply=top_procs_apply)
#        maxThreshold = roc_values['avgMCC'].idxmax()

#TODO: include main
        aucs = self.compute_pixel_probe_ROC(thres_mets_sum)
        auc_keys = aucs.keys()
        for pfx in ['Pixel','Mask']:
            auc_name = ''.join([pfx,'AverageAUC'])
            scoredf[auc_name] = aucs[auc_name]

        #join roc_values to scoredf
        if (self.sbin >= -1) and (maxThreshold > -10):
            #with the maxThreshold, set MaximumMCC for everything. Join that dataframe with this one
            scoredf['MaximumThreshold'] = maxThreshold
            #access the probe_thres_mets_agg for the threshold
            maxMCCdf = probe_thres_mets_agg.xs(maxThreshold,level=1)
            maxMCCdf[probe_id_field] = maxMCCdf.index
            maxMCCdf.drop_duplicates(inplace=True)
            maxMCCdf.rename(columns={'NMM':'MaximumNMM',
                                     'MCC':'MaximumMCC',
                                     'BWL1':'MaximumBWL1',
                                     'TP':'MaximumPixelTP',
                                     'TN':'MaximumPixelTN',
                                     'FP':'MaximumPixelFP',
                                     'FN':'MaximumPixelFN'},inplace=True)
            scoredf = scoredf.merge(maxMCCdf[[probe_id_field,'MaximumNMM','MaximumMCC','MaximumBWL1','MaximumPixelTP','MaximumPixelTN','MaximumPixelFP','MaximumPixelFN']],on=[probe_id_field],how='left')
        else:
            for col in max_cols:
                scoredf[col] = np.nan

        return scoredf
