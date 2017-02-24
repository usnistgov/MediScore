"""
* File: maskreport.py
* Date: 12/22/2016
* Translated by Daniel Zhou
* Original implemented by Yooyoung Lee
* Status: Complete

* Description: This code contains the reporting functions used by
               MaskScorer.py.

* Requirements: This code requires the following packages:

    - opencv
    - pandas
    - numpy

  The other packages should be available on your system by default.

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

import cv2
import pandas as pd
import numpy as np
import sys
import os
import numbers
from string import Template
lib_path='../../lib'
sys.path.append(lib_path)
import maskMetrics as mm

def store_avg(querydf,metlist,store_df,index,precision):
    """
     Average Lists
     * Description: this function is an auxiliary function that computes averages for a dataframe's values and stores them in another dataframe's entries
     * Inputs
     *     querydf: the dataframe with the metrics to be averaged
     *     metlist: a list of metrics to average over
     *     store_df: the dataframe in which to store the averages
     *     index: the index in which to store the averages
     *     precision: the number of digits to round the averages to
    """
    for m in metlist:
        store_df.set_value(index,m,round(querydf[querydf[m].apply(lambda x: isinstance(x,numbers.Number))][m].mean(),precision))

#def avg_scores_by_factors_SSD(df, taskType,factorList={},precision=16):
#    """
#     SSD average mask performance by a factor
#     * Description: this function returns a CSV report with the average mask performance by a factor
#     * Inputs
#     *     df: the dataframe with the scored system output
#     *     taskType: [manipulation, removal, clone]
#     *     factorList: the average to be computed over this dictionary of factors and the values over which to compute them
#     *     precision: number of digits to round figures to
#     * Output
#     *     df_avg: report dataframe
#    """
#
#    num_runs = 1
#    dfdict = {'runID' : [None]*num_runs,
#              'TaskID' : [taskType]*num_runs,
#              'NMM' : np.empty(num_runs),
#              'MCC' : np.empty(num_runs),
#              'WL1' : np.empty(num_runs)}
#
#    #add to dfdict over factorList
#    for k in factorList.keys():
#        dfdict[k] = factorList[k]
#
#    df_avg = pd.DataFrame(dfdict)
#    metrics = ['NMM','MCC','WL1']
#    df_avg.set_value(0,'runID',0)
#    store_avg(df,metrics,df_avg,0,precision)
#
#    return df_avg
#
#def avg_scores_by_factors_DSD(df,taskType,factorList={},precision=16):
#    """
#     DSD average mask performance by a factor
#     * Description: this function returns a CSV report with the average mask performance by a factor
#     * Inputs
#     *     df: the dataframe with the scored system output
#     *     taskType: [splice]
#     *     precision: number of digits to round figures to
#     * Output
#     *     df_avg: report dataframe
#    """
#    num_runs = 1
#    dfdict = {'runID' : [None]*num_runs,
#              'TaskID' : [taskType]*num_runs,
#              'pNMM' : np.empty(num_runs),
#              'pMCC' : np.empty(num_runs),
#              'pWL1' : np.empty(num_runs),
#              'dNMM' : np.empty(num_runs),
#              'dMCC' : np.empty(num_runs),
#              'dWL1' : np.empty(num_runs)}
#
#    #add to dfdict over factorList
#    for k in factorList.keys():
#        dfdict[k] = factorList[k]
#
#    df_avg = pd.DataFrame(dfdict)
#    metrics=['pNMM','pMCC','pWL1','dNMM','dMCC','dWL1']
#    df_avg.set_value(0,'runID',0)
#    store_avg(sub_d,metrics,df_avg,0,precision)
#
#    return df_avg

def createReportSSD(m_df, journalData, probeJournalJoin, index, refDir, sysDir, rbin, sbin,erodeKernSize, dilateKernSize,distractionKernSize, kern,outputRoot,html,verbose,precision):
    """
     Create a CSV report for single source detection, specifically for the manipulation task
     * Description: this function calls each metric function and
                    return the metric value and the colored mask output as a report
     * Inputs
     *     m_df: reference dataframe merged with system output dataframe
     *     journalData: data frame containing the journal names, the manipulations to be considered, and the RGB color codes corresponding to each manipulation per journal
     *     probeJournalJoin: data frame containing the ProbeFileID's and JournalNames, joining probe image information with journal information
     *     index: data frame containing the index file, to be used for internal validation
     *     refDir: reference mask file directory
     *     sysDir: system output mask file directory
     *     rbin: threshold to binarize the reference mask when read in. Select -1 to not threshold (default: 254)
     *     sbin: threshold to binarize the system output mask when read in. Select -1 to not threshold (default: -1)
     *     erodekernSize: Kernel size for Erosion
     *     dilatekernSize: Kernel size for Dilation
     *     distractionkernSize: Kernel size for dilation for the distraction no-score regions
     *     kern: Kernel option for morphological image processing. Choose from 'box','disc','diamond','gaussian','line' (default: 'box')
     *     outputRoot: the directory for outputs to be written
     *     html: whether or not to output an HTML report
     *     verbose: permit printout from metrics
     * Output
     *     merged_df: report dataframe
    """

    # if the confidence score are 'nan', replace the values with the mininum score
    #m_df[pd.isnull(m_df['ConfidenceScore'])] = m_df['ConfidenceScore'].min()
    # convert to the str type to the float type for computations
    #m_df['ConfidenceScore'] = m_df['ConfidenceScore'].astype(np.float)

    maskMetricRunner = mm.maskMetricList(m_df,refDir,sysDir,rbin,sbin,journalData,probeJournalJoin,index)
    df = maskMetricRunner.getMetricList(erodeKernSize,dilateKernSize,distractionKernSize,kern,outputRoot,verbose,html,m_df['ProbeFileName'],precision=precision)

    merged_df = pd.merge(m_df,df,how='left',on='ProbeFileID')

    return merged_df

def createReportDSD(m_df, journalData, probeJournalJoin, index, refDir, sysDir, rbin, sbin,erodeKernSize, dilateKernSize,distractionKernSize, kern,outputRoot,html,verbose,precision):
    """
     Create a CSV report for double source detection, specifically the splice task
     * Description: this function calls each metric function and
                    return the metric value and the colored mask output as a report
     * Inputs
     *     m_df: reference dataframe merged with system output dataframe
     *     journalData: data frame containing the journal names, the manipulations to be considered, and the RGB color codes corresponding to each manipulation per journal
     *     probeJournalJoin: data frame containing the ProbeFileID's and JournalNames, joining probe image information with journal information
     *     index: data frame containing the index file, to be used for internal validation
     *     refDir: reference mask file directory
     *     sysDir: system output mask file directory
     *     rbin: threshold to binarize the reference mask when read in. Select -1 to not threshold (default: 254)
     *     sbin: threshold to binarize the system output mask when read in. Select -1 to not threshold (default: -1)
     *     erodekernSize: Kernel size for Erosion
     *     dilatekernSize: Kernel size for Dilation
     *     distractionkernSize: Kernel size for dilation for the distraction no-score regions
     *     kern: Kernel option for morphological image processing. Choose from 'box','disc','diamond','gaussian','line' (default: 'box')
     *     outputRoot: the directory for outputs to be written
     *     html: whether or not to output an HTML report
     *     verbose: permit printout from metrics
     * Output
     *     merged_df: report dataframe
    """

    #finds rows in index and sys which correspond to target reference
    #sub_index = index[sub_ref['ProbeFileID'].isin(index['ProbeFileID']) & sub_ref['DonorFileID'].isin(index['DonorFileID'])]
    #sub_sys = sys[sub_ref['ProbeFileID'].isin(sys['ProbeFileID']) & sub_ref['DonorFileID'].isin(sys['DonorFileID'])]

    # if the confidence score are 'nan', replace the values with the mininum score
    #m_df[pd.isnull(m_df['ConfidenceScore'])] = m_df['ConfidenceScore'].min()
    # convert to the str type to the float type for computations
    #m_df['ConfidenceScore'] = m_df['ConfidenceScore'].astype(np.float)
    maskMetricRunner = mm.maskMetricList(m_df,refDir,sysDir,rbin,sbin,journalData,probeJournalJoin,index,mode=1)
    probe_df = maskMetricRunner.getMetricList(erodeKernSize,dilateKernSize,0,kern,outputRoot,verbose,html,m_df['ProbeFileName'],precision=precision)

    maskMetricRunner = mm.maskMetricList(m_df,refDir,sysDir,rbin,sbin,journalData,probeJournalJoin,index,mode=2) #donor images
    donor_df = maskMetricRunner.getMetricList(erodeKernSize,dilateKernSize,0,kern,outputRoot,verbose,html,m_df['DonorFileName'],precision=precision)

    probe_df.rename(index=str,columns={"NMM":"pNMM",
                                       "MCC":"pMCC",
                                       "BWL1":"pBWL1",
                                       "GWL1":"pGWL1",
                                       "ColMaskFileName":"ProbeColMaskFileName",
                                       "AggMaskFileName":"ProbeAggMaskFileName"},inplace=True)

    donor_df.rename(index=str,columns={"NMM":"dNMM",
                                       "MCC":"dMCC",
                                       "BWL1":"dBWL1",
                                       "GWL1":"dGWL1",
                                       "ColMaskFileName":"DonorColMaskFileName",
                                       "AggMaskFileName":"DonorAggMaskFileName"},inplace=True)

    pd_df = pd.concat([probe_df,donor_df],axis=1)
    merged_df = pd.merge(m_df,pd_df,how='left',on=['ProbeFileID','DonorFileID'])

    return merged_df

