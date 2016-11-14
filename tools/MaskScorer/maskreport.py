"""
* File: maskreport.py
* Date: 08/30/2016
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
import os
import numbers
execfile('../../lib/masks.py')

def scores_4_mask_pairs(refMaskFName,
                        sysMaskFName,
                        maniImageFName,
                        refDir,
                        sysDir,
                        rbin,
                        sbin,
                        erodeKernSize,
                        dilateKernSize,
                        outputRoot,
                        html=False,
                        verbose=0,
                        precision=5):
    """
     Mask performance per image pair
     *Description: this function calculates metrics for a pair of mask images
     *Inputs
       *numOfRows: the number of pair images 
       *refMaskFName: list of reference mask file names
       *sysMaskFName: list of system output mask file names
       *maniImageFName: list of manipulated image file names
       *refDir: reference mask file directory
       *sysDir: system output mask file directory
       *rbin: threshold to binarize the reference mask when read in. Select -1 to not threshold (default: 254)
       *sbin: threshold to binarize the system output mask when read in. Select -1 to not threshold (default: -1)
       *erodekernSize: Kernel size for Erosion
       *dilatekernSize: Kernel size for Dilation
       *outputRoot: the directory for outputs to be written
       *html: whether or not to output an HTML report
       *verbose: option to permit printout from metrics
       *precision: number of digits to round the metric to for reporting
     *Outputs
       * report dataframe
    """
    numOfRows = len(refMaskFName)
    # Create a dataframe
    df=pd.DataFrame({'ProbeOutputMaskFileName':sysMaskFName,
                     'NMM':[-999.]*numOfRows,
                     'MCC':-999.,
                     'HAM':-999.,
                     'WL1':-999.,
                     'HL1':-999.,
                     'ColMaskFileName':['']*numOfRows,
                     'AggMaskFileName':['']*numOfRows})

    for i,row in df.iterrows():
        refMaskName = os.path.join(refDir,refMaskFName[i])
        sysMaskName = os.path.join(sysDir,sysMaskFName[i])
        if sysMaskName in [None,'']:
            print("Empty mask file at index %d" % i)
            continue
        else:
            rImg = refmask(refMaskName)
            sImg = mask(sysMaskName)
            if rbin >= 0:
                rImg.matrix = rImg.bw(rbin)
            if sbin >= 0:
                sImg.matrix = sImg.bw(sbin)

            metric = rImg.getMetrics(sImg,erodeKernSize,dilateKernSize,thres=sbin,popt=verbose)
            for met in ['NMM','MCC','HAM','WL1','HL1']:
                df.set_value(i,met,round(metric[met],precision))

            if (html):
                if (not os.path.isdir(outputRoot)):
                    os.system('mkdir ' + outputRoot)

                maniImgName = os.path.join(refDir,maniImageFName[i])
                colordirs = rImg.coloredMask_opt1(sysMaskName, maniImgName, metric['mask'], metric['wimg'], metric['eImg'], metric['dImg'], outputRoot)
               
                mywts = np.uint8(255*metric['wimg'])
                sysBase = sysMaskFName[i].split('/')[-1][:-4]
                weightFName = sysBase + '-weights.png'
                weightpath = os.path.join(outputRoot,weightFName)
                cv2.imwrite(weightpath,mywts)

                colMaskName=colordirs['mask']
                aggImgName=colordirs['agg']
                df.set_value(i,'ColMaskFileName',colMaskName)
                df.set_value(i,'AggMaskFileName',aggImgName)

                mymeas = rImg.confusion_measures(sImg,metric['wimg'])
                #html page for mask.
                #include, with headers: original image
                #color mask composite with grayscale manipulated image
                #in table
                #reference mask, system output mask
                #no score zone, color mask
                #NMM figure, and pixel area for FP,FN,TP,NS,Total area colored
                mPath = os.path.join(refDir,maniImageFName[i])
                #copy all to same directory for report
                os.system('cp ' + mPath + ' ' + outputRoot)
                os.system('cp ' + refMaskName + ' ' + outputRoot)
                os.system('cp ' + sysMaskName + ' ' + outputRoot)
                allshapes=min(rImg.get_dims()[1],640) #limit on width for readability of the report
#TODO: python template the below
                htmlstr=u"""<h2>Manipulated Image: {}</h2><br />
<img src="{}" alt="manipulated image" style="width:{}px;">
<br />
<h3>Composite with Color Mask: {}</h3>
<img src="{}" alt="manip image plus mask" style="width:{}px;">
<br />
<table border="1">
  <tbody>
    <tr>
      <td><img src="{}" alt="reference mask" style="width:{}px;"><br /><b>Reference Mask</b></td>
      <td><img src="{}" alt="system output mask" style="width:{}px;"><br /><b>System Output Mask</b></td>
    </tr>
    <tr>
      <td><img src="{}" alt="no-score zone" style="width:{}px;"><br /><b>No-Score Zone</b></td>
      <td><img src="{}" alt="color mask" style="width:{}px;"><br /><b>Mask Scoring Example</b></td>
    </tr>
    <tr>
      <td>NIMBLE Mask Metric (NMM): {}</td>
      <td><b>Total Pixels</b>: {}<br />
<b>False Positives (FP)</b>: {}<br />
<b>False Negatives (FN)</b>: {}<br />
<b>True Positives (TP)</b>: {}<br />
<b>No Score Zone (NS)</b>: {}<br /></td>
    </tr>
  </tbody>
</table>""".format(maniImageFName[i],maniImageFName[i],allshapes,
aggImgName.split('/')[-1],aggImgName.split('/')[-1],allshapes,
refMaskFName[i],allshapes,
sysMaskFName[i].split('/')[-1],allshapes,
weightFName,allshapes,
colMaskName.split('/')[-1],allshapes,
metric['NMM'],rImg.get_dims()[1]*rImg.get_dims()[0],mymeas['FP'],mymeas['FN'],mymeas['TP'],sum(sum(mywts==0))
)

                fprefix=maniImageFName[i].split('/')[-1]
                fprefix=fprefix.split('.')[0]
                fname=os.path.join(outputRoot,fprefix + '.html')
                myhtml=open(fname,'w')
                myhtml.write(htmlstr)
                myhtml.close()

    return df

def store_avg(querydf,metlist,store_df,index,precision):
    """
     Average Lists
     *Description: this function is an auxiliary function that computes averages for a dataframe's values and stores them in another dataframe's entries
     *Inputs
       *querydf: the dataframe with the metrics to be averaged
       *metlist: a list of metrics to average over
       *store_df: the dataframe in which to store the averages
       *index: the index in which to store the averages
       *precision: the number of digits to round the averages to
     *Outputs
       *none
    """
    for m in metlist:
        store_df.set_value(index,m,round(querydf[querydf[m].apply(lambda x: isinstance(x,numbers.Number))][m].mean(),precision))

def avg_scores_by_factors_SSD(df, taskType, byCols=['Collection','PostProcessed'],precision=5):
    """
     SSD average mask performance by a factor
     *Description: this function returns a CSV report with the average mask performance by a factor
     *Inputs
       *df: the dataframe with the scored system output
       *taskType: [manipulation, removal, splice]
       *byCols: the average will be computed over this set of columns
       *precision: number of digits to round figures to
     *Outputs
       *report dataframe
    """

    ids=[None]*len(byCols)
    len_ids=[0]*len(byCols)
    num_runs = 1
    for j in range(0,len(byCols)):
        ids[j] = df[byCols[j]].unique()
        len_ids[j] = len(ids[j])
    num_runs = int(np.product(len_ids))

    #TODO: Partition later
    df_avg = pd.DataFrame({'runID' : [None]*num_runs,
                           'TaskID' : [taskType]*num_runs,
                           'Collection' : [None]*num_runs,
                           'PostProcessed' : [None]*num_runs,
                           'NMM' : np.empty(num_runs),
                           'MCC' : np.empty(num_runs),
                           'HAM' : np.empty(num_runs),
                           'WL1' : np.empty(num_runs),
                           'HL1' : np.empty(num_runs)})
 
    sub_d = df.copy()

    outeridx = 0
    metrics = ['NMM','MCC','HAM','WL1','HL1']
    if (byCols != []):
        grouped = df.groupby(byCols,sort=False)
        for g in grouped.groups:
            sub_d = grouped.get_group(g)
            print("%d - Length of sub_df: %d" % (outeridx+1,sub_d.count()[0]))
            df_avg.set_value(outeridx,'runID',outeridx)
            store_avg(sub_d,metrics,df_avg,outeridx,precision)
            outeridx = outeridx + 1
            #TODO: revise to query the mean by unique values. Incorporate Partition and stuff from DetectionScorer subpackage.

    else:
        df_avg.set_value(0,'runID',0)
        store_avg(sub_d,metrics,df_avg,0,precision)
        
    return df_avg

def avg_scores_by_factors_DSD(df, taskType, byCols=['Collection','PostProcessed'],precision=5):
    """
     DSD average mask performance by a factor
     *Description: this function returns a CSV report with the average mask performance by a factor
     *Inputs
       *df: the dataframe with the scored system output
       *taskType: [manipulation, removal, splice]
       *byCols: the average will be computed over this set of columns
       *precision: number of digits to round figures to
     *Outputs
       *report dataframe
    """
    ids=[None]*len(byCols)
    len_ids=[0]*len(byCols)
    num_runs = 1
    for j in range(0,len(byCols)):
        ids[j] = df[byCols[j]].unique()
        len_ids[j] = len(ids[j])
    num_runs = int(np.product(len_ids))

    #TODO: Partition later
    df_avg = pd.DataFrame({'runID' : [None]*num_runs,
                           'TaskID' : [taskType]*num_runs,
                           'Collection' : [None]*num_runs,
                           'PostProcessed' : [None]*num_runs,
                           'pNMM' : np.empty(num_runs),
                           'pMCC' : np.empty(num_runs),
                           'pHAM' : np.empty(num_runs),
                           'pWL1' : np.empty(num_runs),
                           'pHL1' : np.empty(num_runs),
                           'dNMM' : np.empty(num_runs),
                           'dMCC' : np.empty(num_runs),
                           'dHAM' : np.empty(num_runs),
                           'dWL1' : np.empty(num_runs),
                           'dHL1' : np.empty(num_runs)})
 
    sub_d = df.copy()

    outeridx = 0
    metrics=['pNMM','pMCC','pHAM','pWL1','pHL1','dNMM','dMCC','dHAM','dWL1','dHL1']
    if (byCols != []):
        grouped = df.groupby(byCols,sort=False)
        for g in grouped.groups:
            sub_d = grouped.get_group(g)
            print("%d - Length of sub_df: %d" % (outeridx+1,sub_d.count()[0]))
            df_avg.set_value(outeridx,'runID',outeridx)
            store_avg(sub_d,metrics,df_avg,outeridx,precision)
            outeridx = outeridx + 1
            #TODO: revise to query the mean by unique values. Incorporate Partition and stuff from DetectionScorer subpackage.

    else:
        df_avg.set_value(0,'runID',0)
        store_avg(sub_d,metrics,df_avg,0,precision)

    return df_avg
    
def createReportSSD(ref, sys, index, refDir, sysDir, rbin, sbin, erodeKernSize, dilateKernSize, outputRoot,html,verbose,precision=5):
    """
     Create a CSV report for single source detection
     *Description: this function calls each metric function and 
                   return the metric value and the colored mask output as a report
     *Inputs
       *ref: reference dataframe
       *sys: system output dataframe
       *index: index dataframe
       *refDir: reference mask file directory
       *sysDir: system output mask file directory
       *rbin: threshold to binarize the reference mask when read in. Select -1 to not threshold (default: 254)
       *sbin: threshold to binarize the system output mask when read in. Select -1 to not threshold (default: -1)
       *erodekernSize: Kernel size for Erosion
       *dilatekernSize: Kernel size for Dilation
       *outputRoot: the directory for outputs to be written
       *html: whether or not to output an HTML report
       *verbose: permit printout from metrics
     *Outputs
       *report dataframe
    
       *bySet: the reports will be seperated by DatasetID(e.g., y/n, default: n)
       *byPost: the reports will be seperated by PostProcessingID(e.g., y/n, default: n)
    """
    sub_ref = ref[ref['IsTarget']=="Y"].copy() # grep only the target class
    index = index[['ProbeFileID','ProbeWidth','ProbeHeight']] #due to ProbeFileName duplication

    #finds rows in index and sys which correspond to target reference
    #sub_index = index[sub_ref['ProbeFileID'].isin(index['ProbeFileID'])]
    #sub_sys = sys[sub_ref['ProbeFileID'].isin(index['ProbeFileID'])]

    # merge the ref csv with the index csv (indicated col names due to the duplicated col names between ref and index csv files)    
    idx_df = pd.merge(sub_ref,index,how='left',on='ProbeFileID')
    # merge the ref+index file with the system csv file
    m_df = pd.merge(idx_df,sys,how='left',on='ProbeFileID')
    #f_df <- m_df[!(m_df$ProbeMaskFileName=="" | is.na(m_df$ProbeMaskFileName)),]
    
    # if the confidence score are 'nan', replace the values with the mininum score
    m_df[pd.isnull(m_df['ConfidenceScore'])] = m_df['ConfidenceScore'].min()
    # convert to the str type to the float type for computations
    m_df['ConfidenceScore'] = m_df['ConfidenceScore'].astype(np.float)

    df = scores_4_mask_pairs(m_df['ProbeMaskFileName'],
                             m_df['ProbeOutputMaskFileName'],
                             m_df['ProbeFileName'],
                             refDir,
                             sysDir,
                             rbin,
                             sbin,
                             erodeKernSize,
                             dilateKernSize,
                             outputRoot,
                             html,
                             verbose,
                             precision=precision)

    merged_df = pd.merge(m_df,df,how='left',on='ProbeOutputMaskFileName')

    #generate HTML table report
    if (html):
        html_out = merged_df.copy()

        #os.path.join doesn't seem to work with Pandas Series so just do a manual string addition
        if (outputRoot[-1] == '/'):
            outputRoot = outputRoot[:-1]

        #TODO: save in directory and manually create subdirectory instead?
        subdir = outputRoot.split('/')[-1]
        #set links around the system output data frame files
        html_out['ProbeFileName'] = '<a href="' + subdir + '/' + html_out['ProbeFileName'].str.split('/').str.get(-1).str.split('.').str.get(0) + '.html">' + html_out['ProbeFileName'] + '</a>'
        #write to index.html
        tempRoot = outputRoot + '.csv'
        tempRoot = os.path.dirname(tempRoot)
        fname = os.path.join(tempRoot,'index.html')
        myf = open(fname,'w')
        myf.write(html_out.to_html(escape=False))
        myf.close()

    return merged_df

def createReportDSD(ref, sys, index, refDir, sysDir, rbin, sbin, erodeKernSize, dilateKernSize, outputRoot,html,verbose,precision=5):
    """
     Create a CSV report for double source detection
     *Description: this function calls each metric function and 
                                 return the metric value and the colored mask output as a report
     *Inputs
       *ref: reference dataframe
       *sys: system output dataframe
       *index: index dataframe
       *refDir: reference mask file directory
       *sysDir: system output mask file directory
       *rbin: threshold to binarize the reference mask when read in. Select -1 to not threshold (default: 254)
       *sbin: threshold to binarize the system output mask when read in. Select -1 to not threshold (default: -1)
       *erodekernSize: Kernel size for Erosion
       *dilatekernSize: Kernel size for Dilation
       *outputRoot: the directory for outputs to be written
       *html: whether or not to output an HTML report
       *verbose: permit printout from metrics
     *Outputs
       *report dataframe
    """

    sub_ref = ref[ref['IsTarget']=="Y"].copy()  #grep only the target class
    index = index[['ProbeFileID','DonorFileID','ProbeWidth','ProbeHeight','DonorWidth','DonorHeight']] #due to ProbeFileName/DonorFileName duplication

    #finds rows in index and sys which correspond to target reference 
    #sub_index = index[sub_ref['ProbeFileID'].isin(index['ProbeFileID']) & sub_ref['DonorFileID'].isin(index['DonorFileID'])]
    #sub_sys = sys[sub_ref['ProbeFileID'].isin(sys['ProbeFileID']) & sub_ref['DonorFileID'].isin(sys['DonorFileID'])]

    # merge the ref csv with the index csv (indicated col names due to the duplicated col names between ref and index csv files)    
    idx_df = pd.merge(sub_ref,index,how='left',on=['ProbeFileID','DonorFileID'])
    # merge the ref+index file with the system csv file
    m_df = pd.merge(idx_df,sys,how='left',on=['ProbeFileID','DonorFileID'])
    #f_df <- m_df[!(m_df$ProbeMaskFileName=="" | is.na(m_df$ProbeMaskFileName)),]

    # if the confidence score are 'nan', replace the values with the mininum score
    m_df[pd.isnull(m_df['ConfidenceScore'])] = m_df['ConfidenceScore'].min()
    # convert to the str type to the float type for computations
    m_df['ConfidenceScore'] = m_df['ConfidenceScore'].astype(np.float)

    probe_df = scores_4_mask_pairs(m_df['ProbeMaskFileName'],
                                   m_df['ProbeOutputMaskFileName'],
                                   m_df['ProbeFileName'],
                                   refDir,
                                   sysDir,
                                   rbin,
                                   sbin,
                                   erodeKernSize,
                                   dilateKernSize,
                                   outputRoot,
                                   html,
                                   precision=precision)
    probe_df.rename(index=str,columns={"NMM":"pNMM",
                                       "MCC":"pMCC",
                                       "HAM":"pHAM",
                                       "WL1":"pWL1",
                                       "HL1":"pHL1",
                                       "ColMaskFileName":"ProbeColMaskFileName",
                                       "AggMaskFileName":"ProbeAggMaskFileName"},inplace=True)

    donor_df = scores_4_mask_pairs(m_df['DonorMaskFileName'],
                                   m_df['DonorOutputMaskFileName'],
                                   m_df['DonorFileName'],
                                   refDir,
                                   sysDir,
                                   rbin,
                                   sbin,
                                   erodeKernSize,
                                   dilateKernSize,
                                   outputRoot,
                                   html,
                                   precision=precision)
    donor_df.rename(index=str,columns={"ProbeOutputMaskFileName":"DonorOutputMaskFileName",
                                       "NMM":"dNMM",
                                       "MCC":"dMCC",
                                       "HAM":"dHAM",
                                       "WL1":"dWL1",
                                       "HL1":"dHL1",
                                       "ColMaskFileName":"DonorColMaskFileName",
                                       "AggMaskFileName":"DonorAggMaskFileName"},inplace=True)

    pd_df = pd.concat([probe_df,donor_df],axis=1)
    merged_df = pd.merge(m_df,pd_df,how='left',on=['ProbeOutputMaskFileName','DonorOutputMaskFileName'])

    if (html):
        html_out = merged_df.copy()

        #os.path.join doesn't seem to work with Pandas Series so just do a manual string addition
        if (outputRoot[-1] == '/'):
            outputRoot = outputRoot[:-1]

        subdir = outputRoot.split('/')[-1]
        #set links around the system output data frame files
        html_out['ProbeFileName'] = '<a href="' + subdir + '/' + html_out['ProbeFileName'].str.split('/').str.get(-1).str.split('.').str.get(0) + '.html">' + html_out['ProbeFileName'] + '</a>'
        html_out['DonorFileName'] = '<a href="' + subdir + '/' + html_out['DonorFileName'].str.split('/').str.get(-1).str.split('.').str.get(0) + '.html">' + html_out['DonorFileName'] + '</a>'
        #write to index.html
        tempRoot = outputRoot + '.csv'
        tempRoot = os.path.dirname(tempRoot)
        fname = os.path.join(tempRoot,'index.html')
        myf = open(fname,'w')
        myf.write(html_out.to_html(escape=False))
        myf.close()

    return merged_df
     
