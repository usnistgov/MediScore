#!/usr/bin/python

"""
 *File: metricRunner.py
 *Date: 04/26/2017
 *Original Author: Daniel Zhou
 *Co-Author: Yooyoung Lee
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
from decimal import Decimal
from numpngw import write_apng
from string import Template
lib_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(lib_path)
import masks
import Render as p
from collections import OrderedDict
from detMetrics import Metrics as dmets
from maskMetrics import maskMetrics as maskMetrics1
from maskMetrics_old import maskMetrics as maskMetrics2
from myround import myround
from constants import *

debug_off = False

def scoreMask(args):
    return maskMetricRunner.scoreMoreMasks(*args)

#for use with detection metrics plotter
class detPackage:
    def __init__(self,
                 tpr,
                 fpr,
                 fpr_stop,
                 ci_tpr,
                 auc,
                 nTarget,
                 nNonTarget):
        """
        This class is a wrapper for a collection of metrics for rendering the ROC curve
        of the detection metrics.
        """
        self.tpr = tpr
        self.fpr = fpr
        self.fpr_stop = fpr_stop
        self.ci_tpr = ci_tpr
        self.auc = auc
        self.t_num = nTarget
        self.nt_num = nNonTarget
        self.d = None #TODO: possibility of computing d in the future
        
def plotROC(mydets,plotname,plot_title,outdir):
    #initialize plot options for ROC
#    dict_plot_options_path_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../tools/MaskScorer/plotJsonFiles/plot_options.json")
#    dict_plot_options_path_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../tools/DetectionScorer/plotJsonFiles/plot_options.json")
#    p.gen_default_plot_options(dict_plot_options_path_name,plot_title=plot_title,plot_type='ROC')
#    plot_opts = p.load_plot_options(dict_plot_options_path_name)
    plot_opts = OrderedDict([
            ('title', plot_title),
            ('subtitle', ''),
            ('plot_type', 'ROC'),
            ('title_fontsize', 13),  # 15
            ('subtitle_fontsize', 11),
            ('xticks_size', 'medium'),
            ('yticks_size', 'medium'),
            ('xlabel', "False Alarm Rate [%]"),
            ('xlabel_fontsize', 11),
            ('ylabel', "Miss Detection Rate [%]"),
            ('ylabel_fontsize', 11)])
    
    opts_list = [OrderedDict([('color', 'red'),
                              ('linestyle', 'solid'),
                              ('marker', '.'),
                              ('markersize', 6),
                              ('markerfacecolor', 'red'),
                              ('label',None),
                              ('antialiased', 'False')])]

    #compute AUC and EER with detection metrics and store in 
    #add ci_tpr to rocvalues
#            rocvalues['ci_tpr'] = 0

    configRender = p.setRender([mydets],opts_list,plot_opts)
    myRender = p.Render(configRender)
    myroc = myRender.plot_curve()

    #save roc curve in the output. Automatically closes the plot.
    #TODO: error with savefig. RuntimeError involving plots. Hack around it more elegantly.
    while True:
        try:
            myroc.savefig(os.path.join(outdir,'.'.join([plotname,'pdf'])), bbox_inches='tight')
            break
        except RuntimeError:
            pass

    return myroc

class maskMetricRunner:
    """
    This class computes the metrics given a list of reference and system output mask names.
    Other relevant metadata may also be included depending on the task being evaluated.
    """
    def __init__(self,
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
        self.maskData = mergedf
        self.refDir = refD
        self.sysDir = sysD
        self.rbin = refBin
        self.sbin = sysBin
        self.journalData = journaldf
        self.joinData = joindf
        self.index = index
        self.speedup=speedup
        self.usejpeg2000=color
        self.colordict=colordict
       
    def getSubOutRoot(self,outputRoot,task,mymode,row):
        """
        * Description: generates subdirectories in the output root where relevant
        * Inputs:
        *     outputRoot: the directory where the output of the scorer is saved
        *     task: the task in question, "manipulation" or "splice"
        *     mymode: the kind of masks being evaluated as "probe" or "donor"
        *     row: the row of data from which the dataframe is iterated over
        * Outputs:
        *     subOutRoot: the directory for files to be saved on this iteration of getting the metrics
        """

        #save all images in their own directories instead, rather than pool it all in one subdirectory.
        #depending on whether manipulation or splice (see taskID), make the relevant subdir_name
        subdir_name = ''
        if task == 'manipulation':
            subdir_name = row[''.join([mymode,'FileID'])]
        elif task == 'splice':
            subdir_name = "_".join([row['ProbeFileID'],row['DonorFileID']])
        #save in subdirectory
        subOutRoot = os.path.join(outputRoot,subdir_name)
        if not os.path.isdir(subOutRoot):
            os.system(' '.join(['mkdir',subOutRoot]))
        #further subdirectories for the splice task
        if self.mode == 1:
            subOutRoot = os.path.join(subOutRoot,'probe')
        elif self.mode == 2:
            subOutRoot = os.path.join(subOutRoot,'donor')
        if not os.path.isdir(subOutRoot):
            os.system(' '.join(['mkdir',subOutRoot]))
        return subOutRoot

    def read_ref_mask(self,refMaskName,probeID,mymode,myprintbuffer):
        #read in the reference mask
        rImg = 0
        color_purpose = 0 
        if self.rbin >= 0:
            rImg = masks.refmask_color(refMaskName)
            rImg.binarize(self.rbin)
        elif self.rbin == -1:
            #only need colors if selectively scoring
            myprintbuffer.append("Fetching {}FileID {} from mask data...".format(mymode,probeID))
            if self.mode != 2: #NOTE: temporary measure until we get splice sorted out.
		color_purpose = self.journalData.query("{}FileID=='{}'".format(mymode,probeID))
            if not self.usejpeg2000:
                #NOTE: temporary measure for splice task
                if self.mode == 1:
                    color_purpose = 0
                rImg = masks.refmask_color(refMaskName,jData=color_purpose,mode=self.mode)
            else:
                rImg = masks.refmask(refMaskName,jData=color_purpose,mode=self.mode)
#                myprintbuffer.append("Initializing reference mask {} with colors {}.".format(refMaskName,rImg.colors))
            myprintbuffer.append("Initializing reference mask {}.".format(refMaskName))
        return rImg

    def readMasks(self,refMaskFName,sysMaskFName,probeID,outRoot,myprintbuffer):
        """
        * Description: reads both the reference and system output masks and caches the binarized image
                       into the reference mask. If the journal dataframe is provided, the color and purpose
                       of select mask regions will also be added to the reference mask
        * Inputs:
        *     refMaskFName: the name of the reference mask to be parsed
        *     sysMaskFName: the name of the system output mask to be parsed
        *     probeID: the ProbeFileID corresponding to the reference mask
        *     outRoot: the directory where files are saved. Only relevant where sysMaskFName is blank
        *     myprintbuffer: buffer to append printout for atomic printout
        * Outputs:
        *     rImg: the reference mask object
        *     sImg: the system output mask object
        """
        myprintbuffer.append("Reference Mask: {}, System Mask: {}".format(refMaskFName,sysMaskFName))

        refMaskName = os.path.join(self.refDir,refMaskFName)
        if sysMaskFName in [None,'',np.nan]:
            sysMaskName = os.path.join(outRoot,'whitemask.png')
        else:
            sysMaskName = os.path.join(self.sysDir,sysMaskFName)
        sImg = masks.mask(sysMaskName)
 
        if (self.journalData is 0) and (self.rbin == -1): #no journal saved and rbin not set
            self.rbin = 254 #automatically set binary threshold if no journalData provided.

        binpfx = ''
        mymode = 'Probe'
        if self.mode==1:
            binpfx = 'Binary'
        if self.mode==2:
            mymode = 'Donor'
 
        #read from cache if relevant
        rImg = self.read_ref_mask(refMaskName,probeID,mymode,myprintbuffer)
        rImg.binarize(254)
        #alternative to direct binarization
#        if self.cache_dir:
#            cache_mask_name = os.path.join(self.cache_dir,'%s_mask.png' % probeID)
#            #if file is present, read.
#            if os.path.isfile(cache_mask_name):
#                rImg.bwmat = cv2.imread(cache_mask_name,0)
#            else:
#                print 'binarize and save in the cache'
#                rImg.binarize(254)
#        else:
#            rImg.binarize(254)

        if not rImg.regionIsPresent():
            myprintbuffer.append("The region you are looking for is not in reference mask {}. Scoring neglected.".format(refMaskFName))
            return 0,0

        return rImg,sImg 

    def get_ref_no_scores(self,rImg,probeID,erodeKernSize,dilateKernSize,distractionKernSize,kern):
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

        return wts,bns,sns

    #for apply
    def scoreOneMask(self,maskRow):
        #optout and return a set of preset null metrics if opting out
        if self.optout_mode == 1:
            if maskRow['IsOptOut'] in ['Y','Localization']:
                return maskRow
        elif self.optout_mode == 2:
            if self.mode == 2:
                if maskRow['DonorStatus'] == 'OptOutLocalization':
                    return maskRow
            else:
                if maskRow['ProbeStatus'] in ['OptOutLocalization','OptOutAll']:
                    return maskRow
        
        #parameter control
        binpfx = self.binpfx
        mymode = self.mymode
        outputRoot = self.outputRoot
        task = self.task
        evalcol = self.evalcol
        verbose = self.verbose
        html = self.html
        precision = self.precision
        erodeKernSize = self.erodeKernSize
        dilateKernSize = self.dilateKernSize
        distractionKernSize = self.distractionKernSize
#        noScorePixel = self.noScorePixel
        kern = self.kern
        truncate = self.truncate
        round_modes = ['sd'] #TODO: debug
        if self.truncate:
            round_modes.append('t')

        probe_id_name = '%sFileID' % mymode
        probe_width_name = '%sWidth' % mymode
        probe_height_name = '%sHeight' % mymode

        manipFileID = maskRow[probe_id_name]
        refMaskName = 0
        #how to read in the masks
        if not self.usejpeg2000:
            refMaskName = maskRow['{}{}MaskFileName'.format(binpfx,mymode)]
        else:
            refMaskName = maskRow['{}BitPlaneMaskFileName'.format(mymode)]
        #if optOut value is 'FailedValidation, treat the name as blank
        sysMaskName = maskRow['Output{}MaskFileName'.format(mymode)]
        if 'IsOptOut' in maskRow.keys():
            if maskRow['IsOptOut'] == 'FailedValidation':
                maskRow['Output{}MaskFileName'.format(mymode)] = ''
                sysMaskName = ''
        elif 'ProbeStatus' in maskRow.keys():
            if self.mode == 2:
                if maskRow['DonorStatus'] == 'FailedValidation':
                    maskRow['Output{}MaskFileName'.format(mymode)] = ''
                    sysMaskName = ''
            else:
                if maskRow['ProbeStatus'] == 'FailedValidation':
                    maskRow['Output{}MaskFileName'.format(mymode)] = ''
                    sysMaskName = ''

        #use atomic print buffer with atomic printout at end
        myprintbuffer = []

        try:
            maskMetrics = maskMetrics2
            if self.speedup:
                maskMetrics = maskMetrics1
            subOutRoot = self.getSubOutRoot(outputRoot,task,mymode,maskRow)
            index_row = self.index.query("{}=='{}'".format(probe_id_name,manipFileID))
            if len(index_row) == 0:
                myprintbuffer.append("The probe '{}' is not in the index file. Skipping.".format(manipFileID))
                self.msg_queue.put("\n".join(myprintbuffer))
                return maskRow
            index_row = index_row.iloc[0]

            if refMaskName in [None,'',np.nan]:
                myprintbuffer.append("Empty reference {} mask file.".format(mymode.lower()))
                #save white matrix as mask in question. Dependent on index file dimensions?
                if not self.usejpeg2000:
                    refMaskName = os.path.abspath(os.path.join(subOutRoot,'whitemask_ref.png'))
                    whitemask = 255*np.ones((index_row[probe_height_name],index_row[probe_width_name]),dtype=np.uint8)
                    cv2.imwrite(refMaskName,whitemask)
                else:
                    refMaskName = os.path.abspath(os.path.join(subOutRoot,'whitemask_ref.jp2'))
                    whitemask = np.zeros((index_row[probe_height_name],index_row[probe_width_name]),dtype=np.uint8)
                    glymur.Jp2k(refMaskName,whitemask)
#                continue

            if sysMaskName in [None,'',np.nan]:
                myprintbuffer.append("Empty system {} mask file.".format(mymode.lower()))
                #save white matrix as mask in question. Dependent on index file dimensions?
                whitemask = 255*np.ones((index_row[probe_height_name],index_row[probe_width_name]),dtype=np.uint8)
                cv2.imwrite(os.path.join(subOutRoot,'whitemask.png'),whitemask)
#                continue

            rImg,sImg = self.readMasks(refMaskName,sysMaskName,manipFileID,subOutRoot,myprintbuffer)

            if (rImg is 0) and (sImg is 0):
                #no masks detected with score-able regions, so set to not scored. Use first if need to modify here.
                maskRow['Scored'] = 'N'
                maskRow['OptimumMCC'] = -2 #for reference to filter later
                self.msg_queue.put("\n".join(myprintbuffer))
                return maskRow

            if (rImg.matrix is None) or (sImg.matrix is None):
                #Likely this could be FP or FN. Set scores as usual.
                myprintbuffer.append("The index is at {}.".format(maskRow.name))
                self.msg_queue.put("\n".join(myprintbuffer))
                return maskRow

            rdims = rImg.get_dims()
            sdims = sImg.get_dims()
            myprintbuffer.append("Beginning scoring for reference image {} with dims {} and system image {} with dims {}...".format(rImg.name,rdims,sImg.name,sdims))
            if rdims != sdims:
                myprintbuffer.append("Error: Reference and system image dimensions do not match for probe {}.".format(manipFileID))
                exit(1)

            #threshold before scoring if sbin >= 0. Otherwise threshold after scoring.
            sbin_name = ''
            if self.sbin >= -1:
                sbin_name = os.path.join(subOutRoot,sImg.name.split('/')[-1][:-4] + '-actual_bin.png')
                sImg.save(sbin_name,th=self.sbin)

            #save the image separately for html and further review. Use that in the html report
            myprintbuffer.append("Generating no-score zones...")
            wts,bns,sns = self.get_ref_no_scores(rImg,manipFileID,erodeKernSize,dilateKernSize,distractionKernSize,kern)
#            wts,bns,sns = rImg.aggregateNoScore(erodeKernSize,dilateKernSize,distractionKernSize,kern)

            myprintbuffer.append("Generating reference mask with no-score zones...")
            #do a 3-channel combine with bns and sns for their colors before saving
            myprintbuffer.append("Setting system optOut no-score zone...")
            pppnspx = self.noScorePixel #TODO: have this be separate from pppns?
            if self.perProbePixelNoScore:
                pppnspx = maskRow['%sOptOutPixelValue' % mymode]
            pns=sImg.pixelNoScore(pppnspx)
            if pns is 1:
                myprintbuffer.append("Error: {}OptOutPixelValue {} is not recognized.".format(mymode,pppnspx))
                self.msg_queue.put("\n".join(myprintbuffer))
                exit(1)
            wts = cv2.bitwise_and(wts,pns)
            rbin_name = os.path.join(subOutRoot,'-'.join([rImg.name.split('/')[-1][:-4],'bin.png']))
            rImg.save_color_ns(rbin_name,bns,sns,pns)

            #if wts allows for nothing to be scored, (i.e. no GT pos), print warning message, but score as usual
            if (cv2.bitwise_and(wts,rImg.bwmat)).sum() == 0:
                myprintbuffer.append("Warning: No region in the mask {} is score-able.".format(rImg.name))

            #if wts covers entire mask, skip it
            if wts.sum() == 0:
                myprintbuffer.append("Warning: No-score region covers all of {} {}. Skipping the {}.".format(mymode,manipFileID,mymode))
                maskRow['Scored'] = 'Y'
                maskRow['OptimumThreshold'] = np.nan
                maskRow['AUC'] = np.nan
                maskRow['EER'] = np.nan
                self.msg_queue.put("\n".join(myprintbuffer))

            #computes differently depending on choice to binarize system output mask
            mets = 0
            mymeas = 0
            threshold = 0
            myprintbuffer.append("Generating metrics...")
            metricRunner = maskMetrics(rImg,sImg,wts,self.sbin)
            #not something that needs to be calculated for every iteration of threshold; only needs to be calculated once
            myprintbuffer.append("Metrics generated. Getting metrics...")

            thresMets,threshold = metricRunner.runningThresholds(rImg,sImg,bns,sns,pns,erodeKernSize,dilateKernSize,distractionKernSize,kern,myprintbuffer)
            #thresMets.to_csv(os.path.join(path_or_buf=outputRoot,'{}-thresholds.csv'.format(sImg.name)),index=False) #save to a CSV for reference
            maskRow['OptimumThreshold'] = threshold
            thresMets['TPR'] = 0.
            thresMets['FPR'] = 0.

            genROC = True
            nullRocQuery = "(TP + FN == 0) or (FP + TN == 0)"
            nonNullRocQuery = "(TP + FN > 0) and (FP + TN > 0)"
            nullRocRows = thresMets.query(nullRocQuery)
            nonNullRocRows = thresMets.query(nonNullRocQuery)

            #compute TPR and FPR here for rows that have it.
            if nullRocRows.shape[0] < thresMets.shape[0]:
                thresMets.at[nonNullRocRows.index,'TPR'] = nonNullRocRows['TP']/(nonNullRocRows['TP'] + nonNullRocRows['FN'])
                thresMets.at[nonNullRocRows.index,'FPR'] = nonNullRocRows['FP']/(nonNullRocRows['FP'] + nonNullRocRows['TN'])

            #if no rows have it, don't gen the ROC.
            if nonNullRocRows.shape[0] == 0:
                genROC = False

            #set aside for numeric threshold. If threshold is nan, set everything to 0 or nan as appropriate, make the binarized system mask a whitemask2.png,
            #and pass to HTML accordingly
            amets = 0
            myameas = 0
            #TODO: throw this into maskMetrics for nice grouping?
            if np.isnan(threshold):
                sImg.bwmat = 255*np.ones(sImg.get_dims(),dtype=np.uint8)
                optbin_name = os.path.join(subOutRoot,'whitemask2.png')
                sImg.save(optbin_name)
                metrics = thresMets.iloc[0]
                mets = metrics[['NMM','MCC','BWL1']].to_dict()
                mets['GWL1'] = np.nan
                maskRow['GWL1'] = np.nan
                mymeas = metrics[['TP','TN','FP','FN','N','BNS','SNS','PNS']].to_dict()
                amets = {}
                myameas = {}

                for mes in ['BNS','SNS','PNS','N']:
                    myprintbuffer.append("Setting value for {}...".format(mes))
                    maskRow[''.join(['Pixel',mes])] = mymeas[mes]

                for met in ['NMM','MCC','BWL1','GWL1']:
                    myprintbuffer.append("Setting value for {}...".format(met))
                    maskRow[''.join(['Optimum',met])] = mets[met]
                    if met != 'GWL1':
                        mets[''.join(['Actual',met])] = mets[met]
    
                for mes in ['TP','TN','FP','FN']:#,'BNS','SNS','PNS']:
                    myprintbuffer.append("Setting value for {}...".format(mes))
                    maskRow[''.join(['OptimumPixel',mes])] = mymeas[mes]
                    myameas[mes] = mymeas[mes]
                myameas['N'] = mymeas['N']
    
            else:
                sImg.binarize(threshold) 
                optbin_name = os.path.join(subOutRoot,sImg.name.split('/')[-1][:-4] + '-bin.png')
                sImg.save(optbin_name,th=threshold)
    
                metrics = thresMets.query('Threshold=={}'.format(threshold)).iloc[0]
                mets = metrics[['NMM','MCC','BWL1']].to_dict()
                mymeas = metrics[['TP','TN','FP','FN','N','BNS','SNS','PNS']].to_dict()
                rocvalues = thresMets[['TPR','FPR']]
    
                #insert the ProbeFileID into the manager dict
                self.thresholds.extend(thresMets['Threshold'].tolist())
                self.thresscores[manipFileID] = thresMets
    
                #append 0 and 1 to beginning and end of tpr and fpr respectively
                rocvalues = rocvalues.append(pd.DataFrame([[0,0]],columns=list(rocvalues)),ignore_index=True)
                #reindex rocvalues
                rocvalues = rocvalues.sort_values(by=['FPR','TPR'],ascending=[True,True]).reset_index(drop=True)
    
                #generate a plot and get detection metrics
                fpr = rocvalues['FPR']
                tpr = rocvalues['TPR']
    
                myauc = dmets.compute_auc(fpr,tpr)
                myeer = dmets.compute_eer(fpr,1-tpr)
    
                maskRow['AUC'] = myauc
                maskRow['EER'] = myeer
        
                if genROC:
                    mydets = detPackage(tpr,
                                        fpr,
                                        1,
                                        0,
                                        myauc,
                                        mymeas['TP'] + mymeas['FN'],
                                        mymeas['FP'] + mymeas['TN'])
                
                    myroc = plotROC(mydets,'roc',' '.join(['ROC of',maskRow['ProbeFileID']]),subOutRoot)
    #            if len(thresMets) == 1:
    #                thresMets='' #to minimize redundancy
    
                for mes in ['BNS','SNS','PNS','N']:
                    myprintbuffer.append("Setting value for {}...".format(mes))
                    maskRow[''.join(['Pixel',mes])] = mymeas[mes]
    
                if self.sbin >= -1:
                    #just get scores in one run if threshold is chosen
                    sImg.binarize(self.sbin)
                    amets = metricRunner.getMetrics(rImg,sImg,wts,self.sbin,myprintbuffer)
                    myameas = amets
    #                totalpx = idxW*idxH
    #                weighted_weights = 3 - bns - 2*sns
    #                mymeas['BNS'] = np.sum(weighted_weights == 1)
    #                mymeas['SNS'] = np.sum(weighted_weights >= 2)
    #                if pns is not 0:
    #                    weighted_weights = weighted_weights + 4*(1-pns)
    #                    mymeas['PNS'] = np.sum(weighted_weights >= 4)
    #                else:
    #                    mymeas['PNS'] = 0
                    maskRow['ActualThreshold'] = self.sbin
    #                thresMets = ''
    #            elif self.sbin == -1:
    
    #            if self.sbin == -1:
    #                myprintbuffer.append("Saving binarized system mask...")
    #                sbin_name = os.path.join(subOutRoot,sImg.name.split('/')[-1][:-4] + '-bin.png')
    #                sImg.save(sbin_name,th=threshold)
     
                mets['GWL1'] = maskMetrics.grayscaleWeightedL1(rImg,sImg,wts)
                maskRow['GWL1'] = myround(mets['GWL1'],precision,round_modes)
                for met in ['NMM','MCC','BWL1']:
                    myprintbuffer.append("Setting value for {}...".format(met))
#                    print("Optimum{}: {}, strlen: {}".format(met,myround(mets[met],precision,truncate),len(str(myround(mets[met],precision,truncate))))) #TODO: debug
                    maskRow[''.join(['Optimum',met])] = myround(mets[met],precision,round_modes)
                    if self.sbin >= -1:
                        #record Actual metrics
                        maskRow[''.join(['Actual',met])] = myround(amets[met],precision,round_modes)
                        mets[''.join(['Actual',met])] = myround(amets[met],precision,round_modes)
    
                for mes in ['TP','TN','FP','FN']:#,'BNS','SNS','PNS']:
                    myprintbuffer.append("Setting value for {}...".format(mes))
                    maskRow[''.join(['OptimumPixel',mes])] = mymeas[mes]
                    if self.sbin >= -1:
                        maskRow[''.join(['ActualPixel',mes])] = myameas[mes]
            #save the thresmets information in the image directory for recordkeeping.
            thresMets.to_csv(os.path.join(subOutRoot,'thresMets.csv'),sep="|",index=False)
            manipFileName = maskRow['%sFileName' % mymode]
            maniImgName = os.path.join(self.refDir,manipFileName)
            colordirs = self.aggregateColorMask(rImg,sImg,bns,sns,pns,kern,erodeKernSize,manipFileID,maniImgName,subOutRoot,self.colordict)

            myprintbuffer.append("Metrics computed.")
            maskRow['Scored'] = 'Y'

            #generate the HTML report
            #TODO: ideally want the HTML report to be in a separate module
            if html:
                baseFileName = maskRow['BaseFileName']
                myprintbuffer.append("Generating aggregate color mask for HTML report...")
                colMaskName=colordirs['mask']
                aggImgName=colordirs['agg']
                maskRow['ColMaskFileName'] = colMaskName
                maskRow['AggMaskFileName'] = aggImgName

                #display Actual mask if it shows up. Else display Optimum
                sbinmaskname = optbin_name
                smask_threshold = threshold
                if self.sbin >= -1:
                    sbinmaskname = sbin_name
                    smask_threshold = self.sbin

                for met in ['TP','TN','FP','FN']:
                    mymeas['OptimumPixel%s' % met] = mymeas.pop(met)
                    if self.sbin >= -1:
                        mymeas['ActualPixel%s' % met] = myameas[met]
                mymeas['PixelBNS'] = mymeas.pop('BNS')
                mymeas['PixelSNS'] = mymeas.pop('SNS')
                mymeas['PixelPNS'] = mymeas.pop('PNS')
    
                myprintbuffer.append("Generating HTML report...")
                #TODO: trim the arguments here? Just use threshold and thresMets, at min len 1? Remove mets and mymeas since we have threshold to index.
                self.manipReport(task,subOutRoot,manipFileID,manipFileName,baseFileName,rImg,sImg,rbin_name,sbinmaskname,smask_threshold,thresMets,bns,sns,pns,mets,mymeas,colMaskName,aggImgName,myprintbuffer)
            self.msg_queue.put("\n".join(myprintbuffer))
            return maskRow
        except:
            exc_type,exc_obj,exc_tb = sys.exc_info()
            print("{}FileName {} for {}FileID {} encountered exception {} at line {}.".format(mymode,refMaskName,mymode,manipFileID,exc_type,exc_tb.tb_lineno))
            self.errlist.append(exc_type)
            if not debug_off:
                self.msg_queue.put("\n".join(myprintbuffer))
                while not self.msg_queue.empty():
                    msg = self.msg_queue.get()
                    print("*"*30) #TODO: tentative
                    print(msg)
                for e in self.errlist:
                    print(e)
                raise  #debug assistant
            return maskRow

    def scoreMoreMasks(self,maskData):
        return maskData.apply(self.scoreOneMask,axis=1,reduce=False)

    def scoreMasks(self,maskData,processors):
        maxprocs = max(multiprocessing.cpu_count() - 2,1)
        #if more, print warning message and use max processors
        nrow = maskData.shape[0]
        if (processors > nrow) and (nrow > 0):
            print("Warning: too many processors for rows in the data. Defaulting to rows in data ({}).".format(nrow))
            processors = nrow
        if processors > maxprocs:
            print("Warning: the machine does not have that many processors available. Defaulting to max ({}).".format(max(maxprocs,1)))
            processors = max(maxprocs,1)

        if processors == 1:
            #case for one processor for efficient debugging and to eliminate overhead when running
            maskData = maskData.apply(self.scoreOneMask,axis=1,reduce=False)
        else:
            #split maskData into array of dataframes based on number of processors (and rows in the file)
            chunksize = nrow//processors
            maskDataS = [[self,maskData[i:(i+chunksize)]] for i in range(0,nrow,chunksize)]
    
            p = multiprocessing.Pool(processes=processors)
            maskDataS = p.map(scoreMask,maskDataS)
            p.close()
            p.join()
    
            #re-merge in the order found and return
            maskData = pd.concat(maskDataS)

        if isinstance(maskData,pd.Series):
            maskData = maskData.to_frame().transpose()

        if maskData.query("OptimumMCC==-2").shape[0] > 0:
            self.journalData.loc[self.journalData.query("{}FileID=={}".format(self.mymode,maskData.query("OptimumMCC==-2")[''.join([self.mymode,'FileID'])].tolist())).index,self.evalcol] = 'N'

        return maskData

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
                myroc = plotROC(mydets,plot_name,plot_title,self.outputRoot)
            else:
                aucs[''.join([roc_pfx,'AverageAUC'])] = np.nan
        return aucs

    def scoreMaxMetrics(self,scoredf):
        """
        * Description: the top-level function that scores the maximum metrics.
        """
        max_cols = ['MaximumNMM','MaximumMCC','MaximumBWL1','MaximumPixelTP','MaximumPixelTN','MaximumPixelFP','MaximumPixelFN','MaximumThreshold']
        probe_id_field = ''.join([self.mymode,'FileID'])
        #if there's nothing to score in scoredf, return it
        if (scoredf.query("OptimumMCC > -2").count()['OptimumMCC'] == 0):
            auc_cols = ['PixelAverageAUC','MaskAverageAUC']
            all_cols = max_cols + auc_cols
            for col in all_cols:
                scoredf[col] = np.nan
            return scoredf

        maxThreshold = -10

#        roc_values = pd.DataFrame({'PixelTPR':0.,
#                                   'PixelFPR':0.,
#                                   'ProbeTPR':0.,
#                                   'ProbeFPR':0.,
#                                   'avgMCC':0.},index=self.thresholds)

#        top_procs_apply=6
#        top_procs = 0
#        self.n_sub_procs = processors
#        if processors > top_procs_apply:
#            top_procs = 1
#            self.n_sub_procs = processors//top_procs
   
        scoredf['PixelAverageAUC'] = np.nan
        scoredf['MaskAverageAUC'] = np.nan
        self.probelist = self.thresscores.keys()
        
        #preprocess and then proceed to compute 
        probe_thres_mets_preprocess = self.preprocess_threshold_metrics()
        #if nothing in here, cease further computation
        if len(probe_thres_mets_preprocess) == 0:
            for col in max_cols:
                scoredf[col] = np.nan
            for pfx in ['Pixel','Mask']:
                auc_name = ''.join([pfx,'AverageAUC'])
                scoredf[auc_name] = np.nan
            return scoredf
        probe_thres_mets_agg = pd.concat(probe_thres_mets_preprocess.values(),keys=probe_thres_mets_preprocess.keys(),names=[probe_id_field,'Threshold'])
        thres_mets_sum = probe_thres_mets_agg.sum(level=[1])
        thres_mets_sum['PixelTPR'] = thres_mets_sum['TP']/(thres_mets_sum['TP'] + thres_mets_sum['FN'])
        thres_mets_sum['PixelFPR'] = thres_mets_sum['FP']/(thres_mets_sum['FP'] + thres_mets_sum['TN'])
        thres_mets_sum[['ProbeTPR','ProbeFPR']] = probe_thres_mets_agg[['TPR','FPR']].mean(level=[1])
        maxThreshold = thres_mets_sum['MCC'].idxmax()

#        roc_values = self.parallelize(roc_values,self.runROCvals,scoreAvgROCPerProc,1,top_procs=top_procs,top_procs_apply=top_procs_apply)
#        maxThreshold = roc_values['avgMCC'].idxmax()

        aucs = self.compute_pixel_probe_ROC(thres_mets_sum)
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
#            maxMCCdf = self.maxmets[maxThreshold]
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

    def getMetricList(self,
                      outputRoot,
                      params):
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
        #saving parameters from param object
        self.mode = params.mode
        erodeKernSize = params.eks
        dilateKernSize = params.dks
        distractionKernSize = params.ntdks
        noScorePixel = params.nspx
        self.perProbePixelNoScore = params.pppns
        kern = params.kernel
        verbose = params.verbose
        html = params.html
        precision = params.precision
        processors = params.processors
        global debug_off
        debug_off = params.debug_off
        self.cache_dir = params.cache_dir

        df_cols = self.maskData.columns.values.tolist()        
        self.optout_mode = 0
        if params.optOut:
            if "IsOptOut" in df_cols:
                self.optout_mode = 1
            elif "ProbeStatus" in df_cols:
                self.optout_mode = 2

        #reflist and syslist should come from the same dataframe, so length checking is not required
        mymode='Probe'
        if self.mode==2:
            mymode='Donor'
        self.mymode = mymode
        self.manip_file_id_col = '%sFileID' % mymode

        binpfx = ''
        evalcol='Evaluated'
        if self.mode == 1:
            binpfx = 'Binary'
            evalcol='ProbeEvaluated'
        elif self.mode == 2:
            evalcol='DonorEvaluated'

#        reflist = self.maskData['{}{}MaskFileName'.format(binpfx,mymode)]
#        syslist = self.maskData['Output{}MaskFileName'.format(mymode)]

#        manip_ids = self.maskData[mymode+'FileID']
#        maniImageFName = 0
#        baseImageFName = 0
#        if html:
#            maniImageFName = self.maskData[mymode+'FileName']
#            baseImageFName = self.maskData['BaseFileName']


        #initialize empty frame to minimum scores
#        df=pd.DataFrame({mymode+'FileID':manip_ids,
#                         'NMM':[-1.]*nrow,
#                         'MCC': 0.,
#                         'BWL1': 1.,
#                         'GWL1': 1.,
#                         'ColMaskFileName':['']*nrow,
#                         'AggMaskFileName':['']*nrow})

        df=self.maskData.copy()
        df['Scored'] = 'N' #nothing is scored until it's scored
        nrow = df.shape[0]
        #Include Optimum metrics always. (Global) Maximum and Actual (performer) when the performer specifies it.
        #initialize with numpy matrix
        met_cols = ['OptimumNMM','OptimumMCC','OptimumBWL1','GWL1','AUC','EER',
                    'ActualNMM','ActualMCC','ActualBWL1',
                    'OptimumPixelTP','OptimumPixelTN','OptimumPixelFP','OptimumPixelFN','OptimumThreshold',
                    'ActualPixelTP','ActualPixelTN','ActualPixelFP','ActualPixelFN','ActualThreshold',
                    'PixelN','PixelBNS','PixelSNS','PixelPNS']
        
#        met_mat = np.zeros((nrow,len(met_cols)))
#        met_mat[:,[0,14]] = -1.
#        met_mat[:,[2,3,5]] = 1.
#        met_mat[:,[6,7,8,14,15,16,17,18]] = np.nan
        met_mat = np.empty((nrow,len(met_cols)))
        met_mat.fill(np.nan)
        metrics_init = pd.DataFrame(met_mat,columns=met_cols)
        df = pd.concat([df,metrics_init],axis=1)
        
        df['ColMaskFileName'] = ''
        df['AggMaskFileName'] = ''

        task = self.maskData['TaskID'].iloc[0] #should all be the same for one file
#        ilog = open('index_log.txt','w+')
        self.errlist = []

        #parameter control
        self.task = task
        self.binpfx = binpfx
        self.evalcol = evalcol
        self.erodeKernSize = erodeKernSize
        self.dilateKernSize = dilateKernSize
        self.distractionKernSize = distractionKernSize
        self.noScorePixel = noScorePixel
        self.kern = kern
        self.verbose = verbose
        self.html = html
        self.precision = precision
        self.truncate = params.truncate
        self.outputRoot = outputRoot

        #shared object with multiprocessing manager
        manager = multiprocessing.Manager()
        self.thresscores = manager.dict()
        self.thresholds = manager.list()
        self.msg_queue = manager.Queue()

        #************ Scoring begins here ************
        df = self.scoreMasks(df,processors)
#        for i,row in self.maskData.iterrows():
#            if verbose: print("Scoring {} mask {} out of {}...".format(mymode.lower(),i+1,nrow))
#            scoreMask(row)

        #print all error output at very end and exit (1) if failed at any iteration of loop
        if len(self.errlist) > 1:
            exit(1)
#        ilog.close()

        templist = self.thresholds
        self.thresholds = list(sorted(set(templist)))
        df = self.scoreMaxMetrics(df)

        last_cols = [self.manip_file_id_col,'%sFileName' % mymode,'Scored',
                     'OptimumNMM',
                     'OptimumMCC',
                     'OptimumBWL1',
                     'OptimumThreshold',
                     'GWL1',
                     'AUC',
                     'EER',
                     'PixelAverageAUC',
                     'MaskAverageAUC',
                     'OptimumPixelTP',
                     'OptimumPixelTN',
                     'OptimumPixelFP',
                     'OptimumPixelFN',
                     'MaximumNMM',
                     'MaximumMCC',
                     'MaximumBWL1',
                     'MaximumThreshold',
                     'MaximumPixelTP',
                     'MaximumPixelTN',
                     'MaximumPixelFP',
                     'MaximumPixelFN',
                     'ActualNMM',
                     'ActualMCC',
                     'ActualBWL1',
                     'ActualThreshold',
                     'ActualPixelTP',
                     'ActualPixelTN',
                     'ActualPixelFP',
                     'ActualPixelFN',
                     'PixelN',
                     'PixelBNS',
                     'PixelSNS',
                     'PixelPNS',
                     'ColMaskFileName','AggMaskFileName']

        if self.verbose:
            while not self.msg_queue.empty():
                msg = self.msg_queue.get()
                print("="*30)
                print(msg)

        return df[last_cols].drop('%sFileName' % mymode,1)

    #TODO: drop this into a maskMetricsRender.py object later with renderParams object. Ideally would like to generate reports in a separate loop at the end of computations.
    def num2hex(self,color):
        """
        * Description: this function converts one BGR color to a hex string at a time
        * Inputs:
        *     color: a list of three integers from 0 to 255 denoting a color code
        * Outputs:
        *     hexcolor: the hexadecimal color code corresponding to that color
        """

        myb = hex(color[0])[2:]
        myg = hex(color[1])[2:]
        myr = hex(color[2])[2:]
        if len(myb)==1:
            myb = '0' + myb 
        if len(myg)==1:
            myg = '0' + myg 
        if len(myr)==1:
            myr = '0' + myr
        hexcolor = (''.join([myr,myg,myb])).upper()
        return hexcolor

    def nums2hex(self,colors):
        """
        * Description: this function outputs the hexadecimal strings for a dictionary of colors for the HTML report
        * Inputs:
        *     colors: list of strings corresponding to the colors in self.colordict
        * Outputs:
        *     hexcolors: dictionary of hexadecimal color codes
        """
        hexcolors = {}
        for c in colors:
            mybgr = self.colordict[c]
            hexcolors[c] = self.num2hex(mybgr)
        return hexcolors
    
    def manipReport(self,task,outputRoot,probeFileID,maniImageFName,baseImageFName,rImg,sImg,rbin_name,sbin_name,sys_threshold,thresMets,b_weights,s_weights,p_weights,metrics,confmeasures,colMaskName,aggImgName,myprintbuffer):
        """
        * Description: this function assembles the HTML report for the manipulated image and is meant to be used solely by getMetricList
        * Inputs:
        *     task: the task over which the scorer is run
        *     outputRoot: the directory to deposit the weight image and the HTML file
        *     probeFileID: the ID of the image to be considered
        *     maniImageFName: the manipulated probe file name, relative to the reference directory (self.refDir) 
        *     baseImageFName: the base file name of the probe, relative to the reference directory (self.refDir) 
        *     rImg: the unmodified reference image used for the mask evaluation
        *     sImg: the unmodified system output image used for the mask evaluation
        *     rbin_name: the name of the binarized reference image used for the mask evaluation
        *     sbin_name: the name of the binarized system output image used for the mask evaluation
        *     sys_threshold: the threshold used to binarize the system output mask
        *     thresMets: the table of thresholds for the image and the scores they yielded for that threshold
        *     b_weights: the weighted matrix of the no-score zones of the targeted regions
        *     s_weights: the weighted matrix of the no-score zones generated from the non-target regions
        *     p_weights: the weighted matrix of the no-score zones generated from pixels of a select value in the original mask
        *     metrics: the dictionary of mask scores
        *     confmeasures: truth table measures evaluated between the reference and system output masks
        *     colMaskName: the aggregate mask image of the ground truth, system output, and no-score regions
                           for the HTML report
        *     aggImgName: the above colored mask superimposed on a grayscale of the reference image
        *     myprintbuffer: buffer to append printout for atomic printout
        """
        if not os.path.isdir(outputRoot):
            os.system('mkdir ' + outputRoot)

        #compute the weights
        bwts = np.uint8(b_weights)
        swts = np.uint8(s_weights)

        rImg_name = rImg.name
        sImg_name = sImg.name
        sysBase = os.path.basename(sImg_name)[:-4]
        weightFName = '-'.join([sysBase,'weights.png'])
        weightpath = os.path.join(outputRoot,weightFName)

        myprintbuffer.append("Generating weights image...")
        mywts = cv2.bitwise_and(b_weights,s_weights)

        dims = bwts.shape
        colwts = 255*np.ones((dims[0],dims[1],3),dtype=np.uint8)
        #combine the colors for bwts and swts to colwts
        colwts[bwts==0] = self.colordict['yellow']
        colwts[swts==0] = self.colordict['pink']

        totalpns = 0
        if p_weights is not 0:
            colwts[p_weights==0] = self.colordict['purple']
            totalpns = int(np.sum(p_weights==0))
            mywts = cv2.bitwise_and(p_weights,mywts)

        totalns = (mywts == 0).sum()

        myprintbuffer.append("Saving weights image...")
        cv2.imwrite(weightpath,colwts)

        mPath = os.path.join(self.refDir,maniImageFName)
        allshapes=min(dims[1],640) #limit on width for readability of the report

        # generate HTML files
        myprintbuffer.append("Reading HTML template...")
        #TODO: save as variable in some other file that we can reformat with .format()?
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../tools/MaskScorer/html_template.txt"), 'r') as f:
            htmlstr = Template(f.read())

        #dictionary of colors corresponding to confusion measures
        cols = {'tpcol':'green','fpcol':'red','tncol':'white','fncol':'blue','bnscol':'yellow','snscol':'pink','pnscol':'purple'}
        hexs = self.nums2hex(cols.values()) #get hex strings from the BGR cols

        jtable = ''
        mymode = 'Probe'
        if self.mode == 2:
            mymode = 'Donor'
        elif self.mode == 0: #TODO: temporary measure until we get splice sorted out, originally self.mode != 2
            evalcol='Evaluated'
            if self.mode == 1:
                evalcol='ProbeEvaluated'

            myprintbuffer.append("Composing journal table...")
#            journalID = self.joinData.query("{}FileID=='{}'".format(mymode,probeFileID))['JournalName'].iloc[0]
            journalHeaders = list(self.journalData)
            toSequence = 'Sequence' in journalHeaders

            journalkeys = ['Operation','Purpose','Color',evalcol]
            if self.usejpeg2000:
                journalkeys.insert(0,'BitPlane')
            if toSequence:
                journalkeys.insert(0,'Sequence')

            jdata = self.journalData.query("ProbeFileID=='{}' & Color!=''".format(probeFileID))[journalkeys] #("JournalName=='{}'".format(journalID))[['Operation','Purpose','Color',evalcol]] #NOTE: as long as Purpose is in there. It is otherwise dispensible.
            if toSequence:
                jdata = jdata.sort_values("Sequence",ascending=False)
            #jdata.loc[pd.isnull(jdata['Purpose']),'Purpose'] = '' #make NaN Purposes empty string

            #make those color cells empty with only the color as demonstration
            jDataColors = list(jdata['Color'])
            jDataColArrays = [x[::-1] for x in [c.split(' ') for c in jDataColors]]
            jDataColArrays = [[int(x) for x in c] for c in jDataColArrays]
            jDataHex = pd.Series([self.num2hex(c) for c in jDataColArrays],index=jdata.index) #match the indices
            jdata['Color'] = 'td bgcolor="#' + jDataHex + '"btd'
            jtable = jdata.to_html(index=False)
            jtable = jtable.replace('<td>td','<td').replace('btd','>')
        
        #generate the HTML for the metrics table here. Instead of reading in a template. Use a prebuilt variable of strings instead in a separate package.
        met_table = self.gen_metrics_table(metrics)
                
        optidx = thresMets['MCC'].idxmax()

        if not np.isnan(optidx):
            optT = int(thresMets.loc[optidx]['Threshold'])
    
            met_table_prefix = 'Optimum Threshold: {}<br>'.format(optT)
            if self.sbin >= -1:
                met_table_prefix = "<ul><li><b>Optimum Threshold</b>: {}</li><li><b>Actual Threshold</b>: {}</li></ul><br>".format(optT,self.sbin)
            met_table = ''.join([met_table_prefix,met_table])

        myprintbuffer.append("Computing pixel count...") 
        totalpx = np.sum(mywts==1)
        allpx = dims[0]*dims[1]
        totalbns = confmeasures['PixelBNS'] #np.sum(bwts==0)
        totalsns = confmeasures['PixelSNS'] #np.sum(swts==0)

        perctp="nan"
        percfp="nan"
        perctn="nan"
        percfn="nan"
        percbns="nan"
        percsns="nan"
        percpns="nan"
        perctns="nan"

        if totalpx > 0:
            perctp='{0:.3f}'.format(float(confmeasures['OptimumPixelTP'])/totalpx)
            percfp='{0:.3f}'.format(float(confmeasures['OptimumPixelFP'])/totalpx)
            perctn='{0:.3f}'.format(float(confmeasures['OptimumPixelTN'])/totalpx)
            percfn='{0:.3f}'.format(float(confmeasures['OptimumPixelFN'])/totalpx)

        if allpx > 0: #standard protection
            percbns='{0:.3f}'.format(float(totalbns)/allpx)
            percsns='{0:.3f}'.format(float(totalsns)/allpx)
            percpns='{0:.3f}'.format(float(totalpns)/allpx)
            perctns='{0:.3f}'.format(float(totalns)/allpx)

        #generate table for confusion measures
        conf_measures = {}
        for m in ['TP','FP','TN','FN']:
            m_name = 'OptimumPixel%s' % m 
            conf_measures[m_name] = int(confmeasures[m_name])
            if self.sbin >= -1:
                am_name = 'ActualPixel%s' % m
                conf_measures[am_name] = int(confmeasures[am_name])
        conf_measures['TotalPixels'] = totalpx

        conf_table = self.gen_confusion_table(conf_measures)
        
        #recolor the conf_table measures
        conf_table = (conf_table.replace("TP: green",'<font style="color:#{}">TP: green</font>'.format(hexs[cols['tpcol']]))
                                .replace("FP: red",'<font style="color:#{}">FP: red</font>'.format(hexs[cols['fpcol']]))
                                .replace("TN: white",'<font style="color:#{}">TN: white</font>'.format(hexs[cols['tncol']]))
                                .replace("FN: blue",'<font style="color:#{}">FN: blue</font>'.format(hexs[cols['fncol']])))

        thresString = ''
        plt_width = 540 #NOTE: custom value for plot sizes

        if len(thresMets.dropna()) > 1:
            myprintbuffer.append("Generating MCC per threshold graph...")
            #plot MCC
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                plt.plot(thresMets['Threshold'],thresMets['MCC'],'bo',thresMets['Threshold'],thresMets['MCC'],'k')
                #plot cyan point for supremum, red point for actual if sbin >= 0, and legend with two or three as appropriate
                optidx = thresMets['MCC'].idxmax()
                optT = thresMets.loc[optidx]['Threshold']
                optMCC = thresMets.loc[optidx]['MCC']
                optpt, = plt.plot([optT],[optMCC],'co',markersize=12)
                handles = [optpt]
                labels = ['Optimal MCC']
                if self.sbin >= -1:
                    tlist = thresMets['Threshold'].tolist()
                    actT = sys_threshold
                    if sys_threshold in tlist:
                        actMCC = thresMets.query("Threshold=={}".format(sys_threshold)).iloc[0]['MCC']
                    else:
                        #get max threshold less than or equal to threshold
                        actT = max([t for t in tlist if t <= sys_threshold])
                        actMCC = thresMets.query("Threshold=={}".format(actT)).iloc[0]['MCC']
                    actpt, = plt.plot([actT],[actMCC],'ro',markersize=8)
                    handles.append(actpt)
                    labels.append('Actual MCC')
                plt.legend(handles,labels,loc='upper right', borderaxespad=0, prop={'size':8}, shadow=True, fontsize='small',numpoints=1)
                plt.suptitle('MCC per Threshold',fontsize=14)
                plt.xlabel("Binarization threshold value")
                plt.ylabel("Matthews Correlation Coefficient (MCC)")
                thresString = os.path.join(outputRoot,'thresMets.png')
                plt.savefig(thresString,bbox_inches='tight') #save the graph
                plt.close()
                thresString = "<img src=\"{}\" alt=\"thresholds graph\" style=\"width:{}px;\">".format('thresMets.png',plt_width)
            except:
                raise
                e = sys.exc_info()[0]
#                print("The plotter encountered error {}. Defaulting to table display for the HTML report.".format(e))
                print("Warning: The plotter encountered an issue. Defaulting to table display for the HTML report.")
                thresMets = thresMets.round({'NMM':3,'MCC':3,'BWL1':3,'GWL1':3})
                thresString = '<h4>Measures for Each Threshold</h4><br/>' + thresMets.to_html(index=False).replace("text-align: right;","text-align: center;")
        else:
            thresString = 'Threshold graph not applicable<br>'

        #build soft links for mPath, rImg_name, sImg_name, use compact relative links for all
        mBase = os.path.basename(mPath)
        rBase = os.path.basename(rImg_name)
        sBase = os.path.basename(sImg_name)

        #change to donor depending on mode
        basehtml = ''
        mpfx = 'probe'
        if self.mode==2:
            mpfx = 'donor'
        else:
            bPath = os.path.join(self.refDir,baseImageFName)
            bBase = os.path.basename(bPath)
            bPathNew = os.path.join(outputRoot,''.join(['baseFile',baseImageFName[-4:]]))
            try:
                os.remove(bPathNew)
            except OSError:
                None
            myprintbuffer.append(" ".join(["Creating link for base image",baseImageFName]))
            os.symlink(os.path.abspath(bPath),bPathNew)
            basehtml="<img src={} alt='base image' style='width:{}px;'>".format(''.join(['baseFile',baseImageFName[-4:]]),allshapes)

        rPathNew = os.path.join(outputRoot,'refMask.png') #os.path.join(outputRoot,rBase)
        mPathNew = os.path.join(outputRoot,''.join([mpfx,'File',maniImageFName[-4:]])) #os.path.join(outputRoot,mBase)
        sPathNew = os.path.join(outputRoot,'sysMask.png') #os.path.join(outputRoot,sBase)

        for mypath in [rPathNew,mPathNew,sPathNew]:
            try:
                os.remove(mypath)
            except OSError:
                None

        #if color, create a symbolic link. Otherwise, create and save the refMask.png
        if not self.usejpeg2000:
            myprintbuffer.append(" ".join(["Creating link for reference mask", rImg_name]))
            os.symlink(os.path.abspath(rImg_name),rPathNew)
        else:
            #create and save refMask.png.
            if self.cache_dir: 
                cache_acolor_path = os.path.join(os.path.join(self.cache_dir,'%s_acolor.png' % probeFileID))
                acolor_in_cache = os.path.isfile(cache_acolor_path)
                if acolor_in_cache:
                    #generate sym link
                    os.symlink(os.path.abspath(cache_acolor_path),rPathNew)
                else:
                    refMask = rImg.getAnimatedMask()
                    write_apng(rPathNew,refMask,delay=600,use_palette=False)
            else:
                refMask = rImg.getAnimatedMask()
                write_apng(rPathNew,refMask,delay=600,use_palette=False)
            if self.cache_dir:
                cache_acolor_path = os.path.join(os.path.join(self.cache_dir,'%s_acolor.png' % probeFileID))
                if not acolor_in_cache:
                    write_apng(cache_acolor_path,refMask,delay=600,use_palette=False)
            
        myprintbuffer.append(" ".join(["Creating link for manipulated image", maniImageFName]))
        os.symlink(os.path.abspath(mPath),mPathNew)
        myprintbuffer.append(" ".join(["Creating link for system output mask", sImg_name]))
        os.symlink(os.path.abspath(sImg_name),sPathNew)

        syspfx = ''
        if self.sbin >= -1:
            syspfx = 'Actual '

        myprintbuffer.append("Writing HTML...")
        htmlstr = htmlstr.substitute({'probeName': maniImageFName,
                                      'probeFname': "".join([mpfx,'File',maniImageFName[-4:]]),#mBase,
                                      'width': allshapes,
                                      'baseName': baseImageFName,
                                      'basehtml': basehtml,#mBase,
                                      'aggMask' : os.path.basename(aggImgName),
                                      'refMask' : 'refMask.png',#rBase,
                                      'sysMask' : 'sysMask.png',#sBase,
                                      'binRefMask' : os.path.basename(rbin_name),
                                      'binSysMask' : os.path.basename(sbin_name),
                                      'systh' : sys_threshold,
                                      'noScoreZone' : os.path.basename(weightFName),
                                      'colorMask' : os.path.basename(colMaskName),
                                      'met_table' : met_table,
#  				      'nmm' : round(metrics['NMM'],3),
#  				      'mcc' : round(metrics['MCC'],3),
#  				      'bwL1' : round(metrics['BWL1'],3),
#  				      'gwL1' : round(metrics['GWL1'],3),
                                      'syspfx' : syspfx,
                                      'totalPixels' : totalpx,
                                      'conftable':conf_table,
                                      'bns' : int(totalbns),
                                      'sns' : int(totalsns),
                                      'pns' : int(totalpns),
                                      'tns' : int(totalns),
                                      'bnscol':cols['bnscol'],
                                      'snscol':cols['snscol'],
                                      'pnscol':cols['pnscol'],
                                      'bnshex':hexs[cols['bnscol']],
                                      'snshex':hexs[cols['snscol']],
                                      'pnshex':hexs[cols['pnscol']],
                                      'percbns':percbns,
                                      'percsns':percsns,
                                      'percpns':percpns,
                                      'perctns':perctns,
                                      'jtable':jtable,
                                      'th_table':thresString,
                                      'roc_curve':'<embed src=\"roc.pdf\" alt=\"roc curve\" width=\"{}\" height=\"{}\" type=\'application/pdf\'>'.format(plt_width,plt_width)}) #add journal operations and set bg color to the html

        #print htmlstr
        fprefix=os.path.basename(maniImageFName)
        fprefix=fprefix.split('.')[0]
        fname=os.path.join(outputRoot,'.'.join([fprefix,'html']))
        myhtml=open(fname,'w')
        myhtml.write(htmlstr)
        myprintbuffer.append("HTML page written.")
        myhtml.close()

    def gen_metrics_table(self,metrics,mets_for_some=['NMM','MCC','BWL1'],mets_for_all=['GWL1'],
                          rename_dict={'NMM':'Nimble Mask Metric (NMM)',
                                       'MCC':'Matthews Correlation Coefficient (MCC)',
                                       'BWL1':'Binary Weighted L1 Loss (BWL1)',
                                       'GWL1':'Grayscale Weighted L1 Loss (GWL1)'}):
        """
        *Description: this function generates the HTML string for the table of metrics and is not meant
                      to be used otherwise

        * Inputs:
        *    metrics: values of the metrics to be scored
        *    mets_for_some: list of metrics to evaluated that differ for different thresholds
        *    mets_for_all: list of metrics that do not differ for different thresholds
        *    rename_dict: dictionary of new names for metrics 

        * Output
        *    tablestring: the html string for the generated table
        """
        met_table = pd.DataFrame(index=mets_for_some,columns=['Optimum'])
        
        for met in mets_for_some:
            metstr = "nan"
            #round if numeric
            if (not isinstance(metrics[met],str)) and not np.isnan(metrics[met]):
                metstr = "{0:.3f}".format(metrics[met])

            met_table.at[met,'Optimum'] = metstr
            if self.sbin >= -1:
                ametstr = "nan"
                amet = 'Actual%s' % met
                if not isinstance(metrics[amet],str):
                    ametstr = "{0:.3f}".format(metrics[amet])
                met_table.at[met,'Actual'] = ametstr

        #rename indices
        rename_keys = rename_dict.keys()
        sub_rename_dict = {m:rename_dict[m] for m in mets_for_some if m in rename_keys}
        met_table.rename(index=sub_rename_dict,inplace=True)

        tablestring = met_table.to_html(index=True).replace("text-align: right;","text-align: center;")
        otherrows = ''
        colspan = 1
        #make column span the row
        if self.sbin >= -1:
            colspan = 2
        for met in mets_for_all:
            #add into each row
            if not isinstance(metrics[met],str) and not np.isnan(metrics[met]):
                metstr = "{0:.3f}".format(metrics[met])
            otherrows = '\n'.join([otherrows,"<tr><th>{}</th><td colspan={}>{}</td></tr>".format(rename_dict[met],colspan,metstr)])
        #tack onto end of met_table.rename
        tablestring = (tablestring.replace("</tbody>","".join([otherrows,"</tbody>"]))
                                  .replace("<th></th>","<th>Localization Metrics</th>")
                                  .replace("<th>","<th align='left'>")
                                  .replace("<tr>","<tr align='right'>"))
        
        
        return tablestring

    def gen_confusion_table(self,conf_metrics,mets_for_some=['TP','FP','TN','FN'],mets_for_all=[],
                          rename_dict={'TP':'True Postives (TP: green)',
                                       'FP':'False Postives (FP: red)',
                                       'TN':'True Negatives (TN: white)',
                                       'FN':'False Negatives (FN: blue)'}):
        """
        *Description: this function generates the HTML string for the table of confusion measures (TP, TN, FP, and FN)
                      and is not meant to be used otherwise
        """
        met_table = pd.DataFrame(index=mets_for_some,columns=['OptimumPixelCount','OptimumProportion'])
        totalpx = conf_metrics['TotalPixels']
        for met in mets_for_some:
            metstr = "nan"
            #generate Pixel count and Proportion separation
            optcol = 'OptimumPixel%s' % met
            met_table.at[met,'OptimumPixelCount'] = conf_metrics[optcol]

            #round if numeric
            if totalpx > 0:
                metstr = "{0:.3f}".format(float(conf_metrics[optcol])/totalpx)
            met_table.at[met,'OptimumProportion'] = metstr

            #do the same for Actual metrics
            if self.sbin >= -1:
                ametstr = "nan"
                optcol = 'ActualPixel%s' % met
                met_table.at[met,'ActualPixelCount'] = conf_metrics[optcol]

                if totalpx > 0:
                    ametstr = "{0:.3f}".format(float(conf_metrics[optcol])/totalpx)
                met_table.at[met,'ActualProportion'] = ametstr

        #rename indices
        rename_keys = rename_dict.keys()
        sub_rename_dict = {m:rename_dict[m] for m in mets_for_some if m in rename_keys}
        met_table.rename(index=sub_rename_dict,inplace=True)
        cols = ['OptimumPixelCount','OptimumProportion']
        met_table.OptimumPixelCount = met_table.OptimumPixelCount.astype(int)
        if self.sbin >= -1:
            cols.extend(['ActualPixelCount','ActualProportion'])
            met_table.ActualPixelCount = met_table.ActualPixelCount.astype(int)

        met_table = met_table[cols]

        tablestring = met_table.to_html(index=True).replace("text-align: right;","text-align: center;")
        otherrows = ''
        for met in mets_for_all:
            #add into each row
            if not isinstance(conf_metrics[met],str):
                metstr = "{0:.3f}".format(metrics[met])
            otherrows = '\n'.join([otherrows,"<tr><th>{}</th><td>{}</td></tr>".format(rename_dict[met],metstr)])
        #tack onto end of met_table.rename
        tablestring = (tablestring.replace("</tbody>","".join([otherrows,"</tbody>"]))
                                  .replace("<th></th>","<th>Confuson Measures</th>")
                                  .replace("<th>","<th align='left'>")
                                  .replace("<tr>","<tr align='right'>")
                                  .replace('<table border="1" class="dataframe">','<table border="1" class="dataframe" bgcolor="#C8C8C8">'))

        return tablestring

    #prints out the aggregate mask, reference and other data
    def aggregateColorMask(self,ref,sysmask,bns,sns,pns,kern,erodeKernSize,probeFileID,maniImgName,outputMaskPath,colordict):
        """
        *Description: this function produces the aggregate mask image of the ground truth, system output,
                      and no-score regions for the HTML report, and a composite of the same image superimposed
                      on a grayscale of the reference image

        * Inputs:
        *     ref: the reference mask file
        *     sysmask: the system output mask file to be evaluated
        *     bns: the boundary no-score weighted matrix
        *     sns: the selected no-score weighted matrix
        *     kern: kernel shape to be used
        *     erodeKernSize: length of the erosion kernel matrix
        *     probeFileID: the ID of the image to be considered
        *     maniImgName: a list of reference probe images (not masks) for superimposition
        *     outputMaskPath: the directory in which to output the composite images
        *     colordict: the dictionary of colors to pass in. (e.g. {'red':[0 0 255],'green':[0 255 0]})
       
        * Output
        *     a dictionary containing the colored mask path and the aggregate mask path
        """

        #set new image as some RGB
        mydims = ref.get_dims()
        mycolor = 255*np.ones([mydims[0],mydims[1],3],dtype=np.uint8)

        if erodeKernSize > 0:
            eKern = masks.getKern(kern,erodeKernSize)
            eData = cv2.dilate(ref.bwmat,eKern,iterations=1)
        else:
            eData = ref.bwmat

        #flip all because black is 0 by default. Use the regions to determine where to color.
#        b_sImg = 1-sysmask.bwmat/255
#        b_eImg = 1-eData/255 #erosion of black/white reference mask
#        b_bnsImg = 1-bns
#        b_snsImg = 1-sns
#        b_pnsImg = 1-pns
        _,b_sImg = cv2.threshold(sysmask.bwmat,0,1,cv2.THRESH_BINARY_INV)
        _,b_eImg = cv2.threshold(eData,0,2,cv2.THRESH_BINARY_INV)
        _,b_bnsImg = cv2.threshold(bns,0,4,cv2.THRESH_BINARY_INV)
        _,b_snsImg = cv2.threshold(sns,0,8,cv2.THRESH_BINARY_INV)
        mImg = b_sImg + b_eImg + b_bnsImg + b_snsImg
        if pns is not 0:
            b_pnsImg = 0
        else:
            _,b_pnsImg = cv2.threshold(pns,0,16,cv2.THRESH_BINARY_INV)
            mImg = mImg + b_pnsImg
            
#        b_sImg[b_sImg != 0] = 1
#        b_eImg[b_eImg != 0] = 2
#        b_bnsImg[b_bnsImg != 0] = 4
#        b_snsImg[b_snsImg != 0] = 8
#        if b_pnsImg is not 1:
#            b_pnsImg[b_pnsImg != 0] = 16
#        else:
#            b_pnsImg = 0


        #set pixels equal to some value:
        #red to false accept and false reject
        #blue to no-score zone
        #pink to no-score zone that intersects with system mask
        #yellow to system mask intersect with GT
        #black to true negatives

        #get colors through colordict
        mycolor[mImg==1] = colordict['red'] #only system (FP)
        mycolor[mImg==2] = colordict['blue'] #only erode image (FN) (the part that is scored)
        mycolor[mImg==3] = colordict['green'] #system and erode image coincide (TP)
        mycolor[(mImg>=4) & (mImg <=7)] = colordict['yellow'] #boundary no-score zone
        mycolor[(mImg>=8) & (mImg <=15)] = colordict['pink'] #selection no-score zone
        mycolor[mImg>=16] = colordict['purple'] #system opt out

        #return path to mask
        outputMaskName = sysmask.name.split('/')[-1]
        outputMaskBase = outputMaskName.split('.')[0]
        finalMaskName = "_".join([outputMaskBase,"colored.jpg"])
        path=os.path.join(outputMaskPath,finalMaskName)
        #write the aggregate mask to file
        cv2.imwrite(path,mycolor)

        #also aggregate over the grayscale maniImgName for direct comparison
        #save as animated png if not using color.
        maniImg = masks.mask(maniImgName)
        mData = maniImg.matrix
#        myagg = np.zeros((mydims[0],mydims[1],3),dtype=np.uint8)
        m3chan = np.stack((mData,mData,mData),axis=2)
        #np.reshape(np.kron(mData,np.uint8([1,1,1])),(mData.shape[0],mData.shape[1],3))
        refbw = ref.bwmat
#        myagg[refbw==255]=m3chan[refbw==255]
        myagg = np.copy(m3chan)

        #for modified images, weighted sum the colored mask with the grayscale
        #np.kron(mData,np.uint8([1,1,1]))
        #mData.shape=(mydims[0],mydims[1],3)
        #things change here for pixel overlay
        #NOTE: try/catch the overlay error. Print out the shapes of all involved items.
        alpha=0.7
        try:
            refmat = ref.matrix
            if not self.usejpeg2000:
                modified = cv2.addWeighted(refmat,alpha,m3chan,1-alpha,0)
                myagg[refbw==0] = modified[refbw==0]
                compositeMaskName = "_".join([outputMaskBase,"composite.jpg"])
                compositePath = os.path.join(outputMaskPath,compositeMaskName)
                cv2.imwrite(compositePath,myagg)
            else:
                if self.cache_dir:
                    cache_acolor_path = os.path.join(os.path.join(self.cache_dir,'%s_acolorpart.npy' % probeFileID))
                    if os.path.isfile(cache_acolor_path):
                        #get this animated mask saved in and read from the cache
                        aseq = np.load(cache_acolor_path)
                    else:
                        aseq = ref.getAnimatedMask('partial')
                else:
                    aseq = ref.getAnimatedMask('partial')
                seq = []
                for frame in aseq:
                    #join the frame with the grayscale manipulated image
                    refmat = frame
                    modified = cv2.addWeighted(frame,alpha,m3chan,1-alpha,0)
                    layermask = (frame[:,:,0] != 255) | (frame[:,:,1] != 255) | (frame[:,:,2] != 255)
                    aggfr = np.copy(m3chan)
                    #overlay colors with particular manipulated regions
                    aggfr[layermask] = modified[layermask]
                    seq.append(aggfr)
                compositeMaskName = "_".join([outputMaskBase,"composite.png"])
                compositePath = os.path.join(outputMaskPath,compositeMaskName)
                write_apng(compositePath,seq,delay=600,use_palette=False)
                if self.cache_dir:
                    cache_acolor_path = os.path.join(os.path.join(self.cache_dir,'%s_acolorpart.npy' % probeFileID))
                    if not os.path.isfile(cache_acolor_path):
                        np.save(cache_acolor_path,aseq)
        except:
            exc_type,exc_obj,exc_tb = sys.exc_info()
            print("Exception {} encountered at line {} during mask overlay. Reference mask shape: {}. Probe image shape: {}".format(exc_type,exc_tb.tb_lineno,refmat.shape,m3chan.shape))
            return {'mask':path,'agg':''}

        return {'mask':path,'agg':compositePath}
