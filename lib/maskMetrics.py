#!/usr/bin/python

"""
 *File: maskMetrics.py
 *Date: 12/22/2016
 *Original Author: Daniel Zhou
 *Co-Author: Yooyoung Lee
 *Status: Complete

 *Description: this code contains the metrics for evaluating the accuracy
               of the system output mask regions, as well as the runner that
               runs the metrics over each pair of masks.


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
import math
import copy
import numpy as np
import pandas as pd
import os
import sys
import random
import masks
from decimal import Decimal
from string import Template
lib_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(lib_path)
from detMetrics import Metrics as dmets
from myround import myround
from constants import *

metriccall={'MCC':'matthews',
            'NMM':'NimbleMaskMetric',
            'BWL1':'binaryWeightedL1'}

class maskMetrics:
    """
    This class evaluates the metrics for the reference and system output masks.
    The image parameters necessary to evaluate most of the objects are included
    in the initialization.
    """
    def __init__(self,ref,sys,w,systh=-10,metrics=['MCC','NMM','BWL1']):
        """
        Constructor

        Attributes:
        - ref: the reference mask file
        - sys: the system output mask file to be evaluated
        - w: the weighted matrix
        - systh: the threshold for the system output mask to be thresholded into a
                 binary mask. Letting systh = -1 will compute the metrics across all
                 distinct thresholds for the system output mask, with the threshold
                 corresponding to the highest MCC chosen
        """
        #get masks for ref and sys
        if ref.bwmat is 0:
            ref.binarize(254) #get the black/white mask first if not already gotten

        self.sys_threshold = systh
        self.ref = ref
        self.sys = sys
        self.w = w
        self.metrics = metrics
#        if systh >= 0:
#            sys.binarize(systh)
#        else:
#            distincts = np.unique(sys.matrix)
#            if (np.array_equal(distincts,[0,255])) or (np.array_equal(distincts,[0])) or (np.array_equal(distincts,[255])): #already binarized or uniform, relies on external pipeline
#                sys.bwmat = sys.matrix

        #pass threshold as a parameter here
#        self.conf = self.confusion_measures(ref,sys,w,systh)

        #record this dictionary of parameters
#        self.nmm = self.NimbleMaskMetric(self.conf,ref,w)
#        self.mcc = self.matthews(self.conf)
#        self.bwL1 = self.binaryWeightedL1(self.conf)
#        self.bwL1 = self.binaryWeightedL1(ref,sys,w,systh)

    def getMetrics(self,ref,sys,w,systh=-10,myprintbuffer=0):
        """
        * Description: this function calculates the metrics with an implemented no-score zone.
                       Due to its repeated use for the same reference and system masks, the
                       getMetrics function excludes the GWL1

        * Output:
        *     dictionary of the NMM, MCC, BWL1, and the confusion measures.
        """

        conf = self.confusion_measures(ref,sys,w,systh)

        mcc = self.matthews(conf)
        nmm = self.NimbleMaskMetric(conf)
        bwL1 = self.binaryWeightedL1(conf)

        #for nicer printout
        if myprintbuffer:
            myprintbuffer.append("NMM: {}".format(nmm))
            myprintbuffer.append("MCC (Matthews correlation coeff.): {}".format(mcc))
            myprintbuffer.append("Binary Weighted L1: {}".format(bwL1))
#        ham = self.hamming(sys)
#        if popt==1:
#            if (ham==1) or (ham==0):
#                print("HAM: %d" % ham)
#            else:
#                print("Hamming Loss: %0.9f" % ham)
#        hL1 = self.hingeL1(sys,w)
#        if popt==1:
#            if (hL1==1) or (hL1==0):
#                print("MCC: %d" % mcc)
#            else:
#                print("Hinge Loss L1: %0.9f" % hL1)

        metrics = {'NMM':nmm,'MCC':mcc,'BWL1':bwL1}
        metrics.update(conf)
        return metrics

    def confusion_measures_gs(self,ref,sys,w):
        """
        * Metric: confusion_measures_gs
        * Description: this function calculates the values in the confusion matrix (TP, TN, FP, FN)
                                     between the reference mask and a grayscale system output mask,
                                     accommodating the no score zone
                      This function is currently not used in the mask scoring scheme.
        * Inputs:
        *     ref: the reference mask object
        *     sys: the system output mask object
        *     w: the weight matrix
        * Output:
        *     dictionary of the TP, TN, FP, and FN area, and N (total score region)
        """
        r=ref.bwmat.astype(int) #otherwise, negative values won't be recorded
        s=0
        if self.sys_threshold >= -1:
            s=sys.matrix.astype(int)
        else:
            s=sys.bwmat.astype(int)
        x=np.multiply(w,(r-s)/255.) #entrywise product of w and difference between masks

        #white is 1, black is 0
        y = 1+np.copy(x)
        y[~((x<=0) & (r==0))]=0 #set all values that don't fulfill criteria to 0
        y = np.multiply(w,y)
        tp = np.sum(y) #sum_same_values + sum_neg_values

        y = np.copy(x)
        y[~((x > 0) & (r==255))]=0 #set all values that don't fulfill criteria to 0
        fp = np.sum(y)

        fn = np.sum(np.multiply((r==0),w)) - tp
        tn = np.sum(np.multiply((r==255),w)) - fp

        mydims = r.shape
        n = mydims[0]*mydims[1] - np.sum(1-w[w<1])

        return {'TP':tp,'TN':tn,'FP':fp,'FN':fn,'N':n}

    def confusion_measures(self,ref,sys,w,th):
        """
        * Metric: confusion_measures
        * Description: this function calculates the values in the confusion matrix (TP, TN, FP, FN)
                                     between the reference mask and a black and white system output mask,
                                     accommodating the no score zone
        * Inputs:
        *     ref: the reference mask object
        *     sys: the system output mask object
        *     w: the weight matrix
        *     th: the threshold for binarization
        * Output:
        *     dictionary of the TP, TN, FP, and FN areas, and total score region N
        """
        r = ref.bwmat
#.astype(int)
        if th == -10:
            th = 254

#        s = sys.bwmat.astype(int)
        s = sys.matrix <= th
#        x = (r+s)/255.
        mywts = w==1
#        rpos = cv2.bitwise_and(r==0,mywts)
#        rneg = cv2.bitwise_and(r==255,mywts)
        n = mywts.sum()
        rpos = (r==0) & mywts
        nrpos = rpos.sum()
        rneg = (r==255) & mywts

        tp = np.float64((s & rpos).sum())
        fp = np.float64((s & rneg).sum())
        fn = np.float64(nrpos - tp)
        tn = np.float64(n - nrpos - fp)

        return {'TP':tp,'TN':tn,'FP':fp,'FN':fn,'N':n}

    def confusion_mets_apply_iter(self,thresrow):
        t = thresrow['Threshold']
        wtd_vals = self.wtd_vals
        c_wtd_vals = self.c_wtd_vals
        ref_const = self.ref_const

        tp_idx = wtd_vals <= t
        tn_idx = (wtd_vals > (t + ref_const*255))
        fp_idx = (wtd_vals <= (t + ref_const*255)) & (wtd_vals >= ref_const*255)
        fn_idx = (wtd_vals <= 255) & (wtd_vals > t)
        
        thresrow['TP'] = c_wtd_vals[tp_idx].sum()
        thresrow['TN'] = c_wtd_vals[tn_idx].sum()
        thresrow['FP'] = c_wtd_vals[fp_idx].sum()
        thresrow['FN'] = c_wtd_vals[fn_idx].sum()

        return thresrow

    def confusion_mets_all_thresholds(self,ref,sys,w):
        r = ref.bwmat
        s = sys.matrix
        ref_const = (1 << 8)
        w_const = (1 << 16)

        t_list = np.unique(s).tolist()
        t_list = [-1] + t_list
#        thresMet_fields = ['Reference Mask','System Output Mask','Threshold','NMM','MCC','BWL1','TP','TN','FP','FN','N']
        thresMets = pd.DataFrame({'Reference Mask':ref.name,
                                  'System Output Mask':sys.name,
                                  'Threshold':t_list,
                                  'NMM':-1.,
                                  'MCC':0.,
                                  'BWL1':1.,
                                  'TP':0,
                                  'TN':0,
                                  'FP':0,
                                  'FN':0,
                                  'N':0})

        full_composite = s + r*ref_const + (1-w)*w_const
        all_vals,c_all_vals = np.unique(full_composite,return_counts=True)
        c_wtd_vals = c_all_vals[all_vals < w_const]
        wtd_vals = all_vals[all_vals < w_const]
        self.ref_const = ref_const
        self.wtd_vals = wtd_vals
        self.c_wtd_vals = c_wtd_vals
        thresMets['N'] = c_wtd_vals.sum()
        thresMets = thresMets.apply(self.confusion_mets_apply_iter,axis=1) 

        #want to vectorize computation of the metrics here
        thresMets['MCC'] = thresMets.apply(self.matthews,axis=1)
        thresMets['NMM'] = thresMets.apply(self.NimbleMaskMetric,axis=1)
        thresMets['BWL1'] = thresMets.apply(self.binaryWeightedL1,axis=1)

        return thresMets

    def NimbleMaskMetric(self,conf,c=-1):
        """
        * Metric: NMM
        * Description: this function calculates the system mask score
                                     based on the confusion_measures function
        * Inputs:
        *     conf: the confusion measures. Made explicit here to demonstrate the metric's
                    dependency on the confusion measures
        *     ref: the reference mask object
        *     w: the weight matrix
        *     c: hinge value, the cutoff at which the region is scored (default: -1)
        * Output:
        *     NMM score in range [c, 1]
        """
        tp = conf['TP']
        fn = conf['FN']
#        Rgt=np.sum((ref.bwmat==0) & (w==1))
        Rgt = tp + fn
        if Rgt == 0:
            return np.nan
        fp = conf['FP']
        return max(c,float(tp-fn-fp)/Rgt)

    def matthews(self,conf):
        """
        * Metric: MCC (Matthews correlation coefficient)
        * Description: this function calculates the system mask score
                       based on the MCC function
        * Input:
        *     conf: the confusion measures. Made explicit here to demonstrate the metric's
                    dependency on the confusion measures
        * Output:
        *     Score in range [0, 1]
        """
        tp = conf['TP']
        fp = conf['FP']
        tn = conf['TN']
        fn = conf['FN']
        n = conf['N']

        if n == 0:
            return np.nan

        s=tp+fn
        p=tp+fp

        if (s==n) or (p==n) or (s==0) or (p==0):
            score=0.0
        else:
            score=Decimal(tp*tn-fp*fn)/Decimal((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)).sqrt()
            score = float(score)
        return score

    def hamming(self,ref,sys):
        """
        * Metric: Hamming distance
        * Description: this function calculates the Hamming distance
                       between the reference mask and the system output mask
                      This metric is no longer called in getMetrics.
        * Inputs:
        *     ref: the reference mask object
        *     sys: the system output mask object
        * Output:
        *     Hamming distance value
        """

        rmat = ref.bwmat.astype(int)
        smat=0
        if self.sys_threshold >= -1:
            smat=sys.matrix.astype(int)
        else:
            smat=sys.bwmat.astype(int)

        if (rmat.shape[0]==0) or (rmat.shape[1]==0):
            return np.nan
        ham = np.sum(abs(rmat - smat))/255./(rmat.shape[0]*rmat.shape[1])
        #ham = sum([abs(rmat[i] - mask[i])/255. for i,val in np.ndenumerate(rmat)])/(rmat.shape[0]*rmat.shape[1]) #xor the r and s
        return ham

    def binaryWeightedL1(self,conf):
        """
        * Metric: binary Weighted L1
        * Description: this function calculates the weighted L1 loss
                       for the binarized mask and normalizes the value
                       with the no score zone
        * Inputs:
        *     conf: the confusion measures. Made explicit here to demonstrate the metric's
                    dependency on the confusion measures
        * Outputs:
        *     Normalized binary WL1 value
        """
        n=conf['N'] #expect w to be 0 or 1, but otherwise, allow to be a naive sum for the sake of flexibility
        if n == 0:
            return np.nan

        norm_wL1=float(conf['FP'] + conf['FN'])/n
        return norm_wL1

#    def binaryWeightedL1(self,ref,sys,w,th):
#        """
#        * Metric: binary Weighted L1
#        * Description: this function calculates the weighted L1 loss
#                       for the binarized mask and normalizes the value
#                       with the no score zone
#        * Inputs:
#        *     ref: the reference mask object
#        *     sys: the system output mask object
#        *     w: the weight matrix
#        *     th: the threshold for binarization
#        * Outputs:
#        *     Normalized binary WL1 value
#        """
#        if th is -1:
#            th = 254
#
#        n=np.sum(w) #expect w to be 0 or 1, but otherwise, allow to be a naive sum for the sake of flexibility
#        if n == 0:
#            return np.nan
#
#        rmat = ref.bwmat.astype(int)/255.
#        smat = sys.matrix
#        wL1=np.multiply(w,abs(rmat-(smat > th)))
#        wL1=np.sum(wL1)
#        #wL1=sum([wt*abs(rmat[j]-mask[j])/255 for j,wt in np.ndenumerate(w)])
#        norm_wL1=wL1/n
#        return norm_wL1

    @staticmethod
    def grayscaleWeightedL1(ref,sys,w):
        """
        * Metric: grayscale Weighted L1
        * Description: this function calculates the weighted L1 loss
                       for the unmodified grayscale mask and normalizes
                       the value with the no score zone.
                       The grayscale weighted L1 needs only be computed
                       once for each reference and system mask pair
                       and so is set as a static method
        * Inputs:
        *     ref: the reference mask object
        *     sys: the system output mask object
        *     w: the weight matrix
        * Outputs:
        *     Normalized grayscale WL1 value
        """
        n=w.sum() #expect w to be 0 or 1, but otherwise, allow to be a naive sum for the sake of flexibility
        if n == 0:
            return np.nan

        rmat = ref.bwmat.astype(float)
        smat = sys.matrix

        wL1=np.multiply(w,abs(rmat-smat)/255)
        wL1=wL1.sum()
        #wL1=sum([wt*abs(rmat[j]-mask[j])/255 for j,wt in np.ndenumerate(w)])
#        norm_wL1=wL1/n
        return wL1/n

    def hingeL1(self,ref,sys,w,e=0.1):
        """
        * Metric: Hinge L1
        * Description: this function calculates Hinge L1 loss
                                     and normalize the value with the no score zone
                      This metric is no longer called in getMetrics.
        * Inputs:
        *     ref: the reference mask object
        *     sys: the system output mask object
        *     w: the weight matrix
        *     e: the hinge value at which to truncate the loss. Below this value the loss is counted as 0. (default = 0.1)
        * Outputs:
        *     Normalized HL1 value
        """

        #if ((len(w) != len(r)) or (len(w) != len(s))):
        if (e < 0):
            print("Your chosen epsilon is negative. Setting to 0.")
            e=0
        rmat = ref.bwmat
        rArea=np.sum(rmat==0) #mask area
        wL1 = self.grayscaleWeightedL1(ref,sys,w)
        n=np.sum(w)
        if n == 0:
            return np.nan
        hL1=max(0,wL1-e*rArea/n)
        return hL1

    def assign_mets(self,row):
        """
        * Description: for pandas apply for running thresholds.
        """
        thismet = self.getMetrics(self.ref,self.sys,self.w,row['Threshold'])
        row['NMM'] = thismet['NMM']
        row['MCC'] = thismet['MCC']
        row['BWL1'] = thismet['BWL1']
        row['TP'] = thismet['TP']
        row['TN'] = thismet['TN']
        row['FP'] = thismet['FP']
        row['FN'] = thismet['FN']
        row['N'] = thismet['N']
        return row

    #computes metrics running over the set of thresholds for grayscale mask
    def runningThresholds(self,ref,sys,bns,sns,pns,erodeKernSize,dilateKernSize,distractionKernSize,kern,myprintbuffer=0):
        """
        * Description: this function computes the metrics over a set of thresholds given a grayscale mask

        * Inputs:
        *     ref: the reference mask object
        *     sys: the system output mask object
        *     bns: the boundary no-score weighted matrix
        *     sns: the selected no-score weighted matrix
        *     pns: the pixel no-score weighted matrix
        *     erodeKernSize: total length of the erosion kernel matrix
        *     dilateKernSize: total length of the dilation kernel matrix
        *     distractionKernSize: length of the dilation kernel matrix for the unselected no-score zones.
        *     kern: kernel shape to be used (default: 'box')
        *     myprintbuffer: buffer to store verbose printout for atomic printout.
                           This option is directly tied to the verbose option through MaskScorer.py
       
        * Outputs:
        *     thresMets: a dataframe of the computed threshold metrics
        *     tmax: the threshold yielding the maximum MCC 
        """

        #add bns/sns totals as well
        w = cv2.bitwise_and(bns,sns)
        weighted_weights = (1-bns) + 2*(1-sns)
        btotal = np.sum(weighted_weights == 1)
        stotal = np.sum(weighted_weights >= 2)
        ptotal = 0
        if pns is not 0:
            w = cv2.bitwise_and(w,pns)
            weighted_weights = weighted_weights + 4*(1-pns)
            ptotal = np.sum(weighted_weights >= 4)

#        if not (self.sys_threshold in uniques) and self.sys_threshold > -10:
        smat = sys.matrix
        uniques=np.unique(smat.astype(float))
        uniques=np.sort(np.append(uniques,-1)) #NOTE: adding the threshold that makes everything white. The threshold that makes everything black is already there.

        #for all thresholds
        #sys.binarize(0)
        self.myprintbuffer = myprintbuffer
#        thresMets = thresMets.apply(self.assign_mets,axis=1,reduce=False)
        thresMets = self.confusion_mets_all_thresholds(ref,sys,w)
        thresMets['BNS'] = btotal
        thresMets['SNS'] = stotal
        thresMets['PNS'] = ptotal

        if isinstance(thresMets,pd.Series):
            thresMets = thresMets.to_frame().transpose()

        #generate ROC dataframe for image, preferably from existing library.
        #TPR = TP/(TP + FN); FPR = FP/(FP + TN)

        #no need for roc curve if any of the denominator is zero
        nonNullRows = thresMets.query("(TP + FN > 0) or (FP + TN > 0)")

        #pick max threshold for max MCC
        columns = ['Threshold','NMM','MCC','BWL1','TP','TN','FP','FN','BNS','SNS','PNS','N']

        if nonNullRows.shape[0] > 0:
            tmax = thresMets['Threshold'].iloc[thresMets['MCC'].idxmax()]
            maxMets = thresMets.query("Threshold=={}".format(tmax))
            maxNMM = maxMets.iloc[0]['NMM']
            maxMCC = maxMets.iloc[0]['MCC']
            maxBWL1 = maxMets.iloc[0]['BWL1']
        else:
            tmax = np.nan
            maxNMM = np.nan
            maxMCC = np.nan
            maxBWL1 = np.nan

        thresMets = thresMets[columns]
        if myprintbuffer is not 0:
            myprintbuffer.append("NMM: {}".format(maxNMM))
            myprintbuffer.append("MCC: {}".format(maxMCC))
            myprintbuffer.append("BWL1: {}".format(maxBWL1))

        return thresMets,tmax

    def get_all_metrics(self,sbin,bns,sns,pns,erodeKernSize,dilateKernSize,distractionKernSize,kern,precision=16,round_modes=[],myprintbuffer=0):
        """
        * Description: get all the metrics for one probe
        """
        all_metrics = {}
        w = cv2.bitwise_and(bns,sns)
        if pns is not 0:
            w = cv2.bitwise_and(w,pns)

        thresMets,threshold = self.runningThresholds(self.ref,self.sys,bns,sns,pns,erodeKernSize,dilateKernSize,distractionKernSize,kern,myprintbuffer)
        all_metrics['OptimumThreshold'] = threshold
        thresMets['TPR'] = 0.
        thresMets['FPR'] = 0.

        nullRocQuery = "(TP + FN == 0) or (FP + TN == 0)"
        nonNullRocQuery = "(TP + FN > 0) and (FP + TN > 0)"
        nullRocRows = thresMets.query(nullRocQuery)
        nonNullRocRows = thresMets.query(nonNullRocQuery)

        #compute TPR and FPR here for rows that have it.
        if nullRocRows.shape[0] < thresMets.shape[0]:
#                thresMets.set_value(nonNullRocRows.index,'TPR',nonNullRocRows['TP']/(nonNullRocRows['TP'] + nonNullRocRows['FN']))
#                thresMets.set_value(nonNullRocRows.index,'FPR',nonNullRocRows['FP']/(nonNullRocRows['FP'] + nonNullRocRows['TN']))
            thresMets.at[nonNullRocRows.index,'TPR'] = nonNullRocRows['TP']/(nonNullRocRows['TP'] + nonNullRocRows['FN'])
            thresMets.at[nonNullRocRows.index,'FPR'] = nonNullRocRows['FP']/(nonNullRocRows['FP'] + nonNullRocRows['TN'])

        #set aside for numeric threshold. If threshold is nan, set everything to 0 or nan as appropriate, make the binarized system mask a whitemask2.png,
        #and pass to HTML accordingly
        myameas = {} #dictionary of actual metrics
        if np.isnan(threshold):
            metrics = thresMets.iloc[0]
            all_metrics['GWL1'] = np.nan

            for mes in ['BNS','SNS','PNS','N']:
                if myprintbuffer is not 0:
                    myprintbuffer.append("Setting value for {}...".format(mes))
                all_metrics[''.join(['Pixel',mes])] = metrics[mes]

            for mes in ['TP','TN','FP','FN']:#,'BNS','SNS','PNS']:
                if myprintbuffer is not 0:
                    myprintbuffer.append("Setting value for {}...".format(mes))
                all_metrics[''.join(['OptimumPixel',mes])] = metrics[mes]
                if sbin >= -1:
                    all_metrics[''.join(['ActualPixel',mes])] = metrics[mes]

            for met in ['NMM','MCC','BWL1']:
                if myprintbuffer is not 0:
                    myprintbuffer.append("Setting value for {}...".format(met))
                all_metrics[''.join(['Optimum',met])] = metrics[met]
                if sbin >= -1:
                    all_metrics[''.join(['Actual',met])] = metrics[met]
        else:
            metrics = thresMets.query('Threshold=={}'.format(threshold)).iloc[0]
            mets = metrics[['NMM','MCC','BWL1']].to_dict()
            mymeas = metrics[['TP','TN','FP','FN','N','BNS','SNS','PNS']].to_dict()
            rocvalues = thresMets[['TPR','FPR']]

            #append 0 and 1 to beginning and end of tpr and fpr respectively
            rocvalues = rocvalues.append(pd.DataFrame([[0,0]],columns=list(rocvalues)),ignore_index=True)
            #reindex rocvalues
            rocvalues = rocvalues.sort_values(by=['FPR','TPR'],ascending=[True,True]).reset_index(drop=True)

            #generate a plot and get detection metrics
            fpr = rocvalues['FPR']
            tpr = rocvalues['TPR']

            myauc = dmets.compute_auc(fpr,tpr)
            myeer = dmets.compute_eer(fpr,1-tpr)

            all_metrics['AUC'] = myauc
            all_metrics['EER'] = myeer

            for mes in ['BNS','SNS','PNS','N']:
                if myprintbuffer is not 0:
                    myprintbuffer.append("Setting value for {}...".format(mes))
                all_metrics[''.join(['Pixel',mes])] = mymeas[mes]

            if sbin >= -1:
                #just get scores in one run if threshold is chosen
                #TODO: threshold table lookup this instead.
                self.sys.binarize(sbin)
                myameas = self.getMetrics(self.ref,self.sys,w,sbin,myprintbuffer)
                all_metrics['ActualThreshold'] = sbin

            mets['GWL1'] = maskMetrics.grayscaleWeightedL1(self.ref,self.sys,w)
            all_metrics['GWL1'] = myround(mets['GWL1'],precision,round_modes)
            for met in ['NMM','MCC','BWL1']:
                if myprintbuffer is not 0:
                    myprintbuffer.append("Setting value for {}...".format(met))
#                    print("Optimum{}: {}, strlen: {}".format(met,myround(mets[met],precision,truncate),len(str(myround(mets[met],precision,truncate))))) # debug flag
                all_metrics['Optimum%s' % met] = myround(mets[met],precision,round_modes)
                if sbin >= -1:
                    #record Actual metrics
                    actual_met_name = 'Actual%s' % met
                    actual_met = myround(myameas[met],precision,round_modes)
                    all_metrics[actual_met_name] = actual_met
                    mets[actual_met_name] = actual_met

            for mes in ['TP','TN','FP','FN']:#,'BNS','SNS','PNS']:
                if myprintbuffer is not 0:
                    myprintbuffer.append("Setting value for {}...".format(mes))
                all_metrics[''.join(['OptimumPixel',mes])] = mymeas[mes]
                if sbin >= -1:
                    all_metrics[''.join(['ActualPixel',mes])] = myameas[mes]

        return all_metrics,thresMets
