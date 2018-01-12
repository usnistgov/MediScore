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
from constants import *

class maskMetrics:
    """
    This class evaluates the metrics for the reference and system output masks.
    The image parameters necessary to evaluate most of the objects are included
    in the initialization.
    """
    def __init__(self,ref,sys,w,systh=-10):
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
#        if systh >= 0:
#            sys.binarize(systh)
#        else:
#            distincts = np.unique(sys.matrix)
#            if (np.array_equal(distincts,[0,255])) or (np.array_equal(distincts,[0])) or (np.array_equal(distincts,[255])): #already binarized or uniform, relies on external pipeline
#                sys.bwmat = sys.matrix

        #pass threshold as a parameter here
        self.conf = self.confusion_measures(ref,sys,w,systh)

        #record this dictionary of parameters
        self.nmm = self.NimbleMaskMetric(self.conf,ref,w)
        self.mcc = self.matthews(self.conf)
        self.bwL1 = self.binaryWeightedL1(self.conf)
#        self.bwL1 = self.binaryWeightedL1(ref,sys,w,systh)

    def getMetrics(self,myprintbuffer):
        """
        * Description: this function calculates the metrics with an implemented no-score zone.
                       Due to its repeated use for the same reference and system masks, the
                       getMetrics function excludes the GWL1

        * Output:
        *     dictionary of the NMM, MCC, BWL1, and the confusion measures.
        """

        #for nicer printout
        myprintbuffer.append("NMM: {}".format(self.nmm))
        myprintbuffer.append("MCC (Matthews correlation coeff.): {}".format(self.mcc))
        myprintbuffer.append("Binary Weighted L1: {}".format(self.bwL1))
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

        metrics = {'NMM':self.nmm,'MCC':self.mcc,'BWL1':self.bwL1}
        metrics.update(self.conf)
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
        n = np.sum(mywts)
        rpos = (r==0) & mywts
        nrpos = np.sum(rpos)
        rneg = (r==255) & mywts

        tp = np.float64(np.sum(s & rpos))
        fp = np.float64(np.sum(s & rneg))
        fn = np.float64(nrpos - tp)
        tn = np.float64(n - nrpos - fp)

        return {'TP':tp,'TN':tn,'FP':fp,'FN':fn,'N':n}

    def NimbleMaskMetric(self,conf,ref,w,c=-1):
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
        return max(c,(tp-fn-fp)/Rgt)

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

        norm_wL1=(conf['FP'] + conf['FN'])/n
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
#        #TODO: take stuff from conf too?
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
        n=np.sum(w) #expect w to be 0 or 1, but otherwise, allow to be a naive sum for the sake of flexibility
        if n == 0:
            return np.nan

        rmat = ref.bwmat.astype(int)
        smat = sys.matrix.astype(int)

        wL1=np.multiply(w,abs(rmat-smat)/255.)
        wL1=np.sum(wL1)
        #wL1=sum([wt*abs(rmat[j]-mask[j])/255 for j,wt in np.ndenumerate(w)])
        norm_wL1=wL1/n
        return norm_wL1

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

    #computes metrics running over the set of thresholds for grayscale mask
    def runningThresholds(self,ref,sys,bns,sns,pns,erodeKernSize,dilateKernSize,distractionKernSize,kern,myprintbuffer):
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
        smat = sys.matrix
        uniques=np.unique(smat.astype(float))
#        if not (self.sys_threshold in uniques) and self.sys_threshold > -10:
        uniques=np.sort(np.append(uniques,-1)) #NOTE: adding the threshold that makes everything white. The threshold that makes everything black is already there.

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

#        if len(uniques) == 1:
#            #if mask is uniformly black or uniformly white, assess for some arbitrary threshold
#            if (uniques[0] == 255) or (uniques[0] == 0):
#                thresMets = pd.DataFrame({'Reference Mask':ref.name,
#                                           'System Output Mask':sys.name,
#                                           'Threshold':0,
#                                           'NMM':[-1.],
#                                           'MCC':[0.],
#                                           'BWL1':[1.],
#                                           'TP':[0],
#                                           'TN':[0],
#                                           'FP':[0],
#                                           'FN':[0],
#                                           'BNS':btotal,
#                                           'SNS':stotal,
#                                           'PNS':ptotal,
#                                           'N':[0]})
#                mets = self.getMetrics(myprintbuffer)
#                for m in ['NMM','MCC','BWL1']:
#                    thresMets.set_value(0,m,mets[m])
#                for m in ['TP','TN','FP','FN','N']:
#                    thresMets.set_value(0,m,self.conf[m])
#            else:
#                #assess for both cases where we treat as all black or all white
#                thresMets = pd.DataFrame({'Reference Mask':ref.name,
#                                           'System Output Mask':sys.name,
#                                           'Threshold':127,
#                                           'NMM':[-1.]*2,
#                                           'MCC':[0.]*2,
#                                           'BWL1':[1.]*2,
#                                           'TP':[0]*2,
#                                           'TN':[0]*2,
#                                           'FP':[0]*2,
#                                           'FN':[0]*2,
#                                           'BNS':btotal,
#                                           'SNS':stotal,
#                                           'PNS':ptotal,
#                                           'N':[0]*2})
#                rownum=0
#                #sys.binarize(0)
#                for th in [uniques[0],255]:
#                    #sys.bwmat[sys.matrix==th] = 0
#                    #thismet = maskMetrics(ref,sys,w,-1) #avoid binarizing too much
#                    thismet = maskMetrics(ref,sys,w,th) #avoid binarizing too much
#
#                    thresMets.set_value(rownum,'Threshold',th)
#                    thresMets.set_value(rownum,'NMM',thismet.nmm)
#                    thresMets.set_value(rownum,'MCC',thismet.mcc)
#                    thresMets.set_value(rownum,'BWL1',thismet.bwL1)
#                    thresMets.set_value(rownum,'TP',thismet.conf['TP'])
#                    thresMets.set_value(rownum,'TN',thismet.conf['TN'])
#                    thresMets.set_value(rownum,'FP',thismet.conf['FP'])
#                    thresMets.set_value(rownum,'FN',thismet.conf['FN'])
#                    thresMets.set_value(rownum,'N',thismet.conf['N'])
#                    rownum=rownum+1
#        else:
        #get actual thresholds.
        thresholds=uniques.tolist()
        thresMets = pd.DataFrame({'Reference Mask':ref.name,
                                   'System Output Mask':sys.name,
                                   'Threshold':thresholds,
                                   'NMM':[-1.]*len(thresholds),
                                   'MCC':[0.]*len(thresholds),
                                   'BWL1':[1.]*len(thresholds),
                                   'TP':[0]*len(thresholds),
                                   'TN':[0]*len(thresholds),
                                   'FP':[0]*len(thresholds),
                                   'FN':[0]*len(thresholds),
                                   'BNS':btotal,
                                   'SNS':stotal,
                                   'PNS':ptotal,
                                   'N':[0]*len(thresholds)})
        #for all thresholds
        rownum=0
        #sys.binarize(0)
        for th in thresholds:
            #sys.bwmat[sys.matrix==th] = 0 #increasing thresholds
            #thismet = maskMetrics(ref,sys,w,-1)
            thismet = maskMetrics(ref,sys,w,th)
            thresMets.at[rownum,'Threshold'] = th
            thresMets.at[rownum,'NMM'] = thismet.nmm
            thresMets.at[rownum,'MCC'] = thismet.mcc
            thresMets.at[rownum,'BWL1'] = thismet.bwL1
            thresMets.at[rownum,'TP'] = thismet.conf['TP']
            thresMets.at[rownum,'TN'] = thismet.conf['TN']
            thresMets.at[rownum,'FP'] = thismet.conf['FP']
            thresMets.at[rownum,'FN'] = thismet.conf['FN']
            thresMets.at[rownum,'N'] = thismet.conf['N']
            rownum=rownum+1

        #generate ROC dataframe for image, preferably from existing library.
        #TPR = TP/(TP + FN); FPR = FP/(FP + TN)
        thresMets['TPR'] = 0
        thresMets['FPR'] = 0

        #no need for roc curve if any of the denominator is zero
        nullRows = thresMets.query("(TP + FN == 0) and (FP + TN == 0)")
        nonNullRows = thresMets.query("(TP + FN != 0) or (FP + TN != 0)")

        #pick max threshold for max MCC
        columns = ['Threshold','NMM','MCC','BWL1','TP','TN','FP','FN','BNS','SNS','PNS','N','TPR','FPR']

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
        myprintbuffer.append("NMM: {}".format(maxNMM))
        myprintbuffer.append("MCC: {}".format(maxMCC))
        myprintbuffer.append("BWL1: {}".format(maxBWL1))

        return thresMets,tmax

