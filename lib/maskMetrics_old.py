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
import cv
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

class maskMetrics:
    """
    This class evaluates the metrics for the reference and system output masks.
    The image parameters necessary to evaluate most of the objects are included
    in the initialization.
    """
    def __init__(self,ref,sys,w,systh=-1):
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
        if np.array_equal(ref.bwmat,0):
            ref.binarize(254) #get the black/white mask first if not already gotten

        self.sys_threshold = systh
        if systh >= 0:
            sys.binarize(systh)
        else:
            distincts = np.unique(sys.matrix)
            if (np.array_equal(distincts,[0,255])) or (np.array_equal(distincts,[0])) or (np.array_equal(distincts,[255])): #already binarized or uniform, relies on external pipeline
                sys.bwmat = sys.matrix

        self.conf = self.confusion_measures(ref,sys,w)
        #record this dictionary of parameters
        self.nmm = self.NimbleMaskMetric(self.conf,ref,w)
        self.mcc = self.matthews(self.conf)
        self.bwL1 = self.binaryWeightedL1(ref,sys,w)

    def getMetrics(self,popt=0):
        """
        * Description: this function calculates the metrics with an implemented no-score zone.
                       Due to its repeated use for the same reference and system masks, the
                       getMetrics function excludes the GWL1

        * Output:
        *     dictionary of the NMM, MCC, BWL1, and the confusion measures.
        """

        if popt==1:
            #for nicer printout
            print("NMM: {}".format(self.nmm))
            print("MCC (Matthews correlation coeff.): {}".format(self.mcc))
            print("Binary Weighted L1: {}".format(self.bwL1))
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
        if self.sys_threshold >= 0:
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

    def confusion_measures(self,ref,sys,w):
        """
        * Metric: confusion_measures
        * Description: this function calculates the values in the confusion matrix (TP, TN, FP, FN)
                                     between the reference mask and a black and white system output mask,
                                     accommodating the no score zone
        * Inputs:
        *     ref: the reference mask object
        *     sys: the system output mask object
        *     w: the weight matrix
        * Output:
        *     dictionary of the TP, TN, FP, and FN areas, and total score region N
        """
        r = ref.bwmat.astype(int)
        #TODO: placeholder until something better covers it
        if sys.bwmat is 0:
            sys.binarize(254)

        s = sys.bwmat.astype(int)
        x = (r+s)/255.

        tp = np.float64(np.sum((x==0.) & (w==1)))
        fp = np.float64(np.sum((x==1.) & (r==255) & (w==1)))
        fn = np.float64(np.sum((x==1.) & (w==1)) - fp)
        tn = np.float64(np.sum((x==2.) & (w==1)))
        n = np.sum(w==1)

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
        fp = conf['FP']
        fn = conf['FN']
        Rgt=np.sum((ref.bwmat==0) & (w==1))
        if Rgt == 0:
            print("Mask {} has no region to score for the NMM.".format(ref.name))
            return np.nan
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
        if self.sys_threshold >= 0:
            smat=sys.matrix.astype(int)
        else:
            smat=sys.bwmat.astype(int)

        if (rmat.shape[0]==0) | (rmat.shape[1]==0):
            return np.nan

        ham = np.sum(abs(rmat - smat))/255./(rmat.shape[0]*rmat.shape[1])
        #ham = sum([abs(rmat[i] - mask[i])/255. for i,val in np.ndenumerate(rmat)])/(rmat.shape[0]*rmat.shape[1]) #xor the r and s
        return ham

    def binaryWeightedL1(self,ref,sys,w):
        """
        * Metric: binary Weighted L1
        * Description: this function calculates the weighted L1 loss
                       for the binarized mask and normalizes the value
                       with the no score zone
        * Inputs:
        *     ref: the reference mask object
        *     sys: the system output mask object
        *     w: the weight matrix
        * Outputs:
        *     Normalized binary WL1 value
        """

        rmat = ref.bwmat.astype(int)
        smat = sys.bwmat.astype(int)

        wL1=np.multiply(w,abs(rmat-smat)/255.)
        wL1=np.sum(wL1)
        #wL1=sum([wt*abs(rmat[j]-mask[j])/255 for j,wt in np.ndenumerate(w)])
        n=np.sum(w) #expect w to be 0 or 1, but otherwise, allow to be a naive sum for the sake of flexibility
        if n==0:
            return np.nan
        norm_wL1=wL1/n
        return norm_wL1

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

        rmat = ref.bwmat.astype(int)
        smat = sys.matrix.astype(int)

        wL1=np.multiply(w,abs(rmat-smat)/255.)
        wL1=np.sum(wL1)
        #wL1=sum([wt*abs(rmat[j]-mask[j])/255 for j,wt in np.ndenumerate(w)])
        n=np.sum(w) #expect w to be 0 or 1, but otherwise, allow to be a naive sum for the sake of flexibility
        if n==0:
            return np.nan
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
        if n==0:
            return np.nan
        hL1=max(0,wL1-e*rArea/n)
        return hL1

    #computes metrics running over the set of thresholds for grayscale mask
    def runningThresholds(self,ref,sys,bns,sns,pns,erodeKernSize,dilateKernSize,distractionKernSize,kern='box',popt=0):
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
        *     popt: whether or not to print messages for getMetrics.
                    This option is directly tied to the verbose option through MaskScorer.py
       
        * Outputs:
        *     thresMets: a dataframe of the computed threshold metrics
        *     tmax: the threshold yielding the maximum MCC 
        """
        smat = sys.matrix
        uniques=np.unique(smat.astype(float))

        #add bns/sns totals as well
        w = cv2.bitwise_and(bns,sns)
        weighted_weights = (1-bns) + 2*(1-sns)
        btotal = np.sum(weighted_weights == 1)
        stotal = np.sum(weighted_weights >= 2)
        ptotal = 0
        if pns is not 0:
            weighted_weights = weighted_weights + 4*(1-pns)
            w = cv2.bitwise_and(w,pns)
            ptotal = np.sum(weighted_weights >= 4)

        if len(uniques) == 1:
            #if mask is uniformly black or uniformly white, assess for some arbitrary threshold
            if (uniques[0] == 255) or (uniques[0] == 0):
                thresMets = pd.DataFrame({'Reference Mask':ref.name,
                                           'System Output Mask':sys.name,
                                           'Threshold':127,
                                           'NMM':[-1.],
                                           'MCC':[0.],
                                           'BWL1':[1.],
                                           'TP':[0],
                                           'TN':[0],
                                           'FP':[0],
                                           'FN':[0],
                                           'BNS':btotal,
                                           'SNS':stotal,
                                           'PNS':ptotal,
                                           'N':[0]})
                mets = self.getMetrics(popt=popt)
                for m in ['NMM','MCC','BWL1']:
                    thresMets.set_value(0,m,mets[m])
                for m in ['TP','TN','FP','FN','N']:
                    thresMets.set_value(0,m,self.conf[m])
            else:
                #assess for both cases where we treat as all black or all white
                thresMets = pd.DataFrame({'Reference Mask':ref.name,
                                           'System Output Mask':sys.name,
                                           'Threshold':127,
                                           'NMM':[-1.]*2,
                                           'MCC':[0.]*2,
                                           'BWL1':[1.]*2,
                                           'TP':[0]*2,
                                           'TN':[0]*2,
                                           'FP':[0]*2,
                                           'FN':[0]*2,
                                           'BNS':btotal,
                                           'SNS':stotal,
                                           'PNS':ptotal,
                                           'N':[0]*2})
                rownum=0
                #sys.binarize(0)
                for th in [uniques[0],255]:
                    #sys.bwmat[sys.matrix==th] = 0
                    #thismet = maskMetrics(ref,sys,w,-1) #avoid binarizing too much
                    thismet = maskMetrics(ref,sys,w,th) #avoid binarizing too much

                    thresMets.set_value(rownum,'Threshold',th)
                    thresMets.set_value(rownum,'NMM',thismet.nmm)
                    thresMets.set_value(rownum,'MCC',thismet.mcc)
                    thresMets.set_value(rownum,'BWL1',thismet.bwL1)
                    thresMets.set_value(rownum,'TP',thismet.conf['TP'])
                    thresMets.set_value(rownum,'TN',thismet.conf['TN'])
                    thresMets.set_value(rownum,'FP',thismet.conf['FP'])
                    thresMets.set_value(rownum,'FN',thismet.conf['FN'])
                    thresMets.set_value(rownum,'N',thismet.conf['N'])
                    rownum=rownum+1
        else:
            #get actual thresholds. Remove 255.
            thresholds=uniques.tolist()
            if 255 in thresholds:
                thresholds.remove(255)
            
            thresMets = pd.DataFrame({'Reference Mask':ref.name,
                                       'System Output Mask':sys.name,
                                       'Threshold':127,
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
                thresMets.set_value(rownum,'Threshold',th)
                thresMets.set_value(rownum,'NMM',thismet.nmm)
                thresMets.set_value(rownum,'MCC',thismet.mcc)
                thresMets.set_value(rownum,'BWL1',thismet.bwL1)
                thresMets.set_value(rownum,'TP',thismet.conf['TP'])
                thresMets.set_value(rownum,'TN',thismet.conf['TN'])
                thresMets.set_value(rownum,'FP',thismet.conf['FP'])
                thresMets.set_value(rownum,'FN',thismet.conf['FN'])
                thresMets.set_value(rownum,'N',thismet.conf['N'])
                rownum=rownum+1

        #generate ROC dataframe for image, preferably from existing library.
        #TPR = TP/(TP + FN); FPR = FP/(FP + TN)
        #no need for roc curve if any of the denominator is zero
        numNullRows = thresMets.query("(TP + FN == 0) or (FP + TN == 0)").shape[0]
        if numNullRows == 0:
            #set rows for ROC curve
            thresMets['TPR'] = thresMets['TP']/(thresMets['TP'] + thresMets['FN'])
            thresMets['FPR'] = thresMets['FP']/(thresMets['FP'] + thresMets['TN'])

        columns = ['Threshold','NMM','MCC','BWL1','TP','TN','FP','FN','BNS','SNS','PNS','N']
        if numNullRows == 0:
            columns = columns + ['TPR','FPR']

        #pick max threshold for max MCC
        tmax = thresMets['Threshold'].iloc[thresMets['MCC'].idxmax()]
        thresMets = thresMets[columns]
        if popt==1:
            maxMets = thresMets.query("Threshold=={}".format(tmax))
            maxNMM = maxMets.iloc[0]['NMM']
            maxMCC = maxMets.iloc[0]['MCC']
            maxBWL1 = maxMets.iloc[0]['BWL1']
            if (maxNMM==1) or (maxNMM==-1):
                print("NMM: %d" % maxNMM)
            else:
                print("NMM: %0.3f" % maxNMM)
            if (maxMCC==1) or (maxMCC==-1):
                print("MCC: %d" % maxMCC)
            else:
                print("MCC (Matthews correlation coeff.): %0.3f" % maxMCC)
            if (maxBWL1==1) or (maxBWL1==0):
                print("BWL1: %d" % maxBWL1)
            else:
                print("Binary Weighted L1: %0.3f" % maxBWL1)

        return thresMets,tmax

#    def getPlot(self,thresMets,metric='all',display=True,multi_fig=False):
#        """
#        *Description: this function plots a curve of the running threshold values
#			obtained from the above runningThreshold function
#
#        *Inputs
#            * thresMets: the DataFrame of metrics computed in the runningThreshold function
#            * metric: a string denoting the metrics to trace out on the plot. Default: 'all'
#            * display: whether or not to display the plot in a window. Default: True
#            * multi_fig: whether or not to save the plots for each metric on separate images. Default: False
#
#        * Outputs
#            * path where the plots for the function are saved
#        """
#        import Render as p
#        import json
#        from collections import OrderedDict
#        from itertools import cycle
#
#        #TODO: put this in Render.py. Combine later.
#        #generate plot options
#        ptitle = 'Running Thresholds'
#        if metric!='all':
#            ptitle=metric
#        mon_dict = OrderedDict([
#            ('title', ptitle),
#            ('plot_type', ptitle),
#            ('title_fontsize', 15),
#            ('xticks_size', 'medium'),
#            ('yticks_size', 'medium'),
#            ('xlabel', "Thresholds"),
#            ('xlabel_fontsize', 12),
#            ('ylabel', "Metric values"),
#            ('ylabel_fontsize', 12)])
#        with open('./plot_options.json', 'w') as f:
#            f.write(json.dumps(mon_dict).replace(',', ',\n'))
#
#        plot_opts = p.load_plot_options()
#        Curve_opt = OrderedDict([('color', 'red'),
#                                 ('linestyle', 'solid'),
#                                 ('marker', '.'),
#                                 ('markersize', 8),
#                                 ('markerfacecolor', 'red'),
#                                 ('label',None),
#                                 ('antialiased', 'False')])
#
#        opts_list = list() #do the same as in DetectionScorer.py. Generate defaults.
#        colors = ['red','blue','green','cyan','magenta','yellow','black']
#        linestyles = ['solid','dashed','dashdot','dotted']
#        # Give a random rainbow color to each curve
#        #color = iter(cm.rainbow(np.linspace(0,1,len(DM_List)))) #YYL: error here
#        color = cycle(colors)
#        lty = cycle(linestyles)
#
#        #TODO: make the metric plot option a list instead?
#        if metric=='all':
#            metvals = [thresMets[m] for m in ['NMM','MCC','HAM','WL1','HL1']]
#        else:
#            metvals = [thresMets[metric]]
#        thresholds = thresMets['Threshold']
#
#        for i in range(len(metvals)):
#            new_curve_option = OrderedDict(Curve_opt)
#            col = next(color)
#            new_curve_option['color'] = col
#            new_curve_option['markerfacecolor'] = col
#            new_curve_option['linestyle'] = next(lty)
#            opts_list.append(new_curve_option)
#
#        #a function defined to serve as the main plotter for getPlot. Put as separate function rather than nested?
#        def plot_fig(metvals,fig_number,opts_list,plot_opts,display, multi_fig=False):
#            fig = plt.figure(num=fig_number, figsize=(7,6), dpi=120, facecolor='w', edgecolor='k')
#
#            xtick_labels = range(0,256,15)
#            xtick = xtick_labels
#            x_tick_labels = [str(x) for x in xtick_labels]
#            ytick_labels = np.linspace(metvals.min(),metvals.max(),17)
#            ytick = ytick_labels
#            y_tick_labels = [str(y) for y in ytick_labels]
#
#            #TODO: faulty curve function. Get help.
#            print(len(opts_list))
#            print(len(metvals))
#            if multi_fig:
#                plt.plot(thresholds, metvals, **opts_list[fig_number])
#            else:
#                for i in range(len(metvals)):
#                    plt.plot(thresholds, metvals[i], **opts_list[i])
#
#            plt.plot((0, 1), '--', lw=0.5) # plot bisector
#            plt.xlim([0, 255])
#
#            #plot formatting options.
#            plt.xticks(xtick, x_tick_labels, size=plot_opts['xticks_size'])
#            plt.yticks(ytick, y_tick_labels, size=plot_opts['yticks_size'])
#            plt.suptitle(plot_opts['title'], fontsize=plot_opts['title_fontsize'])
#            plt.xlabel(plot_opts['xlabel'], fontsize=plot_opts['xlabel_fontsize'])
#            plt.ylabel(plot_opts['ylabel'], fontsize=plot_opts['ylabel_fontsize'])
#            plt.grid()
#
#    #        plt.legend(bbox_to_anchor=(0., -0.35, 1., .102), loc='lower center', prop={'size':8}, shadow=True, fontsize='medium')
#    #        fig.tight_layout(pad=7)
#
#            if opts_list[0]['label'] != None:
#                plt.legend(bbox_to_anchor=(0., -0.35, 1., .102), loc='lower center', prop={'size':8}, shadow=True, fontsize='medium')
#                # Put a nicer background color on the legend.
#                #legend.get_frame().set_facecolor('#00FFCC')
#                #plt.legend(loc='upper left', prop={'size':6}, bbox_to_anchor=(1,1))
#                fig.tight_layout(pad=7)
#
#            if display:
#                plt.show()
#
#            return fig
#
#        #different plotting options depending on multi_fig
#        if multi_fig:
#            fig_list = list()
#            for i,mm in enumerate(metvals):
#                fig = plot_fig(mm,i,opts_list,plot_opts,display,multi_fig)
#                fig_list.append(fig)
#            return fig_list
#        else:
#            fig = plot_fig(metvals[0],1,opts_list,plot_opts,display,multi_fig)
#            return fig

