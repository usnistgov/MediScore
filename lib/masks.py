#!/usr/bin/python

"""
 *File: masks.py
 *Date: 8/29/2016
 *Translation by: Daniel Zhou
 *Original Author: Yooyoung Lee
 *Status: Complete

 *Description: this code contains the image object for the ground truth image.
 The image object contains methods for evaluating the scores for the ground
 truth image.  


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
import matplotlib.pyplot as plt
import os
import random
from decimal import Decimal

#returns a kernel matrix
def getKern(opt,size):
    if (size % 2 == 0):
        raise Exception('ERROR: One of your kernel sizes is not an odd integer.')
    
    opt=opt.lower()
    if opt=='box':
        kern=cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
    elif opt=='disc':
        kern=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))
    elif opt=='diamond':
        #build own kernel out of numpy ndarray
        kern=np.ones((size,size),dtype=np.uint8)
        center=(size-1)/2
        for idx,x in np.ndenumerate(kern):
            if abs(idx[0]-center) + abs(idx[1]-center) > center:
                kern[idx]=0
    elif opt=='gaussian':
        sigma=0.3 #as used in the R implementation
        center=(size-1)/2
        x=np.asmatrix(np.linspace(-center,center,size))
        x=np.dot(x.transpose(),np.asmatrix(np.ones(size)))
        xsq=np.square(x)
        z=np.exp(-(xsq+xsq.transpose())/(2*sigma**2))
        kern=z/np.sum(z) #final normalization
    elif opt=='line':
        #45 degree line, or identity matrix
        kern=np.eye(erodeKernSize,dtype=np.uint8)
    else:
        print("I don't recognize your kernel. Please enter a valid kernel.")
    return kern

#erode and dilate take in numpy matrix representations.
#TODO: v2: scikit image this?
def erode(img,kern):
    if kern.dtype == np.float64:
        #convolve with image first and take the min
        kCenter = (kern.shape[0]-1)/2 #always assume odd kernel
        eImg = np.zeros(img.shape)
        xlen = img.shape[0]
        ylen = img.shape[1]
        xkern = kern.shape[0]
        ykern = kern.shape[1]
        #EDIT: any way to simplify this loop?
        for j in range(0,ylen):
            for i in range(0,xlen):
                if (i < kCenter):
                    if (j < kCenter):
                        subkern=kern[(kCenter-i):xkern,(kCenter-j):ykern]
                    elif (j > ylen - 1 - kCenter):
                        subkern=kern[(kCenter-i):xkern,0:(ylen - j + kCenter)]
                    else:
                        subkern=kern[(kCenter-i):xkern,:]
                elif (i > xlen - 1 - kCenter):
                    if (j < kCenter):
                        subkern=kern[0:(xlen - i + kCenter),(kCenter-j):ykern]
                    elif (j > ylen - 1 - kCenter):
                        subkern=kern[0:(xlen - i + kCenter),0:(ylen - j + kCenter)]
                    else:
                        subkern=kern[0:(xlen - i + kCenter),:]
                else:
                    if (j < kCenter):
                        subkern=kern[:,(kCenter-j):ykern]
                    elif (j > ylen - 1 - kCenter):
                        subkern=kern[:,0:(ylen - j + kCenter)]
                    else:
                        subkern=kern
                #Hadamard product kern with the img subset, subsetting both matrices as necessary
                smallimg=img[max(0,i-kCenter):min(xlen,i+kCenter+1),max(0,j-kCenter):min(ylen,j+kCenter+1)]
                #if smallimg is uniform (e.g. all white or all black) set pixel and skip multiplication stage.
                if (np.array_equal(smallimg[0,0]*np.ones(subkern.shape),smallimg)):
                    eImg[i,j] = smallimg[0,0]
                    continue
                hdprod=np.multiply(smallimg,subkern)
                eImg[i,j]=smallimg[smallimg==smallimg.flatten()[hdprod.argmin()]].min()
        eImg=eImg.astype(np.uint8)
    else:
        eImg=cv2.erode(img,kern,iterations=1)
    return eImg

def dilate(img,kern):
    if kern.dtype == np.float64:
        #convolve with image first and take the min
        kCenter = (kern.shape[0]-1)/2 #always assume odd kernel
        dImg = np.zeros(img.shape)
        xlen = img.shape[0]
        ylen = img.shape[1]
        xkern = kern.shape[0]
        ykern = kern.shape[1]
        #EDIT: any way to simplify this loop?
        for j in range(0,ylen):
            for i in range(0,xlen):
                if (i < kCenter):
                    if (j < kCenter):
                        subkern=kern[(kCenter-i):xkern,(kCenter-j):ykern]
                    elif (j > ylen - 1 - kCenter):
                        subkern=kern[(kCenter-i):xkern,0:(ylen - j + kCenter)]
                    else:
                        subkern=kern[(kCenter-i):xkern,:]
                elif (i > xlen - 1 - kCenter):
                    if (j < kCenter):
                        subkern=kern[0:(xlen - i + kCenter),(kCenter-j):ykern]
                    elif (j > ylen - 1 - kCenter):
                        subkern=kern[0:(xlen - i + kCenter),0:(ylen - j + kCenter)]
                    else:
                        subkern=kern[0:(xlen - i + kCenter),:]
                else:
                    if (j < kCenter):
                        subkern=kern[:,(kCenter-j):ykern]
                    elif (j > ylen - 1 - kCenter):
                        subkern=kern[:,0:(ylen - j + kCenter)]
                    else:
                        subkern=kern
                #Hadamard product kern with the img subset, subsetting both matrices as necessary
                smallimg=img[max(0,i-kCenter):min(xlen,i+kCenter+1),max(0,j-kCenter):min(ylen,j+kCenter+1)]
                if (np.array_equal(smallimg[0,0]*np.ones(subkern.shape),smallimg)):
                    dImg[i,j] = smallimg[0,0]
                    continue
                hdprod=np.multiply(smallimg,subkern)
                dImg[i,j]=smallimg[smallimg==smallimg.flatten()[hdprod.argmax()]].max()

        dImg=dImg.astype(np.uint8)
    else:
        dImg=cv2.dilate(img,kern,iterations=1)
    return dImg

class mask:
    #set readopt=1 to read in as RGB, readopt=-1 to include possibility of alpha channels. Default 0.
    def __init__(self,n,readopt=0):
        self.name=n
        self.matrix=cv2.imread(n,readopt)

    def get_dims(self):
        return [self.matrix.shape[0],self.matrix.shape[1]]

    def get_copy(self):
        mycopy = copy.deepcopy(self)
        mycopy.name = self.name[-4:] + '-2.png'
        return mycopy

    #returns binarized matrix without affecting base matrix
    def bw(self,threshold):
        ret,mymat = cv2.threshold(self.matrix,threshold,255,cv2.THRESH_BINARY)
        return mymat

    #turns to black all pixels less than perfectly white
    def flatten(self):
        return self.bw(254)

    #flips mask
    def binary_flip(self):
        return 1 - self.matrix/255.

    #dimensional validation against masks. Image assumed to have been entered as a matrix for flexibility.
    def dimcheck(self,img):
        return self.get_dims() == img.shape

    #overlays the mask on top of the grayscale image. If you want a color mask, reread the image in as a color mask.
    def overlay(self,imgName):
        mymat = np.copy(self.matrix)
        gImg = cv2.imread(imgName,0)
        gImg = np.dstack(gImg,gImg,gImg) 
        if len(self.matrix.shape)==2:
            mymat = np.dstack(mymat,mymat,mymat)
        elif len(self.matrix.shape)==4:
            print("{} cannot account for alpha channels yet.".format(self.name))
            return 1

        alpha=0.7
        overmat = cv2.addWeighted(mymat,alpha,gImg,1-alpha,0)
        return overmat

    def binarize3Channel(self,RThresh,GThresh,BThresh,v,g):
        if len(self.matrix.shape)==2:
            mymat = np.copy(self.matrix)
            th=min(RThresh,GThresh,BThresh)
            mymat[mymat <= th] = v
            mymat[mymat > th] = g
            return mymat
        elif len(self.matrix.shape)==4:
            print("{} cannot binarize a file with an alpha channel.".format(self.name))
            return 1
        else:
            dims = self.get_dims()
            bimg = np.zeros((dims[0],dims[1]))
            reds,greens,blues=cv2.split(self.matrix)
            upthresh = (reds <= RThresh) | (greens <= GThresh) | (blues <= BThresh)
            bimg[upthresh] = v
            bimg[~upthresh] = g
            return bimg
            
    #save mask to file
    def save(self,fname,compression=0):
        if fname[-4:] != '.png':
            print("You should only save {} as a png.".format(self.name))
            return 0
        params=list()
        params.append(cv.CV_IMWRITE_PNG_COMPRESSION)
        params.append(compression)
        cv2.imwrite(fname,self.matrix,compression)

class refmask(mask):

    def noScoreZone(self,erodeKernSize,dilateKernSize,opt):
        if ((erodeKernSize==0) and (dilateKernSize==0)):
            dims = self.get_dims()
            weight = np.ones(dims,dtype=np.uint8)
            return {'rimg':self.matrix,'wimg':weight}
    
        opt = opt.lower()
        eKern=getKern(opt,erodeKernSize)
        dKern=getKern(opt,dilateKernSize)

        #note: erodes relative to 0. We have to invert it twice to get the actual effects we want relative to 255.
        eImg=255-erode(255-self.matrix,eKern)
        dImg=255-dilate(255-self.matrix,dKern)

        weight=(eImg-dImg)/255 #note: eImg - dImg because black is treated as 0.
        wFlip=1-weight
    
        return {'wimg':wFlip,'eimg':eImg,'dimg':dImg}
            
    def confusion_measures(self,sys,w):
        """
        *Metric: confusion_measures
        *Description: this function calculates the values in the confusion matrix (TP, TN, FP, FN)
                                     between the reference mask and the system output mask,
                                     accommodating the no score zone
        
        *Inputs
        * sys: system output mask object, with binary or grayscale matrix data
        * w: binary data from weighted table generated from this mask
        
        *Outputs
        * MET: list of the TP, TN, FP, and FN area, and N (total score region)
        """
        r=self.matrix.astype(int) #otherwise, negative values won't be recorded
        s=sys.matrix.astype(int)
        x=np.multiply(w,(r-s)/255.) #entrywise product of w and difference between masks

        #white is 1, black is 0 
        y = 1+np.copy(x)
        y[~((x<=0) & (r==0))]=0 #set all values that don't fulfill criteria to 0
        y = np.multiply(w,y)
        tp = np.sum(abs(y)) #sum_same_values + sum_neg_values
 
        y = np.copy(x)
        y[~((x > 0) & (r==255))]=0 #set all values that don't fulfill criteria to 0
        fp = np.sum(y)
    
        fn = np.sum(np.multiply((r==0),w)) - tp
        tn = np.sum(np.multiply((r==255),w)) - fp

        mydims = r.shape
        n = mydims[0]*mydims[1] - np.sum(1-w[w<1])

        return {'TP':tp,'TN':tn,'FP':fp,'FN':fn,'N':n}
    
    #add mask scoring, with this as the GT. img is assumed to be a numpy matrix for flexibility of implementation.
    
    def NimbleMaskMetric(self,sys,w,c=-1):
        """
        *Metric: NMM
        *Description: this function calculates the system mask score
                                     based on the confusion_measures function
        * Inputs
        *     mask: system output mask with grayscale data
        *     w: binary data from weighted table generated from reference mask
        *     c: forgiveness value
        * Outputs
        *     Score range [c, 1]
        """
        met = self.confusion_measures(sys,w)
        tp = met['TP']
        fn = met['FN']
        fp = met['FP']
        rmat = self.matrix

        Rgt=np.sum((rmat==0) & (w==1))
        return max(c,(tp-fn-fp)/Rgt)

    def matthews(self,sys,w):
        """ 
        *Metric: MCC (Matthews correlation coefficient)
        *Description: this function calculates the system mask score
                                     based on the MCC function
        * Inputs
        *     sys: system output mask object, with binary or grayscale matrix data
        *     w: binary data from weighted table generated from reference mask
        * Outputs
        *     Score range [0, 1]
        """
        met = self.confusion_measures(sys,w)
        tp = met['TP']
        tn = met['TN']
        fn = met['FN']
        fp = met['FP']
        n = met['N']

        s=(tp+fn)/n
        p=(tp+fp)/n
        if ((s==1) or (p==1) or (s==0) or (p==0)):
            score=0.0
        else:
            score=Decimal(tp*tn-fp*fn)/Decimal((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)).sqrt()
            score = float(score)
        return score
    
    def hamming(self,sys):
        """
        *Description: this function calculates Hamming distance
                                     between the reference mask and the system output mask
        
        *Inputs
        *     sys: system output mask object, with binary or grayscale matrix data
        
        * Outputs
            * Hamming distance value
        """

        rmat = self.matrix
        smat = sys.matrix
        ham = np.sum(abs(rmat - smat))/255./(rmat.shape[0]*rmat.shape[1])
        #ham = sum([abs(rmat[i] - mask[i])/255. for i,val in np.ndenumerate(rmat)])/(rmat.shape[0]*rmat.shape[1]) #xor the r and s
        return ham
    
    def weightedL1(self,sys,w):
        """
        *Description: this function calculates Weighted L1 loss
                                     and normalize the value with the no score zone
        
        * Inputs
        *     sys: system output mask object, with binary or grayscale matrix data
        *     w: binary data from weighted table generated from reference mask
        
        * Outputs
            * Normalized WL1 value
        """
        
        rmat = self.matrix
        smat = sys.matrix

        wL1=np.multiply(w,abs(rmat-smat))
        wL1=np.sum(wL1)/255.
        #wL1=sum([wt*abs(rmat[j]-mask[j])/255 for j,wt in np.ndenumerate(w)])
        n=np.sum(w) #expect w to be 0 or 1, but otherwise, allow to be a naive sum for the sake of flexibility
        norm_wL1=wL1/n
        return norm_wL1
    
    def hingeL1(self,sys,w,e=0.1):
        """
         *Description: this function calculates Hinge L1 loss
                                     and normalize the value with the no score zone
        
        * Inputs
        *     sys: system output mask object, with binary or grayscale matrix data
        *     w: binary data from weighted table generated from reference mask
        *     e: epsilon (default = 0.1)
        
        * Outputs
            * Normalized HL1 value"
        """

        #if ((len(w) != len(r)) or (len(w) != len(s))):
        if (e < 0):
            print("Your chosen epsilon is negative. Setting to 0.")
            e=0
        rmat = self.matrix.astype(np.float64)
        smat = sys.matrix.astype(np.float64)
        wL1=np.multiply(w,(rmat-smat))
        wL1=np.sum(abs(wL1))/255.
        rArea=np.sum(rmat==0) #mask area
        hL1=max(0,wL1-e*rArea)    
        n=np.sum(w)
        norm_hL1=hL1/n
        return norm_hL1

    #computes metrics running over the set of thresholds
    def runningThresholds(self,sys,erodeKernSize,dilateKernSize,kern='box',popt=0):
        smat = sys.matrix
        uniques=np.unique(smat.astype(float))
        if len(uniques) == 1:
            #if mask is uniformly black or uniformly white, assess as is
            if (uniques[0] == 255) or (uniques[0] == 0):
                thresMets = pd.DataFrame({'Reference Mask':self.name,
                                           'System Output Mask':sys.name,
                                           'Threshold':127.,
                                           'NMM':[-1024.],
                                           'MCC':[-1024.],
                                           'HAM':[-1024.],
                                           'WL1':[-1024.],
                                           'HL1':[-1024.]})
                mets = self.getMetrics(sys,erodeKernSize,dilateKernSize,kern=kern,popt=popt)
                for m in ['NMM','MCC','HAM','WL1','HL1']:
                    thresMets.set_value(0,m,mets[m])
            else:
                #assess for both cases where we treat as all black or all white
                thresMets = pd.DataFrame({'Reference Mask':self.name,
                                           'System Output Mask':sys.name,
                                           'Threshold':127.,
                                           'NMM':[-1024.]*2,
                                           'MCC':[-1024.]*2,
                                           'HAM':[-1024.]*2,
                                           'WL1':[-1024.]*2,
                                           'HL1':[-1024.]*2})
                rownum=0
                for th in [uniques[0]-1,uniques[0]+1]:
                    tmask = sys.get_copy()
                    tmask.matrix = tmask.bw(th)
                    thresMets.set_value(rownum,'Threshold',th)
                    thresMets.set_value(rownum,'NMM',self.NimbleMaskMetric(tmask,w))
                    thresMets.set_value(rownum,'MCC',self.matthews(tmask,w))
                    thresMets.set_value(rownum,'HAM',self.hamming(tmask))
                    thresMets.set_value(rownum,'WL1',self.weightedL1(tmask,w))
                    thresMets.set_value(rownum,'HL1',self.hingeL1(tmask,w))
                    rownum=rownum+1
        else:
            thresholds=map(lambda x,y: (x+y)/2.,uniques[:-1],uniques[1:]) #list of thresholds
            thresMets = pd.DataFrame({'Reference Mask':self.name,
                                       'System Output Mask':sys.name,
                                       'Threshold':127.,
                                       'NMM':[-1024.]*len(thresholds),
                                       'MCC':[-1024.]*len(thresholds),
                                       'HAM':[-1024.]*len(thresholds),
                                       'WL1':[-1024.]*len(thresholds),
                                       'HL1':[-1024.]*len(thresholds)})
            #for all thresholds
            noScore = self.noScoreZone(erodeKernSize,dilateKernSize,kern)
            w = noScore['wimg']
            rownum=0
            for th in thresholds:
                tmask = sys.get_copy()
                tmask.matrix = tmask.bw(th)
                thresMets.set_value(rownum,'Threshold',th)
                thresMets.set_value(rownum,'NMM',self.NimbleMaskMetric(tmask,w))
                thresMets.set_value(rownum,'MCC',self.matthews(tmask,w))
                thresMets.set_value(rownum,'HAM',self.hamming(tmask))
                thresMets.set_value(rownum,'WL1',self.weightedL1(tmask,w))
                thresMets.set_value(rownum,'HL1',self.hingeL1(tmask,w))
                rownum=rownum+1
        return thresMets
 
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


    def getMetrics(self,sys,erodeKernSize,dilateKernSize,thres=0,kern='box',popt=0):

        #preset values
        nmm = -999
        mcc = -999
        ham = -999
        wL1 = -999
        hL1 = -999

        if thres > 0:
            if popt==1:
                print("Converting mask to binary with threshold %0.2f" % thres)
            sys.matrix=sys.bw(thres)

        noScore = self.noScoreZone(erodeKernSize,dilateKernSize,kern)
        eImg = noScore['eimg']
        dImg = noScore['dimg']
        w = noScore['wimg']

        nmm = self.NimbleMaskMetric(sys,w)
        if popt==1:
            #for nicer printout
            if (nmm==1) or (nmm==-1):
                print("NMM: %d" % nmm)
            else:
                print("NMM: %0.9f" % nmm)
        mcc = self.matthews(sys,w)
        if popt==1:
            if (mcc==1) or (mcc==-1):
                print("MCC: %d" % mcc)
            else:
                print("MCC (Matthews correlation coeff.): %0.9f" % mcc)
        ham = self.hamming(sys)
        if popt==1:
            if (ham==1) or (ham==0):
                print("HAM: %d" % ham)
            else:
                print("Hamming Loss: %0.9f" % ham)
        wL1 = self.weightedL1(sys,w)
        if popt==1:
            if (wL1==1) or (wL1==0):
                print("WL1: %d" % mcc)
            else:
                print("Weighted L1: %0.9f" % wL1)
        hL1 = self.hingeL1(sys,w)
        if popt==1:
            if (hL1==1) or (hL1==0):
                print("MCC: %d" % mcc)
            else:
                print("Hinge Loss L1: %0.9f" % hL1)

        return {'mask':sys.matrix,'wimg':w,'eImg':eImg,'dImg':dImg,'NMM':nmm,'MCC':mcc,'HAM':ham,'WL1':wL1,'HL1':hL1}

    #prints out the aggregate mask, reference and other data
    def coloredMask_opt1(self,sysImgName, maniImgName, sData, wData, eData, dData, outputMaskPath):
        #set new image as some RGB
        mydims = self.get_dims()
        mycolor = np.zeros((mydims[0],mydims[1],3),dtype=np.uint8)

        #flip all because black is 0 by default. Use the regions to determine where to color.
        b_sImg = 1-sData/255
        b_wImg = 1-wData
        b_eImg = 1-eData/255

        b_sImg[b_sImg != 0] = 1
        b_eImg[b_eImg != 0] = 2
        b_wImg[b_wImg != 0] = 4

        mImg = b_eImg + b_wImg + b_sImg

        #set pixels equal to some value:
        #red to false accept and false reject
        #blue to no-score zone
        #pink to no-score zone that intersects with system mask
        #yellow to system mask intersect with GT
        #black to true negatives

        mycolor[(mImg==1) | (mImg==2)] = [0,0,255] #either only system (FP) or only erode image (FN)
        mycolor[mImg==4] = [255,51,51] #no-score zone
        mycolor[mImg==5] = [255,51,255] #system intersecting with no-score zone
        mycolor[mImg==3] = [0,255,255] #system and erode image coincide (TP)

        #return path to mask
        outputMaskName = sysImgName.split('/')[-1]
        outputMaskBase = outputMaskName.split('.')[0]
        finalMaskName=outputMaskBase + "_colored.jpg"
        path=os.path.join(outputMaskPath,finalMaskName)
        #write the aggregate mask to file
        cv2.imwrite(path,mycolor)

        #also aggregate over the grayscale maniImgName for direct comparison
        maniImg = mask(maniImgName)
        mData = maniImg.matrix
        myagg = np.zeros((mydims[0],mydims[1],3),dtype=np.uint8)
        m3chan = np.reshape(np.kron(mData,np.uint8([1,1,1])),(mData.shape[0],mData.shape[1],3))
        myagg[mImg==0]=m3chan[mImg==0]

        #for modified images, weighted sum the colored mask with the grayscale
        alpha=0.7
        mData = np.kron(mData,np.uint8([1,1,1]))
        mData.shape=(mydims[0],mydims[1],3)
        modified = cv2.addWeighted(mycolor,alpha,mData,1-alpha,0)
        myagg[mImg!=0]=modified[mImg!=0]

        compositeMaskName=outputMaskBase + "_composite.jpg"
        compositePath = os.path.join(outputMaskPath,compositeMaskName)
        cv2.imwrite(compositePath,myagg)

        return {'mask':path,'agg':compositePath}

