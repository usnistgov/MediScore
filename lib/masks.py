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
def getKern(kernopt,size):
    if (size % 2 == 0):
        raise Exception('ERROR: One of your kernel sizes is not an odd integer.')

    kernopt=kernopt.lower()
    if kernopt=='box':
        kern=cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
    elif kernopt=='disc':
        kern=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))
    elif kernopt=='diamond':
        #build own kernel out of numpy ndarray
        kern=np.ones((size,size),dtype=np.uint8)
        center=(size-1)/2
        for idx,x in np.ndenumerate(kern):
            if abs(idx[0]-center) + abs(idx[1]-center) > center:
                kern[idx]=0
    elif kernopt=='gaussian':
        sigma=0.3 #as used in the R implementation
        center=(size-1)/2
        x=np.asmatrix(np.linspace(-center,center,size))
        x=np.dot(x.transpose(),np.asmatrix(np.ones(size)))
        xsq=np.square(x)
        z=np.exp(-(xsq+xsq.transpose())/(2*sigma**2))
        kern=z/np.sum(z) #final normalization
        kern[kern > 0] = 1
        kern = np.uint8(kern)
    elif kernopt=='line':
        #45 degree line, or identity matrix
        kern=np.eye(erodeKernSize,dtype=np.uint8)
    else:
        print("I don't recognize your kernel. Please enter a valid kernel from the following: ['box','disc','diamond','gaussian','line'].")
    return kern

class mask(object):
    #set readopt=1 to read in as RGB, readopt=-1 to include possibility of alpha channels. Default 0.
    def __init__(self,n,readopt=0):
        self.name=n
        self.matrix=cv2.imread(n,readopt)  #output own error message when catching error
        if self.matrix is None:
            masktype = 'System'
            if isinstance(self,refmask):
                masktype = 'Reference'
            print("{} mask file {} is unreadable.".format(masktype,n))
        self.bwmat = 0 #initialize bw matrix to zero. Substitute as necessary.

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
            mymat = np.dstack([mymat,mymat,mymat])

        alpha=0.7
        overmat = cv2.addWeighted(mymat,alpha,gImg,1-alpha,0)
        return overmat

    def intensityBinarize3Channel(self,RThresh,GThresh,BThresh,v,g):
        """
        * Description: generalized binarization based on intensity
        """
        dims = self.get_dims()
        bimg = np.zeros((dims[0],dims[1]))
        reds,greens,blues=cv2.split(self.matrix)
        upthresh = (reds <= RThresh) | (greens <= GThresh) | (blues <= BThresh)
        bimg[upthresh] = v
        bimg[~upthresh] = g
        bimg = np.uint8(bimg)
        return bimg

    def binarize3Channel(self):
        """
        * Description: sets all non-white pixels to black
        """
        bimg = intensityBinarize3Channel(254,254,254,0,255)
        return bimg

    #general binarize
    def binarize(self,threshold):
        if len(self.matrix.shape)==2:
            self.bwmat = self.bw(threshold)
            return self.bwmat
        else:
            self.bwmat = self.intensityBinarize3Channel(threshold,threshold,threshold,0,255)
            return self.bwmat

    #save mask to file
    def save(self,fname,compression=0,bw=False):
        if fname[-4:] != '.png':
            print("You should only save {} as a png. Remember to add on the prefix.".format(self.name))
            return 0
        params=list()
        params.append(cv.CV_IMWRITE_PNG_COMPRESSION)
        params.append(compression)
        outmat = self.matrix
        if bw:
            if np.array_equal(self.bwmat,0):
                self.bwmat = binarize(254)
            outmat = self.bwmat
        cv2.imwrite(fname,outmat,params)

class refmask(mask):

    def __init__(self,n,readopt=0,cs=[],tmt='all'):
        super(refmask,self).__init__(n,readopt)
        #store colors and corresponding type
        self.colors = [[int(p) for p in c.split(' ')[::-1]] for c in cs]
        self.targetManiType = tmt #just for the record

    def aggregateNoScore(self,erodeKernSize,dilateKernSize,kern):
        """
        *Description: this function calculates and generates the aggregate no score zone of the mask
                             by performing a bitwise and (&) on the elements of the noScoreZone and the
                             distractionNoScoreZone functions
        *Inputs
        * erodeKernSize: total length of the erosion kernel matrix
        * dilateKernSize: total length of the dilation kernel matrix
        * kern: kernel shape to be used 
        """
        baseNoScore = noScoreZone(erodeKernSize,dilateKernSize,kern)['wimg']
        distractionNoScore = distractionNoScoreZone(dilateKernSize,kern)
        wimg = cv2.bitwise_and(baseNoScore,distractionNoScore)

        return wimg


    def noScoreZone(self,erodeKernSize,dilateKernSize,kern):
        """
        *Description: this function calculates and generates the no score zone of the mask,
                             as well as the eroded and dilated masks for additional reference
        *Inputs
        * erodeKernSize: total length of the erosion kernel matrix
        * dilateKernSize: total length of the dilation kernel matrix
        * kern: kernel shape to be used 
        """
        if (erodeKernSize==0) and (dilateKernSize==0):
            dims = self.get_dims()
            weight = np.ones(dims,dtype=np.uint8)
            return {'rimg':self.matrix,'wimg':weight}

        mymat = self.binarize(254)
        kern = kern.lower()
        eKern=getKern(kern,erodeKernSize)
        dKern=getKern(kern,dilateKernSize)

        #note: erodes relative to 0. We have to invert it twice to get the actual effects we want relative to 255.
        eImg=255-cv2.erode(255-mymat,eKern,iterations=1)
        dImg=255-cv2.dilate(255-mymat,dKern,iterations=1)

        weight=(eImg-dImg)/255 #note: eImg - dImg because black is treated as 0.
        wFlip=1-weight

        return {'wimg':wFlip,'eimg':eImg,'dimg':dImg}

    def distractionNoScoreZone(self,dilateKernSize,kern):
        """
        *Description: this function calculates the no score zone of the mask regions counted as distractors.
                            The resulting mask is meant to be paired with the no score zone function above as
                            
                            It parses the BGR color codes to determine the color corresponding to the
                            type of manipulation
        *Inputs
        * dilateKernSize: total length of the dilation kernel matrix
        * kern: kernel shape to be used 
        """

        mymat = self.binarize(254)
        dims = self.get_dims()
        if dilateKernSize==0:
            weights = np.ones(dims,dtype=np.uint8)
            return weights

        kern = kern.lower()
        dKern=getKern(kern,dilateKernSize)

        
        #take all distinct 3-channel colors in mymat, subtract the colors that are reported, and then iterate
        notcolors = set(tuple(p) for m2d in mymat.matrix for p in m2d)
        for c in self.colors:
            notcolors = notcolors.remove(c)

        mybin = np.ones(dims[0],dims[1])
        for c in notcolors:
            #set equal to cs
            tbin = ~((mymat[:,:,0]==c[0]) & (mymat[:,:,1]==c[1]) & (mymat[:,:,2]==c[2]))
            tbin = tbin.astype(np.uint8)
            mybin = cv2.bitwise_and(mybin,tbin)

        #note: erodes relative to 0. We have to invert it twice to get the actual effects we want relative to 255.
        dImg=1-cv2.dilate(1-mybin,dKern,iterations=1)
        weights=dImg.astype(np.uint8)

        return weights


class maskMetrics:

    def __init__(self,ref,sys,w,systh=-1):
        #get masks for ref and sys
        if np.array_equal(ref.bwmat,0):
            ref.binarize(254) #get the black/white mask first if not already gotten
        self.refMask = ref
        self.sysMask = sys
        self.weights = w
        self.threshold = systh #applies only for system mask
        if systh==-1:
            self.conf = self.confusion_measures_gs()
        else:
            self.sysMask.binarize(systh)
            self.conf = self.confusion_measures()

    def confusion_measures_gs(self):
        """
        *Metric: confusion_measures_gs
        *Description: this function calculates the values in the confusion matrix (TP, TN, FP, FN)
                                     between the reference mask and a grayscale system output mask,
                                     accommodating the no score zone
                      This function is currently not used in the mask scoring scheme.

        *Outputs
        * MET: list of the TP, TN, FP, and FN area, and N (total score region)
        """
        r=self.refMask.bwmat.astype(int) #otherwise, negative values won't be recorded
        s=self.sysMask.matrix.astype(int)
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

    def confusion_measures(self):
        """
        *Metric: confusion_measures
        *Description: this function calculates the values in the confusion matrix (TP, TN, FP, FN)
                                     between the reference mask and a black and white system output mask,
                                     accommodating the no score zone

        *Outputs
        * MET: list of the TP, TN, FP, and FN area, and N (total score region)
        """
        r = self.refMask.bwmat.astype(int)
        s = self.sysMask.bwmat.astype(int)
        x = (r+s)/255.
        tp = np.float64(np.sum((x==0) & (w==1)))
        fp = np.float64(np.sum((x==1) & (r==255) & (w==1)))
        fn = np.float64(np.sum((x==1) & (w==1)) - fp)
        tn = np.float64(np.sum((x==2) & (w==1)))
        n = np.sum(w==1)

        return {'TP':tp,'TN':tn,'FP':fp,'FN':fn,'N':n}

    def NimbleMaskMetric(self,c=-1):
        """
        *Metric: NMM
        *Description: this function calculates the system mask score
                                     based on the confusion_measures function
        * Inputs
        *     c: forgiveness value
        * Outputs
        *     Score range [c, 1]
        """
        tp = self.conf['TP']
        fp = self.conf['FP']
        fn = self.conf['FN']
        Rgt=np.sum((self.refMask.bwmat==0) & (self.weights==1))
        return max(c,(tp-fn-fp)/Rgt)

    def matthews(self):
        """
        *Metric: MCC (Matthews correlation coefficient)
        *Description: this function calculates the system mask score
                                     based on the MCC function
        * Outputs
        *     Score range [0, 1]
        """
        tp = self.conf['TP']
        fp = self.conf['FP']
        tn = self.conf['TN']
        fn = self.conf['FN']
        n = self.conf['N']

        s=np.float64(tp+fn)/n
        p=np.float64(tp+fp)/n
        if (s==1) or (p==1) or (s==0) or (p==0):
            score=0.0
        else:
            score=Decimal(tp*tn-fp*fn)/Decimal((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)).sqrt()
            score = float(score)
        return score

    def hamming(self):
        """
        *Description: this function calculates the Hamming distance
                                     between the reference mask and the system output mask
                      This metric is no longer called in getMetrics.

        * Outputs
            * Hamming distance value
        """

        rmat = self.refMask.bwmat
        smat = self.sysMask.matrix
        ham = np.sum(abs(rmat - smat))/255./(rmat.shape[0]*rmat.shape[1])
        #ham = sum([abs(rmat[i] - mask[i])/255. for i,val in np.ndenumerate(rmat)])/(rmat.shape[0]*rmat.shape[1]) #xor the r and s
        return ham

    def weightedL1(self):
        """
        *Description: this function calculates Weighted L1 loss
                                     and normalize the value with the no score zone

        * Outputs
            * Normalized WL1 value
        """

        rmat = self.refMask.bwmat.astype(np.float64)
        smat = self.sysMask.matrix.astype(np.float64)

        wL1=np.multiply(self.weights,abs(rmat-smat)/255.)
        wL1=np.sum(wL1)
        #wL1=sum([wt*abs(rmat[j]-mask[j])/255 for j,wt in np.ndenumerate(w)])
        n=np.sum(self.weights) #expect w to be 0 or 1, but otherwise, allow to be a naive sum for the sake of flexibility
        norm_wL1=wL1/n
        return norm_wL1

    def hingeL1(self,e=0.1):
        """
        *Description: this function calculates Hinge L1 loss
                                     and normalize the value with the no score zone
                      This metric is no longer called in getMetrics.

        * Inputs
        *     e: the hinge value at which to truncate the loss. Below this value the loss is counted as 0. (default = 0.1)

        * Outputs
        *     Normalized HL1 value
        """

        #if ((len(w) != len(r)) or (len(w) != len(s))):
        if (e < 0):
            print("Your chosen epsilon is negative. Setting to 0.")
            e=0
        rmat = self.refMask.bwmat
        rArea=np.sum(rmat==0) #mask area
        wL1 = self.weightedL1()
        n=np.sum(self.weights)
        hL1=max(0,wL1-e*rArea/n)
        return hL1

    def getMetrics(self,erodeKernSize,dilateKernSize,colors = [],kern='box',popt=0):
        """
         *Description: this function calculates the metrics with an implemented no-score zone

        * Inputs
        *     erodeKernSize: total length of the erosion kernel matrix
        *     dilateKernSize: total length of the dilation kernel matrix
        *     kern: kernel shape to be used 
        """

        #preset values
        nmm = -1
        mcc = 0
        wL1 = 1

        noScore = self.noScoreZone(erodeKernSize,dilateKernSize,kern)
        
        eImg = noScore['eimg']
        dImg = noScore['dimg']
        w = self.refMask.aggregateNoScore(erodeKernSize,dilateKernSize,kern)

        nmm = self.NimbleMaskMetric()
        if popt==1:
            #for nicer printout
            if (nmm==1) or (nmm==-1):
                print("NMM: %d" % nmm)
            else:
                print("NMM: %0.9f" % nmm)
        mcc = self.matthews()
        if popt==1:
            if (mcc==1) or (mcc==-1):
                print("MCC: %d" % mcc)
            else:
                print("MCC (Matthews correlation coeff.): %0.9f" % mcc)
        wL1 = self.weightedL1()
        if popt==1:
            if (wL1==1) or (wL1==0):
                print("WL1: %d" % mcc)
            else:
                print("Weighted L1: %0.9f" % wL1)
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

        return {'mask':self.sysMask.matrix,'wimg':self.weights,'eImg':eImg,'dImg':dImg,'NMM':nmm,'MCC':mcc,'WL1':wL1}

    #computes metrics running over the set of thresholds for grayscale mask
    def runningThresholds(self,erodeKernSize,dilateKernSize,kern='box',popt=0):
        """
        *Description: this function computes the metrics over a set of thresholds given a grayscale mask

        *Inputs:
        * erodeKernSize: total length of the erosion kernel matrix
        * dilateKernSize: total length of the dilation kernel matrix
        * kern: kernel shape to be used (default: 'box')
        * popt: whether or not to print messages for getMetrics (default 0)
       
        *Outputs:
        * thresMets: a dataframe of the computed threshold metrics
        * tmax: the threshold yielding the maximum MCC 
        """
        smat = self.sysMask.matrix
        uniques=np.unique(smat.astype(float))
        if len(uniques) == 1:
            #if mask is uniformly black or uniformly white, assess for some arbitrary threshold
            if (uniques[0] == 255) or (uniques[0] == 0):
                thresMets = pd.DataFrame({'Reference Mask':self.refMask.name,
                                           'System Output Mask':self.sysMask.name,
                                           'Threshold':127.,
                                           'NMM':[-1.],
                                           'MCC':[0.],
                                           'WL1':[1.]})
                mets = self.getMetrics(erodeKernSize,dilateKernSize,kern=kern,popt=popt)
                for m in ['NMM','MCC','WL1']:
                    thresMets.set_value(0,m,mets[m])
            else:
                #assess for both cases where we treat as all black or all white
                thresMets = pd.DataFrame({'Reference Mask':self.refMask.name,
                                           'System Output Mask':self.sysMask.name,
                                           'Threshold':127.,
                                           'NMM':[-1.]*2,
                                           'MCC':[0.]*2,
                                           'WL1':[1.]*2})
                rownum=0
                for th in [uniques[0]-1,uniques[0]+1]:
                    tmask = self.sysMask.get_copy()
                    tmask.matrix = tmask.bw(th)
                    thismet = maskMetrics(self.refMask,tmask,self.weights,self.gs_score)

                    thresMets.set_value(rownum,'Threshold',th)
                    thresMets.set_value(rownum,'NMM',thismet.NimbleMaskMetric())
                    thresMets.set_value(rownum,'MCC',thismet.matthews())
                    thresMets.set_value(rownum,'WL1',thismet.weightedL1())
                    rownum=rownum+1
        else:
            thresholds=map(lambda x,y: (x+y)/2.,uniques[:-1],uniques[1:]) #list of thresholds
            thresMets = pd.DataFrame({'Reference Mask':self.refMask.name,
                                       'System Output Mask':self.sysMask.name,
                                       'Threshold':127.,
                                       'NMM':[-1.]*len(thresholds),
                                       'MCC':[0.]*len(thresholds),
                                       'WL1':[1.]*len(thresholds)})
            #for all thresholds
            noScore = self.noScoreZone(erodeKernSize,dilateKernSize,kern)
            w = noScore['wimg']
            rownum=0
            for th in thresholds:
                tmask = self.sysMask.get_copy()
                tmask.matrix = tmask.bw(th)
                thismet = maskMetrics(self.refMask,tmask,self.weights,self.gs_score)

                thresMets.set_value(rownum,'Threshold',th)
                thresMets.set_value(rownum,'NMM',thismet.NimbleMaskMetric())
                thresMets.set_value(rownum,'MCC',thismet.matthews())
                thresMets.set_value(rownum,'WL1',thismet.weightedL1())
                rownum=rownum+1

            #pick max threshold for max absolute MCC
            tmax = thresMets.query('abs(MCC)=={}'.format(min(abs(tmax['MCC']))))['Threshold'].iloc[0]

        return thresMets,tmax

    #prints out the aggregate mask, reference and other data
    def aggregateColorMask(self,maniImgName, eData, outputMaskPath):
        #set new image as some RGB
        mydims = self.refMask.get_dims()
        mycolor = np.zeros((mydims[0],mydims[1],3),dtype=np.uint8)

        #flip all because black is 0 by default. Use the regions to determine where to color.
        b_sImg = 1-self.sysMask.matrix/255
        b_wImg = 1-self.weights
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
        outputMaskName = self.sysMask.name.split('/')[-1]
        outputMaskBase = outputMaskName.split('.')[0]
        finalMaskName=outputMaskBase + "_colored.jpg"
        path=os.path.join(outputMaskPath,finalMaskName)
        #write the aggregate mask to file
        cv2.imwrite(path,mycolor)

        #also aggregate over the grayscale maniImgName for direct comparison
        maniImg = mask(maniImgName)
        mData = maniImg.matrix
        myagg = np.zeros((mydims[0],mydims[1],3),dtype=np.uint8)
        m3chan = np.stack((mData,mData,mData),axis=2)
        #np.reshape(np.kron(mData,np.uint8([1,1,1])),(mData.shape[0],mData.shape[1],3))
        myagg[mImg==0]=m3chan[mImg==0]

        #for modified images, weighted sum the colored mask with the grayscale
        alpha=0.7
        mData = np.stack((mData,mData,mData),axis=2)
        #np.kron(mData,np.uint8([1,1,1]))
        #mData.shape=(mydims[0],mydims[1],3)
        modified = cv2.addWeighted(mycolor,alpha,mData,1-alpha,0)
        myagg[mImg!=0]=modified[mImg!=0]

        compositeMaskName=outputMaskBase + "_composite.jpg"
        compositePath = os.path.join(outputMaskPath,compositeMaskName)
        cv2.imwrite(compositePath,myagg)

        return {'mask':path,'agg':compositePath}


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

