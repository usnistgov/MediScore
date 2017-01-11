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
#import matplotlib.pyplot as plt
import os
import random
import masks
from decimal import Decimal
from string import Template

class maskMetricList:
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
                 journaldf=0,
                 mode='Probe',
                 colordict={'red':[0,0,255],'blue':[255,51,51],'yellow':[0,255,255],'green':[0,207,0],'pink':[193,182,255],'white':[255,255,255]}):
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
        - mode: determines the data to access. In most cases, it will be 'Probe', but
        - colordict: the dictionary of colors to use for the HTML output, in BGR array format,
                     to be used as reference
        """
        self.maskData = mergedf
        self.refDir = refD
        self.sysDir = sysD
        self.rbin = refBin
        self.sbin = sysBin
        self.journalData = journaldf
        self.mode=mode
        self.colordict=colordict
       
    def readMasks(self,refMaskFName,sysMaskFName,targetManiType,verbose):
        """
        * Description: reads both the reference and system output masks and caches the binarized image
                       into the reference mask. If the journal dataframe is provided, the color and purpose
                       of select mask regions will also be added to the reference mask
        * Inputs:
        *     refMaskFName: the name of the reference mask to be parsed
        *     sysMaskFName: the name of the system output mask to be parsed
        *     targetManiType: the target types to search to be manipulated
        *     verbose: permit printout from metrics
        * Outputs:
        *     rImg: the reference mask object
        *     sImg: the system output mask object
        """

        if verbose:
            print("Reference Mask: {}, System Mask: {}".format(refMaskFName,sysMaskFName))

        refMaskName = os.path.join(self.refDir,refMaskFName)
        sysMaskName = os.path.join(self.sysDir,sysMaskFName)
  
        if (self.journalData is 0) and (self.rbin == -1): #no journal saved and rbin not set
            self.rbin = 254 #automatically set binary threshold if no journalData provided.
 
        #read in the reference mask
        if self.rbin >= 0:
            rImg = masks.refmask(refMaskName)
            rImg.binarize(self.rbin)
        elif self.rbin == -1:
            #only need colors if selectively scoring
            myProbeID = self.maskData.query("ProbeMaskFileName=='{}' & OutputProbeMaskFileName=='{}'".format(refMaskFName,sysMaskFName))['ProbeFileID'].iloc[0]
            colorlist = list(self.journalData.query("ProbeFileID=='{}' & Evaluated=='Y'".format(myProbeID))['Color']) #get the target colors
            purposes = list(self.journalData.query("ProbeFileID=='{}' & Evaluated=='Y'".format(myProbeID))['Purpose'])
            purposes_unique = []
            [purposes_unique.append(p) for p in purposes if p not in purposes_unique]
            rImg = masks.refmask(refMaskName,cs=colorlist,tmt=purposes_unique)
            rImg.binarize(254)

            #check to see if the color in question is even present
            if targetManiType is not 'all':
                presence = 0
                for c in rImg.colors:
                    rmat = rImg.matrix
                    presence = presence + np.sum((rmat[:,:,0]==c[0]) & (rmat[:,:,1]==c[1]) & (rmat[:,:,2]==c[2]))
                    if presence > 0:
                        break
                if presence == 0:
                    if verbose:
                        print("The region you are looking for is not in reference mask {}. Scoring neglected.".format(refMaskFName))
                    return 0,0

        sImg = masks.mask(sysMaskName)
        return rImg,sImg 

    def getMetricList(self,targetManiType,erodeKernSize,dilateKernSize,distractionKernSize,kern,outputRoot,verbose,html,maniImageFName='',precision=16):
        """
        * Description: gets metrics for each pair of reference and system masks
        * Inputs:
        *     targetManiType: the target types to search to be manipulated
        *     erodeKernSize: length of the erosion kernel matrix
        *     dilateKernSize: length of the dilation kernel matrix
        *     distractionKernSize: length of the dilation kernel matrix for the unselected no-score zones.
                                   0 means nothing will be scored
        *     kern: kernel shape to be used
        *     outputRoot: the directory for outputs to be written
        *     verbose: permit printout from metrics
        *     html: whether or not to generate an HTML report
        *     maniImageFName: the list of Probe File images. Only relevant if html=True
        *     precision: the number of digits to round the computed metrics to.
        * Output:
        *     df: a dataframe of the computed metrics
        """
        #reflist and syslist should come from the same dataframe, so length checking is not required
        reflist = self.maskData['{}MaskFileName'.format(self.mode)]
        syslist = self.maskData['Output{}MaskFileName'.format(self.mode)]

        manip_ids = self.maskData['{}FileID'.format(self.mode)]
        if html:
            maniImageFName = self.maskData['{}FileName'.format(self.mode)]

        nrow = len(reflist) 
        #initialize empty frame to minimum scores 
        df=pd.DataFrame({'{}FileID'.format(self.mode):manip_ids,
                         'NMM':[-1.]*nrow,
                         'MCC': 0.,
                         'WL1': 1.,
                         'ColMaskFileName':['']*nrow,
                         'AggMaskFileName':['']*nrow})

        task = self.maskData['TaskID'].iloc[0] #should all be the same for one file

        for i,row in df.iterrows():
            if syslist[i] in [None,'',np.nan]:
                if verbose:
                    print("Empty system mask file at index %d" % i)
                continue
            else:
                rImg,sImg = self.readMasks(reflist[i],syslist[i],targetManiType,verbose)
                if (rImg is 0) and (sImg is 0):
                    #no masks detected with score-able regions, so set to not scored
                    self.journalData.set_value(i,'Evaluated','N')
                    df.set_value(i,'MCC',-2) #for reference to filter later
                    continue
                if (rImg.matrix is None) or (sImg.matrix is None):
                    #Likely this could be FP or FN. Set scores as usual.
                    if verbose:
                        print("The index is at %d." % i)
                    continue

                #save all images in their own directories instead, rather than pool it all in one subdirectory.
                #depending on whether manipulation or splice (see taskID), make the relevant subdir_name
                if task == 'manipulation':
                    subdir_name = self.maskData['ProbeFileID'].iloc[i]
                elif task == 'splice':
                    subdir_name = "{}_{}".format(self.maskData['ProbeFileID'].iloc[i],self.maskData['DonorFileID'].iloc[i])
                
                #save in subdirectory
                subOutRoot = os.path.join(outputRoot,subdir_name)
                if not os.path.isdir(subOutRoot):
                    os.system('mkdir ' + subOutRoot)

                #threshold before scoring if sbin >= 0. Otherwise threshold after scoring.
                if self.sbin >= 0:
                    sbin_name = os.path.join(subOutRoot,sImg.name.split('/')[-1][:-4] + '-bin.png')
                    sImg.save(sbin_name,th=self.sbin)
    
                #save the image separately for html and further review. Use that in the html report
                wts,bns,sns = rImg.aggregateNoScore(erodeKernSize,dilateKernSize,distractionKernSize,kern)

                #do a 3-channel combine with bns and sns for their colors before saving
                rImgbin = rImg.get_copy()
                rbin_name = os.path.join(subOutRoot,rImg.name.split('/')[-1][:-4] + '-bin.png')
                rbinmat = np.copy(rImgbin.bwmat)
                rImgbin.matrix = np.stack((rbinmat,rbinmat,rbinmat),axis=2)
                rImgbin.matrix[bns==0] = self.colordict['yellow']
                rImgbin.matrix[sns==0] = self.colordict['pink']
                rImgbin.save(rbin_name)

                #computes differently depending on choice to binarize system output mask
                mets = 0
                mymeas = 0
                threshold = 0
                metricRunner = maskMetrics(rImg,sImg,wts,self.sbin)
                if self.sbin >= 0:
                    #just get scores in one run if threshold is chosen
                    mets = metricRunner.getMetrics(popt=verbose)
                    mymeas = metricRunner.conf
                    threshold = self.sbin
                elif self.sbin == -1:
                    #get everything through an iterative run of max threshold
                    thresMets,threshold = metricRunner.runningThresholds(rImg,sImg,wts,erodeKernSize,dilateKernSize,kern=kern,popt=verbose)
                    #thresMets.to_csv(os.path.join(path_or_buf=outputRoot,'{}-thresholds.csv'.format(sImg.name)),index=False) #save to a CSV for reference
                    metrics = thresMets.query('Threshold=={}'.format(threshold)).iloc[0]
                    mets = metrics[['NMM','MCC','WL1']].to_dict()
                    mymeas = metrics[['TP','TN','FP','FN','N']].to_dict()

                if self.sbin == -1:
                    sbin_name = os.path.join(subOutRoot,sImg.name.split('/')[-1][:-4] + '-bin.png')
                    sImg.save(sbin_name,th=threshold)
 
                for met in ['NMM','MCC','WL1']:
                    df.set_value(i,met,round(mets[met],precision))
   
                if html:
                    maniImgName = os.path.join(self.refDir,maniImageFName[i])
                    colordirs = self.aggregateColorMask(rImg,sImg,bns,sns,kern,erodeKernSize,maniImgName,subOutRoot)
                    colMaskName=colordirs['mask']
                    aggImgName=colordirs['agg']
                    df.set_value(i,'ColMaskFileName',colMaskName)
                    df.set_value(i,'AggMaskFileName',aggImgName)
                    self.manipReport(task,subOutRoot,maniImageFName[i],rImg.name,sImg.name,rbin_name,sbin_name,threshold,bns,sns,mets,mymeas,targetManiType,colMaskName,aggImgName)

        return df

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
        hexcolor = (myr + myg + myb).upper()
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
    
    def manipReport(self,task,outputRoot,maniImageFName,rImg_name,sImg_name,rbin_name,sbin_name,sys_threshold,b_weights,s_weights,metrics,confmeasures,targetManiType,colMaskName,aggImgName):
        """
        * Description: this function assembles the HTML report for the manipulated image and is meant to be used solely by getMetricList
        * Inputs:
        *     task: the task over which the scorer is run
        *     outputRoot: the directory to deposit the weight image and the HTML file
        *     maniImageFName: the manipulated probe file name, relative to the reference directory (self.refDir) 
        *     rImg_name: the name of the unmodified reference image used for the mask evaluation
        *     sImg_name: the name of the unmodified system output image used for the mask evaluation
        *     rbin_name: the name of the binarized reference image used for the mask evaluation
        *     sbin_name: the name of the binarized system output image used for the mask evaluation
        *     sys_threshold: the threshold used to binarize the system output mask
        *     b_weights: the weighted matrix of the no-score zones of the targeted regions
        *     s_weights: the weighted matrix of the no-score zones generated from the non-target regions
        *     metrics: the dictionary of mask scores
        *     confmeasures: truth table measures evaluated between the reference and system output masks
        *     targetManiType: the target types to search to be manipulated
        *     colMaskName: the aggregate mask image of the ground truth, system output, and no-score regions
                           for the HTML report
        *     aggImgName: the above colored mask superimposed on a grayscale of the reference image
        """
        if not os.path.isdir(outputRoot):
            os.system('mkdir ' + outputRoot)

        bwts = np.uint8(255*b_weights)
        swts = np.uint8(255*s_weights)
        sysBase = os.path.basename(sImg_name)[:-4]
        weightFName = sysBase + '-weights.png'
        weightpath = os.path.join(outputRoot,weightFName)

        mywts = cv2.bitwise_and(b_weights,s_weights)

        dims = bwts.shape
        colwts = 255*np.ones((dims[0],dims[1],3),dtype=np.uint8)
        #combine the colors for bwts and swts to colwts
        colwts[bwts==0] = self.colordict['yellow']
        colwts[swts==0] = self.colordict['pink']

        cv2.imwrite(weightpath,colwts)

        mPath = os.path.join(self.refDir,maniImageFName)
        allshapes=min(dims[1],640) #limit on width for readability of the report

        # generate HTML files
        with open("html_template.txt", 'r') as f:
            htmlstr = Template(f.read())

        #dictionary of colors corresponding to confusion measures
        cols = {'tpcol':'green','fpcol':'red','tncol':'white','fncol':'blue','bnscol':'yellow','snscol':'pink'}
        hexs = self.nums2hex(cols.values()) #get hex strings from the BGR cols

        jtable = ''
        if task=='manipulation':
            myProbeID = self.maskData.query("ProbeFileName=='{}'".format(maniImageFName))['ProbeFileID'].iloc[0]
            jdata = self.journalData.query("ProbeFileID=='{}'".format(myProbeID))[['Purpose','Color','Evaluated']]
           
            #make those color cells empty with only the color as demonstration
            jDataColors = list(jdata['Color'])
            jDataColArrays = [x[::-1] for x in [c.split(' ') for c in jDataColors]]
            jDataColArrays = [[int(x) for x in c] for c in jDataColArrays]
            jDataHex = pd.Series([self.num2hex(c) for c in jDataColArrays],index=jdata.index) #match the indices
            jdata['Color'] = 'td bgcolor="#' + jDataHex + '"btd'
            jtable = jdata.to_html(index=False)
            jtable = jtable.replace('<td>td','<td').replace('btd','>')
        
        totalpx = np.sum(mywts==1)
        totalbns = np.sum(bwts==0)
        totalsns = np.sum(swts==0)

        #build soft links for mPath, rImg_name, sImg_name, use compact relative links for all
        mBase = os.path.basename(mPath)
        rBase = os.path.basename(rImg_name)
        sBase = os.path.basename(sImg_name)

        mPathNew = os.path.join(outputRoot,mBase)
        rPathNew = os.path.join(outputRoot,rBase)
        sPathNew = os.path.join(outputRoot,sBase)

        try:
            os.remove(mPathNew)
        except OSError:
            None
        try:
            os.remove(rPathNew)
        except OSError:
            None
        try:
            os.remove(sPathNew)
        except OSError:
            None

        os.symlink(os.path.abspath(mPath),mPathNew)
        os.symlink(os.path.abspath(rImg_name),rPathNew)
        os.symlink(os.path.abspath(sImg_name),sPathNew)

        htmlstr = htmlstr.substitute({'probeFname': mBase,
                                      'width': allshapes,
                                      'aggMask' : os.path.basename(aggImgName),
                                      'refMask' : rBase,
                                      'sysMask' : sBase,
                                      'binRefMask' : os.path.basename(rbin_name),
                                      'binSysMask' : os.path.basename(sbin_name),
                                      'systh' : sys_threshold,
                                      'noScoreZone' : os.path.basename(weightFName),
                                      'colorMask' : os.path.basename(colMaskName),
  				      'nmm' : round(metrics['NMM'],3),
  				      'mcc' : round(metrics['MCC'],3),
  				      'wL1' : round(metrics['WL1'],3),
                                      'totalPixels' : totalpx,
                                      'tp' : int(confmeasures['TP']),
                                      'fp' : int(confmeasures['FP']),
                                      'tn' : int(confmeasures['TN']),
                                      'fn' : int(confmeasures['FN']),
                                      'bns' : totalbns,
                                      'sns' : totalsns,
                                      'tpcol':cols['tpcol'],
                                      'fpcol':cols['fpcol'],
                                      'tncol':cols['tncol'],
                                      'fncol':cols['fncol'],
                                      'bnscol':cols['bnscol'],
                                      'snscol':cols['snscol'],
                                      'tphex':hexs[cols['tpcol']],
                                      'fphex':hexs[cols['fpcol']],
                                      'tnhex':hexs[cols['tncol']],
                                      'fnhex':hexs[cols['fncol']],
                                      'bnshex':hexs[cols['bnscol']],
                                      'snshex':hexs[cols['snscol']],
                                      'perctp':round(float(confmeasures['TP'])/totalpx,3),
                                      'percfp':round(float(confmeasures['FP'])/totalpx,3),
                                      'perctn':round(float(confmeasures['TN'])/totalpx,3),
                                      'percfn':round(float(confmeasures['FN'])/totalpx,3),
                                      'percbns':round(float(totalbns)/totalpx,3),
                                      'percsns':round(float(totalsns)/totalpx,3),
                                      'tmt':targetManiType,
                                      'jtable':jtable}) #add journal operations and set bg color to the html

        #print htmlstr
        fprefix=os.path.basename(maniImageFName)
        fprefix=fprefix.split('.')[0]
        fname=os.path.join(outputRoot,fprefix + '.html')
        myhtml=open(fname,'w')
        myhtml.write(htmlstr)
        myhtml.close()

    #prints out the aggregate mask, reference and other data
    def aggregateColorMask(self,ref,sys,bns,sns,kern,erodeKernSize,maniImgName,outputMaskPath):
        """
        *Description: this function produces the aggregate mask image of the ground truth, system output,
                      and no-score regions for the HTML report, and a composite of the same image superimposed
                      on a grayscale of the reference image

        * Inputs:
        *     ref: the reference mask file
        *     sys: the system output mask file to be evaluated
        *     bns: the boundary no-score weighted matrix
        *     sns: the selected no-score weighted matrix
        *     kern: kernel shape to be used
        *     erodeKernSize: length of the erosion kernel matrix
        *     maniImgName: a list of reference images (not masks) for superimposition
        *     outputMaskPath: the directory in which to output the composite images
       
        * Output
        *     a dictionary containing the colored mask path and the aggregate mask path
        """

        #set new image as some RGB
        mydims = ref.get_dims()
        mycolor = 255*np.ones((mydims[0],mydims[1],3),dtype=np.uint8)

        eKern = masks.getKern(kern,erodeKernSize)
        eData = 255 - cv2.erode(255 - ref.bwmat,eKern,iterations=1)

        #flip all because black is 0 by default. Use the regions to determine where to color.
        b_sImg = 1-sys.matrix/255
        b_eImg = 1-eData/255 #erosion of black/white reference mask
        b_bnsImg = 1-bns
        b_snsImg = 1-sns

        b_sImg[b_sImg != 0] = 1
        b_eImg[b_eImg != 0] = 2
        b_bnsImg[b_bnsImg != 0] = 4
        b_snsImg[b_snsImg != 0] = 8

        mImg = b_sImg + b_eImg + b_bnsImg + b_snsImg

        #set pixels equal to some value:
        #red to false accept and false reject
        #blue to no-score zone
        #pink to no-score zone that intersects with system mask
        #yellow to system mask intersect with GT
        #black to true negatives

        #get colors through self.colordict
        mycolor[mImg==1] = self.colordict['red'] #only system (FP)
        mycolor[mImg==2] = self.colordict['blue'] #only erode image (FN) (the part that is scored)
        mycolor[mImg==3] = self.colordict['green'] #system and erode image coincide (TP)
        mycolor[(mImg>=4) & (mImg <=7)] = self.colordict['yellow'] #boundary no-score zone
        mycolor[mImg>=8] = self.colordict['pink'] #selection no-score zone

        #return path to mask
        outputMaskName = sys.name.split('/')[-1]
        outputMaskBase = outputMaskName.split('.')[0]
        finalMaskName=outputMaskBase + "_colored.jpg"
        path=os.path.join(outputMaskPath,finalMaskName)
        #write the aggregate mask to file
        cv2.imwrite(path,mycolor)

        #also aggregate over the grayscale maniImgName for direct comparison
        maniImg = masks.mask(maniImgName)
        mData = maniImg.matrix
        myagg = np.zeros((mydims[0],mydims[1],3),dtype=np.uint8)
        m3chan = np.stack((mData,mData,mData),axis=2)
        #np.reshape(np.kron(mData,np.uint8([1,1,1])),(mData.shape[0],mData.shape[1],3))
        refbw = ref.bwmat
        myagg[refbw==255]=m3chan[refbw==255]

        #for modified images, weighted sum the colored mask with the grayscale
        alpha=0.7
        mData = np.stack((mData,mData,mData),axis=2)
        #np.kron(mData,np.uint8([1,1,1]))
        #mData.shape=(mydims[0],mydims[1],3)
        modified = cv2.addWeighted(ref.matrix,alpha,mData,1-alpha,0)
        myagg[refbw==0]=modified[refbw==0]

        compositeMaskName=outputMaskBase + "_composite.jpg"
        compositePath = os.path.join(outputMaskPath,compositeMaskName)
        cv2.imwrite(compositePath,myagg)

        return {'mask':path,'agg':compositePath}

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
        if systh >= 0:
            sys.binarize(systh)
        elif len(np.unique(sys.matrix)) <= 2: #already binarized or uniform
            sys.bwmat = sys.matrix

        self.sys_threshold = systh
        self.conf = self.confusion_measures(ref,sys,w)

        #record this dictionary of parameters
        self.nmm = self.NimbleMaskMetric(self.conf,ref,w)
        self.mcc = self.matthews(self.conf)
        self.wL1 = self.weightedL1(ref,sys,w)

    def getMetrics(self,popt=0):
        """
        * Description: this function calculates the metrics with an implemented no-score zone

        * Output:
        *     dictionary of the NMM, MCC, WL1, and the confusion measures.
        """

        if popt==1:
            #for nicer printout
            if (self.nmm==1) or (self.nmm==-1):
                print("NMM: %d" % self.nmm)
            else:
                print("NMM: %0.9f" % self.nmm)
            if (self.mcc==1) or (self.mcc==-1):
                print("MCC: %d" % self.mcc)
            else:
                print("MCC (Matthews correlation coeff.): %0.9f" % self.mcc)
            if (self.wL1==1) or (self.wL1==0):
                print("WL1: %d" % self.wL1)
            else:
                print("Weighted L1: %0.9f" % self.wL1)
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

        metrics = {'NMM':self.nmm,'MCC':self.mcc,'WL1':self.wL1}
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
        tp = np.float64(np.sum((x==0) & (w==1)))
        fp = np.float64(np.sum((x==1) & (r==255) & (w==1)))
        fn = np.float64(np.sum((x==1) & (w==1)) - fp)
        tn = np.float64(np.sum((x==2) & (w==1)))
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
            print("Mask {} has no region to score for the NMM. Defaulting to minimum score.".format(ref.name))
            return c
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

        s=np.float64(tp+fn)/n
        p=np.float64(tp+fp)/n
        if (s==1) or (p==1) or (s==0) or (p==0):
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

        ham = np.sum(abs(rmat - smat))/255./(rmat.shape[0]*rmat.shape[1])
        #ham = sum([abs(rmat[i] - mask[i])/255. for i,val in np.ndenumerate(rmat)])/(rmat.shape[0]*rmat.shape[1]) #xor the r and s
        return ham

    def weightedL1(self,ref,sys,w):
        """
        * Metric: Weighted L1
        * Description: this function calculates the weighted L1 loss
                       and normalizes the value with the no score zone
        * Inputs:
        *     ref: the reference mask object
        *     sys: the system output mask object
        *     w: the weight matrix
        * Outputs:
        *     Normalized WL1 value
        """

        rmat = ref.bwmat.astype(int)
        smat=0
        if self.sys_threshold >= 0:
            smat=sys.matrix.astype(int)
        else:
            smat=sys.bwmat.astype(int)

        wL1=np.multiply(w,abs(rmat-smat)/255.)
        wL1=np.sum(wL1)
        #wL1=sum([wt*abs(rmat[j]-mask[j])/255 for j,wt in np.ndenumerate(w)])
        n=np.sum(w) #expect w to be 0 or 1, but otherwise, allow to be a naive sum for the sake of flexibility
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
        wL1 = self.weightedL1(ref,sys,w)
        n=np.sum(w)
        hL1=max(0,wL1-e*rArea/n)
        return hL1

    #computes metrics running over the set of thresholds for grayscale mask
    def runningThresholds(self,ref,sys,w,erodeKernSize,dilateKernSize,kern='box',popt=0):
        """
        * Description: this function computes the metrics over a set of thresholds given a grayscale mask

        * Inputs:
        *     ref: the reference mask object
        *     sys: the system output mask object
        *     w: the weight matrix
        *     erodeKernSize: total length of the erosion kernel matrix
        *     dilateKernSize: total length of the dilation kernel matrix
        *     kern: kernel shape to be used (default: 'box')
        *     popt: whether or not to print messages for getMetrics.
                    This option is directly tied to the verbose option through MaskScorer.py
       
        * Outputs:
        *     thresMets: a dataframe of the computed threshold metrics
        *     tmax: the threshold yielding the maximum MCC 
        """
        smat = sys.matrix
        uniques=np.unique(smat.astype(float))
        if len(uniques) == 1:
            #if mask is uniformly black or uniformly white, assess for some arbitrary threshold
            if (uniques[0] == 255) or (uniques[0] == 0):
                thresMets = pd.DataFrame({'Reference Mask':ref.name,
                                           'System Output Mask':sys.name,
                                           'Threshold':127.,
                                           'NMM':[-1.],
                                           'MCC':[0.],
                                           'WL1':[1.],
                                           'TP':[0],
                                           'TN':[0],
                                           'FP':[0],
                                           'FN':[0],
                                           'N':[0]})
                mets = self.getMetrics(popt=popt)
                for m in ['NMM','MCC','WL1']:
                    thresMets.set_value(0,m,mets[m])
                for m in ['TP','TN','FP','FN','N']:
                    thresMets.set_value(0,m,self.conf[m])
            else:
                #assess for both cases where we treat as all black or all white
                thresMets = pd.DataFrame({'Reference Mask':ref.name,
                                           'System Output Mask':sys.name,
                                           'Threshold':127.,
                                           'NMM':[-1.]*2,
                                           'MCC':[0.]*2,
                                           'WL1':[1.]*2,
                                           'TP':[0]*2,
                                           'TN':[0]*2,
                                           'FP':[0]*2,
                                           'FN':[0]*2,
                                           'N':[0]*2})
                rownum=0
                for th in [uniques[0]-1,uniques[0]+1]:
                    sys.binarize(th)
                    thismet = maskMetrics(ref,sys,w,th)

                    thresMets.set_value(rownum,'Threshold',th)
                    thresMets.set_value(rownum,'NMM',thismet.nmm)
                    thresMets.set_value(rownum,'MCC',thismet.mcc)
                    thresMets.set_value(rownum,'WL1',thismet.wL1)
                    thresMets.set_value(rownum,'TP',thismet.conf['TP'])
                    thresMets.set_value(rownum,'TN',thismet.conf['TN'])
                    thresMets.set_value(rownum,'FP',thismet.conf['FP'])
                    thresMets.set_value(rownum,'FN',thismet.conf['FN'])
                    thresMets.set_value(rownum,'N',thismet.conf['N'])
                    rownum=rownum+1
        else:
            thresholds=map(lambda x,y: (x+y)/2.,uniques[:-1],uniques[1:]) #list of thresholds
            thresMets = pd.DataFrame({'Reference Mask':ref.name,
                                       'System Output Mask':sys.name,
                                       'Threshold':127.,
                                       'NMM':[-1.]*len(thresholds),
                                       'MCC':[0.]*len(thresholds),
                                       'WL1':[1.]*len(thresholds),
                                       'TP':[0]*len(thresholds),
                                       'TN':[0]*len(thresholds),
                                       'FP':[0]*len(thresholds),
                                       'FN':[0]*len(thresholds),
                                       'N':[0]*len(thresholds)})
            #for all thresholds
            noScore = ref.boundaryNoScoreRegion(erodeKernSize,dilateKernSize,kern)
            w = noScore['wimg']
            rownum=0
            for th in thresholds:
                tmask = sys.get_copy()
                tmask.matrix = tmask.binarize(th)
                thismet = maskMetrics(ref,tmask,w,th)

                thresMets.set_value(rownum,'Threshold',th)
                thresMets.set_value(rownum,'NMM',thismet.nmm)
                thresMets.set_value(rownum,'MCC',thismet.mcc)
                thresMets.set_value(rownum,'WL1',thismet.wL1)
                thresMets.set_value(rownum,'TP',thismet.conf['TP'])
                thresMets.set_value(rownum,'TN',thismet.conf['TN'])
                thresMets.set_value(rownum,'FP',thismet.conf['FP'])
                thresMets.set_value(rownum,'FN',thismet.conf['FN'])
                thresMets.set_value(rownum,'N',thismet.conf['N'])
                rownum=rownum+1

            #pick max threshold for max MCC
        
        tmax = thresMets['Threshold'].iloc[thresMets['MCC'].idxmax()]
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

