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
                 journaldf,
                 joindf,
                 index,
                 mode=0,
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
        - joindf: the dataframe joining information between the reference file and journal
        - index: the index file dataframe to be saved. Used solely for dimensionality validation of reference mask.
        - mode: determines the data to access. 0 denotes the default 'manipulation' task. 1 denotes the 'splice' task
                with the probe image, 2 denotes the 'splice' task with the donor image.
        - colordict: the dictionary of colors to use for the HTML output, in BGR array format,
                     to be used as reference
        """
        self.maskData = mergedf
        self.refDir = refD
        self.sysDir = sysD
        self.rbin = refBin
        self.sbin = sysBin
        self.journalData = journaldf
        self.joinData = joindf
        self.index = index
        self.mode=mode
        self.colordict=colordict
       
    def readMasks(self,refMaskFName,sysMaskFName,verbose):
        """
        * Description: reads both the reference and system output masks and caches the binarized image
                       into the reference mask. If the journal dataframe is provided, the color and purpose
                       of select mask regions will also be added to the reference mask
        * Inputs:
        *     refMaskFName: the name of the reference mask to be parsed
        *     sysMaskFName: the name of the system output mask to be parsed
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

        binpfx = ''
        mymode = 'Probe'
        if self.mode==1:
            binpfx = 'Binary'
        if self.mode==2:
            mymode = 'Donor'
 
        #read in the reference mask
        if self.rbin >= 0:
            rImg = masks.refmask(refMaskName)
            rImg.binarize(self.rbin)
        elif self.rbin == -1:
            #only need colors if selectively scoring
            #TODO: next time take the probeFileID in as input
            myProbeID = self.maskData.query("{}{}MaskFileName=='{}' & Output{}MaskFileName=='{}'".format(binpfx,mymode,refMaskFName,mymode,sysMaskFName))[mymode + 'FileID'].iloc[0]
            if verbose: print("Fetching {}FileID {} from maskData...".format(mymode,myProbeID))

            evalcol='Evaluated'
            if self.mode == 0: #TODO: temporary measure until we get splice sorted out. Originally mode != 2
                if self.mode == 1:
                    evalcol='ProbeEvaluated'

                #get the target colors
                #joins = self.joinData.query("{}FileID=='{}'".format(mymode,myProbeID))[['JournalName','StartNodeID','EndNodeID']]
                #color_purpose = pd.merge(joins,self.journalData.query("{}=='Y'".format(evalcol)),how='left',on=['JournalName','StartNodeID','EndNodeID'])[['Color','Purpose']].drop_duplicates()
                color_purpose = self.journalData.query("{}FileID=='{}' & {}=='Y'".format(mymode,myProbeID,evalcol))[['Color','Purpose']]
                colorlist = list(color_purpose['Color'])
                purposes = list(color_purpose['Purpose'])
                purposes_unique = []
                [purposes_unique.append(p) for p in purposes if p not in purposes_unique]
                if verbose: print("Initializing reference mask {} with colors {}.".format(refMaskName,colorlist))
            else:
                colorlist = ['0 0 0']
                purposes_unique = ['add']

            rImg = masks.refmask(refMaskName,cs=colorlist,purposes=purposes_unique)
            rImg.binarize(254)

            #check to see if the color in question is even present
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

    def getMetricList(self,erodeKernSize,dilateKernSize,distractionKernSize,kern,outputRoot,verbose,html,precision=16):
        """
        * Description: gets metrics for each pair of reference and system masks
        * Inputs:
        *     erodeKernSize: length of the erosion kernel matrix
        *     dilateKernSize: length of the dilation kernel matrix
        *     distractionKernSize: length of the dilation kernel matrix for the unselected no-score zones.
                                   0 means nothing will be scored
        *     kern: kernel shape to be used
        *     outputRoot: the directory for outputs to be written
        *     verbose: permit printout from metrics
        *     html: whether or not to generate an HTML report
        *     precision: the number of digits to round the computed metrics to.
        * Output:
        *     df: a dataframe of the computed metrics
        """
        #reflist and syslist should come from the same dataframe, so length checking is not required
        mymode='Probe'
        if self.mode==2:
            mymode='Donor'

        binpfx = ''
        evalcol='Evaluated'
        if self.mode == 1:
            binpfx = 'Binary'
            evalcol='ProbeEvaluated'
        elif self.mode == 2:
            evalcol='DonorEvaluated'

        reflist = self.maskData['{}{}MaskFileName'.format(binpfx,mymode)]
        syslist = self.maskData['Output{}MaskFileName'.format(mymode)]

        manip_ids = self.maskData[mymode+'FileID']
        maniImageFName = 0
        baseImageFName = 0
        if html:
            maniImageFName = self.maskData[mymode+'FileName']
            baseImageFName = self.maskData['BaseFileName']

        nrow = len(reflist) 

        #initialize empty frame to minimum scores
#        df=pd.DataFrame({mymode+'FileID':manip_ids,
#                         'NMM':[-1.]*nrow,
#                         'MCC': 0.,
#                         'BWL1': 1.,
#                         'GWL1': 1.,
#                         'ColMaskFileName':['']*nrow,
#                         'AggMaskFileName':['']*nrow})

        df=self.maskData[[mymode+'FileID',mymode+'MaskFileName','Scored']].copy()
        df['NMM'] = [-1.]*nrow
        df['MCC'] = [0.]*nrow
        df['BWL1'] = [1.]*nrow
        df['GWL1'] = [1.]*nrow
        df['ColMaskFileName'] = ['']*nrow
        df['AggMaskFileName'] = ['']*nrow

        task = self.maskData['TaskID'].iloc[0] #should all be the same for one file
        ilog = open('index_log.txt','w+')

        for i,row in self.maskData.iterrows():
            if verbose: print("Scoring mask {} out of {}...".format(i+1,nrow))
            if syslist[i] in [None,'',np.nan]:
                self.journalData.loc[self.journalData.query("{}FileID=='{}'".format(mymode,manip_ids[i])).index,evalcol] = 'N'
                #self.journalData.set_value(i,evalcol,'N')
                df.set_value(i,'Scored','N')
                if verbose: print("Empty system mask file at index %d" % i)
                continue
            else:
                rImg,sImg = self.readMasks(reflist[i],syslist[i],verbose)
                if (rImg is 0) and (sImg is 0):
                    #no masks detected with score-able regions, so set to not scored
                    self.journalData.loc[self.journalData.query("{}FileID=='{}'".format(mymode,manip_ids[i])).index,evalcol] = 'N'
                    #self.journalData.loc[self.journalData.query("JournalName=='{}'".format(self.joinData.query("{}FileID=='{}'".format(mymode,manip_ids[i]))["JournalName"].iloc[0])).index,evalcol] = 'N'
                    #self.journalData.set_value(i,evalcol,'N')
                    df.set_value(i,'Scored','N')
                    df.set_value(i,'MCC',-2) #for reference to filter later
                    continue

                rdims = rImg.get_dims()
                idxdims = self.index.query("{}FileID=='{}'".format(mymode,manip_ids[i])).iloc[0]
                idxW = idxdims[mymode+'Width']
                idxH = idxdims[mymode+'Height']

#                if (rdims[0] != idxH) or (rdims[1] != idxW):
#                    self.journalData.loc[self.journalData.query("{}FileID=='{}'".format(mymode,manip_ids[i])).index,evalcol] = 'N'
#                    #self.journalData.loc[self.journalData.query("JournalName=='{}'".format(self.joinData.query("{}FileID=='{}'".format(mymode,manip_ids[i]))["JournalName"].iloc[0])).index,evalcol] = 'N'
#                    #self.journalData.set_value(i,evalcol,'N')
#                    print("Reference mask {} at index {} has dimensions {} x {}. It does not match dimensions {} x {} in the index files as recorded.\
# Please notify the NIST team of the issue. Skipping for now.".format(rImg.name,i,rdims[0],rdims[1],idxH,idxW))
#                    #write mask name, mask dimensions, and image dimensions to index_log.txt
#                    ilog.write('Mask: {}, Mask Dimensions: {} x {}, Index Dimensions: {} x {}\n'.format(rImg.name,rdims[0],rdims[1],idxH,idxW))
#                    continue

                if (rImg.matrix is None) or (sImg.matrix is None):
                    #Likely this could be FP or FN. Set scores as usual.
                    if verbose: print("The index is at %d." % i)
                    continue

                #save all images in their own directories instead, rather than pool it all in one subdirectory.
                #depending on whether manipulation or splice (see taskID), make the relevant subdir_name

                if task == 'manipulation':
                    subdir_name = manip_ids[i]
                elif task == 'splice':
                    subdir_name = "{}_{}".format(self.maskData['ProbeFileID'].iloc[i],self.maskData['DonorFileID'].iloc[i])
                
                #save in subdirectory
                subOutRoot = os.path.join(outputRoot,subdir_name)
                if not os.path.isdir(subOutRoot):
                    os.system('mkdir ' + subOutRoot)
                #further subdirectories for the splice task
                if self.mode == 1:
                    subOutRoot = os.path.join(subOutRoot,'probe')
                elif self.mode == 2:
                    subOutRoot = os.path.join(subOutRoot,'donor')
                if not os.path.isdir(subOutRoot):
                    os.system('mkdir ' + subOutRoot)

                #threshold before scoring if sbin >= 0. Otherwise threshold after scoring.
                sbin_name = ''
                if self.sbin >= 0:
                    sbin_name = os.path.join(subOutRoot,sImg.name.split('/')[-1][:-4] + '-bin.png')
                    sImg.save(sbin_name,th=self.sbin)
    
                #save the image separately for html and further review. Use that in the html report
                wts,bns,sns = rImg.aggregateNoScore(erodeKernSize,dilateKernSize,distractionKernSize,kern,self.mode)

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
                #not something that needs to be calculated for every iteration of threshold; only needs to be calculated once
                gwL1 = maskMetrics.grayscaleWeightedL1(rImg,sImg,wts) 
                if self.sbin >= 0:
                    #just get scores in one run if threshold is chosen
                    mets = metricRunner.getMetrics(popt=verbose)
                    mymeas = metricRunner.conf
                    threshold = self.sbin
                    thresMets = ''
                elif self.sbin == -1:
                    #get everything through an iterative run of max threshold
                    thresMets,threshold = metricRunner.runningThresholds(rImg,sImg,bns,sns,erodeKernSize,dilateKernSize,distractionKernSize,kern=kern,popt=verbose)
                    #thresMets.to_csv(os.path.join(path_or_buf=outputRoot,'{}-thresholds.csv'.format(sImg.name)),index=False) #save to a CSV for reference
                    metrics = thresMets.query('Threshold=={}'.format(threshold)).iloc[0]
                    mets = metrics[['NMM','MCC','BWL1']].to_dict()
                    mymeas = metrics[['TP','TN','FP','FN','N']].to_dict()
                    if len(thresMets) == 1:
                        thresMets='' #to minimize redundancy

                if self.sbin == -1:
                    sbin_name = os.path.join(subOutRoot,sImg.name.split('/')[-1][:-4] + '-bin.png')
                    sImg.save(sbin_name,th=threshold)
 
                mets['GWL1'] = gwL1
                for met in ['NMM','MCC','BWL1','GWL1']:
                    df.set_value(i,met,round(mets[met],precision))
 
                if html:
                    maniImgName = os.path.join(self.refDir,maniImageFName[i])
                    colordirs = self.aggregateColorMask(rImg,sImg,bns,sns,kern,erodeKernSize,maniImgName,subOutRoot)
                    colMaskName=colordirs['mask']
                    aggImgName=colordirs['agg']
                    df.set_value(i,'ColMaskFileName',colMaskName)
                    df.set_value(i,'AggMaskFileName',aggImgName)
                    #TODO: trim the arguments here down a little? Just use threshold and thresMets, at min len 1? Remove mets and mymeas since we have threshold to index.
                    #TODO: add base image
                    self.manipReport(task,subOutRoot,df[mymode+'FileID'].loc[i],maniImageFName[i],baseImageFName[i],rImg.name,sImg.name,rbin_name,sbin_name,threshold,thresMets,bns,sns,mets,mymeas,colMaskName,aggImgName,verbose)

        ilog.close()
        return df.drop(mymode+'MaskFileName',1)

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
    
    def manipReport(self,task,outputRoot,probeFileID,maniImageFName,baseImageFName,rImg_name,sImg_name,rbin_name,sbin_name,sys_threshold,thresMets,b_weights,s_weights,metrics,confmeasures,colMaskName,aggImgName,verbose):
        """
        * Description: this function assembles the HTML report for the manipulated image and is meant to be used solely by getMetricList
        * Inputs:
        *     task: the task over which the scorer is run
        *     outputRoot: the directory to deposit the weight image and the HTML file
        *     probeFileID: the ID of the image to be considered
        *     maniImageFName: the manipulated probe file name, relative to the reference directory (self.refDir) 
        *     baseImageFName: the base file name of the probe, relative to the reference directory (self.refDir) 
        *     rImg_name: the name of the unmodified reference image used for the mask evaluation
        *     sImg_name: the name of the unmodified system output image used for the mask evaluation
        *     rbin_name: the name of the binarized reference image used for the mask evaluation
        *     sbin_name: the name of the binarized system output image used for the mask evaluation
        *     sys_threshold: the threshold used to binarize the system output mask
        *     thresMets: the table of thresholds for the image and the scores they yielded for that threshold
        *     b_weights: the weighted matrix of the no-score zones of the targeted regions
        *     s_weights: the weighted matrix of the no-score zones generated from the non-target regions
        *     metrics: the dictionary of mask scores
        *     confmeasures: truth table measures evaluated between the reference and system output masks
        *     colMaskName: the aggregate mask image of the ground truth, system output, and no-score regions
                           for the HTML report
        *     aggImgName: the above colored mask superimposed on a grayscale of the reference image
        *     verbose: whether or not to exercise printout
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
        mymode = 'Probe'
        if self.mode == 2:
            mymode = 'Donor'
        elif self.mode == 0: #TODO: temporary measure until we get splice sorted out, originally self.mode != 2
            evalcol='Evaluated'
            if self.mode == 1:
                evalcol='ProbeEvaluated'

#            journalID = self.joinData.query("{}FileID=='{}'".format(mymode,probeFileID))['JournalName'].iloc[0]
            jdata = self.journalData.query("ProbeFileID=='{}'".format(probeFileID))[['Operation','Purpose','Color',evalcol]] #("JournalName=='{}'".format(journalID))[['Operation','Purpose','Color',evalcol]]
            #jdata.loc[pd.isnull(jdata['Purpose']),'Purpose'] = '' #make NaN Purposes empty string

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

        thresString = ''
        if len(thresMets) > 1:
            thresMets = thresMets.round({'NMM':3,'MCC':3,'BWL1':3,'GWL1':3}) #TODO: eventually substitute with a plot from matplotlib based off of thresMets for MCC
            thresString = '<h4>Measures for Each Threshold</h4><br/>' + thresMets.to_html(index=False).replace("text-align: right;","text-align: center;")

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
            bPathNew = os.path.join(outputRoot,'baseFile' + baseImageFName[-4:])
            try:
                os.remove(bPathNew)
            except OSError:
                None
            if verbose: print "Creating link for base image " + baseImageFName
            os.symlink(os.path.abspath(bPath),bPathNew)
            basehtml="<img src=bPathNew alt='base image' style='width:{}px;'.format(allshapes)>"

        #TODO: condition self.mode != 2 for baseImageFName stuff

        mPathNew = os.path.join(outputRoot,mpfx+'File' + maniImageFName[-4:]) #os.path.join(outputRoot,mBase)
        rPathNew = os.path.join(outputRoot,'refMask.png') #os.path.join(outputRoot,rBase)
        sPathNew = os.path.join(outputRoot,'sysMask.png') #os.path.join(outputRoot,sBase)

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

        if verbose: print "Creating link for manipulated image " + maniImageFName
        os.symlink(os.path.abspath(mPath),mPathNew)
        if verbose: print "Creating link for refernce mask " + rImg_name
        os.symlink(os.path.abspath(rImg_name),rPathNew)
        if verbose: print "Creating link for system output mask " + sImg_name
        os.symlink(os.path.abspath(sImg_name),sPathNew)

        if verbose: print("Writing HTML...")
        htmlstr = htmlstr.substitute({'probeName': maniImageFName,
                                      'probeFname': mpfx + 'File' + maniImageFName[-4:],#mBase,
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
  				      'nmm' : round(metrics['NMM'],3),
  				      'mcc' : round(metrics['MCC'],3),
  				      'bwL1' : round(metrics['BWL1'],3),
  				      'gwL1' : round(metrics['GWL1'],3),
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
                                      'jtable':jtable,
                                      'th_table':thresString}) #add journal operations and set bg color to the html

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
        b_sImg = 1-sys.bwmat/255
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
            if (self.nmm==1) or (self.nmm==-1):
                print("NMM: %d" % self.nmm)
            else:
                print("NMM: %0.3f" % self.nmm)
            if (self.mcc==1) or (self.mcc==-1):
                print("MCC: %d" % self.mcc)
            else:
                print("MCC (Matthews correlation coeff.): %0.3f" % self.mcc)
            if (self.bwL1==1) or (self.bwL1==0):
                print("BWL1: %d" % self.bwL1)
            else:
                print("Binary Weighted L1: %0.3f" % self.bwL1)
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
        hL1=max(0,wL1-e*rArea/n)
        return hL1

    #computes metrics running over the set of thresholds for grayscale mask
    def runningThresholds(self,ref,sys,bns,sns,erodeKernSize,dilateKernSize,distractionKernSize,kern='box',popt=0):
        """
        * Description: this function computes the metrics over a set of thresholds given a grayscale mask

        * Inputs:
        *     ref: the reference mask object
        *     sys: the system output mask object
        *     bns: the boundary no-score weighted matrix
        *     sns: the selected no-score weighted matrix
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
        btotal = np.sum(bns)
        stotal = np.sum(sns)
        w = cv2.bitwise_and(bns,sns)

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

        #pick max threshold for max MCC
        tmax = thresMets['Threshold'].iloc[thresMets['MCC'].idxmax()]
        thresMets = thresMets[['Threshold','NMM','MCC','BWL1','TP','TN','FP','FN','BNS','SNS','N']]
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

