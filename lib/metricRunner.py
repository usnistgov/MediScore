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
lib_path = os.path.dirname(os.path.abspath(__file__))

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
                 mode=0,
                 speedup=False,
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
        - mode: determines the data to access. 0 denotes the default 'manipulation' task. 1 denotes the 'splice' task
                with the probe image, 2 denotes the 'splice' task with the donor image.
        - colordict: the dictionary of colors to use for the HTML output, in BGR array format,
                     to be used as reference
        - speedup: determines the mask metric computation method to be used
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
        self.speedup=speedup
       
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
            subdir_name = row[mymode+'FileID']
        elif task == 'splice':
            subdir_name = "{}_{}".format(row['ProbeFileID'],row['DonorFileID'])
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
        return subOutRoot

    def readMasks(self,refMaskFName,sysMaskFName,probeID,outRoot,verbose):
        """
        * Description: reads both the reference and system output masks and caches the binarized image
                       into the reference mask. If the journal dataframe is provided, the color and purpose
                       of select mask regions will also be added to the reference mask
        * Inputs:
        *     refMaskFName: the name of the reference mask to be parsed
        *     sysMaskFName: the name of the system output mask to be parsed
        *     probeID: the ProbeFileID corresponding to the reference mask
        *     outRoot: the directory where files are saved. Only relevant where sysMaskFName is blank
        *     verbose: permit printout from metrics
        * Outputs:
        *     rImg: the reference mask object
        *     sImg: the system output mask object
        """

        if verbose:
            print("Reference Mask: {}, System Mask: {}".format(refMaskFName,sysMaskFName))

        refMaskName = os.path.join(self.refDir,refMaskFName)
        if sysMaskFName in [None,'',np.nan]:
            sysMaskName = os.path.join(outRoot,'whitemask.png')
        else:
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
            if verbose: print("Fetching {}FileID {} from maskData...".format(mymode,probeID))

            evalcol='Evaluated'
            if self.mode == 0: #TODO: temporary measure until we get splice sorted out. Originally mode != 2
                if self.mode == 1:
                    evalcol='ProbeEvaluated'

                #get the target colors
                #joins = self.joinData.query("{}FileID=='{}'".format(mymode,myProbeID))[['JournalName','StartNodeID','EndNodeID']]
                #color_purpose = pd.merge(joins,self.journalData.query("{}=='Y'".format(evalcol)),how='left',on=['JournalName','StartNodeID','EndNodeID'])[['Color','Purpose']].drop_duplicates()
                color_purpose = self.journalData.query("{}FileID=='{}' & {}=='Y'".format(mymode,probeID,evalcol))[['Color','Purpose']]
                colorlist = list(color_purpose['Color'])
                colorlist = list(filter(lambda a: a != '',colorlist))
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

    def getMetricList(self,
                      erodeKernSize,
                      dilateKernSize,
                      distractionKernSize,
                      noScorePixel,
                      kern,
                      outputRoot,
                      verbose,
                      html,
                      precision=16):
        """
        * Description: gets metrics for each pair of reference and system masks
        * Inputs:
        *     erodeKernSize: length of the erosion kernel matrix
        *     dilateKernSize: length of the dilation kernel matrix
        *     distractionKernSize: length of the dilation kernel matrix for the unselected no-score zones.
                                   0 means nothing will be scored
        *     noScorePixel: pixel value in the mask to treat as custom no-score region.
        *     kern: kernel shape to be used
        *     outputRoot: the directory for outputs to be written
        *     verbose: permit printout from metrics
        *     html: whether or not to generate an HTML report
        *     precision: the number of digits to round the computed metrics to.
        * Output:
        *     df: a dataframe of the computed metrics
        """
        #reflist and syslist should come from the same dataframe, so length checking is not required
        if self.speedup:
            from maskMetrics import maskMetrics
        else:
            from maskMetrics_old import maskMetrics

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
            if verbose: print("Scoring {} mask {} out of {}...".format(mymode.lower(),i+1,nrow))
            subOutRoot = self.getSubOutRoot(outputRoot,task,mymode,row)
            index_row = self.index.query("{}FileID=='{}'".format(mymode,row[mymode+'FileID']))
            if len(index_row) == 0:
                if verbose: print("The probe '{}' is not in the index file. Skipping.".format(row[mymode+'FileID']))
                continue
            index_row = index_row.iloc[0]

            if syslist[i] in [None,'',np.nan]:
#                self.journalData.loc[self.journalData.query("{}FileID=='{}'".format(mymode,manip_ids[i])).index,evalcol] = 'N'
                #self.journalData.set_value(i,evalcol,'N')
#                df.set_value(i,'Scored','N')
                if verbose: print("Empty system {} mask file at index {}.".format(mymode.lower(),i))
                #save white matrix as mask in question
                whitemask = 255*np.ones((index_row[mymode+'Height'],index_row[mymode+'Width']))
                cv2.imwrite(os.path.join(subOutRoot,'whitemask.png'),whitemask)
#                continue

            rImg,sImg = self.readMasks(reflist[i],syslist[i],row[mymode+'FileID'],subOutRoot,verbose)
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

            #noScorePixel here
            pns=0
            if noScorePixel >= 0:
                pns=sImg.pixelNoScore(noScorePixel)
                rImgbin.matrix[pns==0] = self.colordict['purple'] #TODO: temporary measure until different color is picked
                wts = cv2.bitwise_and(wts,pns)

            rImgbin.save(rbin_name)
            #if wts allows for nothing to be scored, (i.e. no GT pos), print error message and record all scores as np.nan 
            if np.sum(cv2.bitwise_and(wts,rImgbin.bwmat)) == 0:
                print("Warning: No region in the mask {} is score-able.".format(rImg.name))

            #computes differently depending on choice to binarize system output mask
            mets = 0
            mymeas = 0
            threshold = 0
            metricRunner = maskMetrics(rImg,sImg,wts,self.sbin)
            #not something that needs to be calculated for every iteration of threshold; only needs to be calculated once
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
 
            mets['GWL1'] = maskMetrics.grayscaleWeightedL1(rImg,sImg,wts) 
            for met in ['NMM','MCC','BWL1','GWL1']:
                df.set_value(i,met,round(mets[met],precision))
 
            if html:
                maniImgName = os.path.join(self.refDir,maniImageFName[i])
                colordirs = self.aggregateColorMask(rImg,sImg,bns,sns,pns,kern,erodeKernSize,maniImgName,subOutRoot,self.colordict)
                colMaskName=colordirs['mask']
                aggImgName=colordirs['agg']
                df.set_value(i,'ColMaskFileName',colMaskName)
                df.set_value(i,'AggMaskFileName',aggImgName)
                #TODO: trim the arguments here down a little? Just use threshold and thresMets, at min len 1? Remove mets and mymeas since we have threshold to index.
                self.manipReport(task,subOutRoot,row[mymode+'FileID'],row[mymode+'FileName'],baseImageFName[i],rImg.name,sImg.name,rbin_name,sbin_name,threshold,thresMets,bns,sns,pns,mets,mymeas,colMaskName,aggImgName,verbose)

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
    
    def manipReport(self,task,outputRoot,probeFileID,maniImageFName,baseImageFName,rImg_name,sImg_name,rbin_name,sbin_name,sys_threshold,thresMets,b_weights,s_weights,p_weights,metrics,confmeasures,colMaskName,aggImgName,verbose):
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
        *     p_weights: the weighted matrix of the no-score zones generated from pixels of a select value in the original mask
        *     metrics: the dictionary of mask scores
        *     confmeasures: truth table measures evaluated between the reference and system output masks
        *     colMaskName: the aggregate mask image of the ground truth, system output, and no-score regions
                           for the HTML report
        *     aggImgName: the above colored mask superimposed on a grayscale of the reference image
        *     verbose: whether or not to exercise printout
        """
        if not os.path.isdir(outputRoot):
            os.system('mkdir ' + outputRoot)

        bwts = np.uint8(b_weights)
        swts = np.uint8(s_weights)
        sysBase = os.path.basename(sImg_name)[:-4]
        weightFName = sysBase + '-weights.png'
        weightpath = os.path.join(outputRoot,weightFName)
        mywts = cv2.bitwise_and(b_weights,s_weights)

        dims = bwts.shape
        colwts = 255*np.ones((dims[0],dims[1],3),dtype=np.uint8)
        #combine the colors for bwts and swts to colwts
        colwts[bwts==0] = self.colordict['yellow']
        colwts[swts==0] = self.colordict['pink']

        totalpns = 0
        if p_weights is not 0:
            colwts[p_weights==0] = self.colordict['purple']
            totalpns = np.sum(p_weights==0)

        cv2.imwrite(weightpath,colwts)

        mPath = os.path.join(self.refDir,maniImageFName)
        allshapes=min(dims[1],640) #limit on width for readability of the report

        # generate HTML files
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

#            journalID = self.joinData.query("{}FileID=='{}'".format(mymode,probeFileID))['JournalName'].iloc[0]
            jdata = self.journalData.query("ProbeFileID=='{}' & Color!=''".format(probeFileID))[['Operation','Purpose','Color',evalcol]] #("JournalName=='{}'".format(journalID))[['Operation','Purpose','Color',evalcol]]
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

        perctp="nan"
        percfp="nan"
        perctn="nan"
        percfn="nan"
        percbns="nan"
        percsns="nan"
        percpns="nan"
        if totalpx > 0:
            perctp=round(float(confmeasures['TP'])/totalpx,3)
            percfp=round(float(confmeasures['FP'])/totalpx,3)
            perctn=round(float(confmeasures['TN'])/totalpx,3)
            percfn=round(float(confmeasures['FN'])/totalpx,3)
            percbns=round(float(totalbns)/totalpx,3)
            percsns=round(float(totalsns)/totalpx,3)
            percpns=round(float(totalpns)/totalpx,3)

        thresString = ''
        if len(thresMets) > 1:
            #plot MCC
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                plt.plot(thresMets['Threshold'],thresMets['MCC'],'bo',thresMets['Threshold'],thresMets['MCC'],'k')
                plt.xlabel("Binarization threshold value")
                plt.ylabel("Matthews Correlation Coefficient (MCC)")
                thresString = os.path.join(outputRoot,'thresMets.png')
                plt.savefig(thresString,bbox_inches='tight') #save the graph
                thresString = "<img src=\"{}\" alt=\"thresholds graph\" style=\"width:{}px;\">".format('thresMets.png',allshapes)
            except:
                e = sys.exc_info()[0]
#                print("The plotter encountered error {}. Defaulting to table display for the HTML report.".format(e))
                print("Warning: The plotter encountered an issue. Defaulting to table display for the HTML report.")
                thresMets = thresMets.round({'NMM':3,'MCC':3,'BWL1':3,'GWL1':3})
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
            basehtml="<img src={} alt='base image' style='width:{}px;'>".format('baseFile' + baseImageFName[-4:],allshapes)

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
        if verbose: print "Creating link for reference mask " + rImg_name
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
                                      'pns' : totalpns,
                                      'tpcol':cols['tpcol'],
                                      'fpcol':cols['fpcol'],
                                      'tncol':cols['tncol'],
                                      'fncol':cols['fncol'],
                                      'bnscol':cols['bnscol'],
                                      'snscol':cols['snscol'],
                                      'pnscol':cols['pnscol'],
                                      'tphex':hexs[cols['tpcol']],
                                      'fphex':hexs[cols['fpcol']],
                                      'tnhex':hexs[cols['tncol']],
                                      'fnhex':hexs[cols['fncol']],
                                      'bnshex':hexs[cols['bnscol']],
                                      'snshex':hexs[cols['snscol']],
                                      'pnshex':hexs[cols['pnscol']],
                                      'perctp':perctp,
                                      'percfp':percfp,
                                      'perctn':perctn,
                                      'percfn':percfn,
                                      'percbns':percbns,
                                      'percsns':percsns,
                                      'percpns':percpns,
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
    def aggregateColorMask(self,ref,sys,bns,sns,pns,kern,erodeKernSize,maniImgName,outputMaskPath,colordict):
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
        *     colordict: the dictionary of colors to pass in. (e.g. {'red':[0 0 255],'green':[0 255 0]})
       
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
        b_pnsImg = 1-pns

        b_sImg[b_sImg != 0] = 1
        b_eImg[b_eImg != 0] = 2
        b_bnsImg[b_bnsImg != 0] = 4
        b_snsImg[b_snsImg != 0] = 8
        if b_pnsImg is not 1:
            b_pnsImg[b_pnsImg != 0] = 16
        else:
            b_pnsImg = 0

        mImg = b_sImg + b_eImg + b_bnsImg + b_snsImg + b_pnsImg

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
