#!/usr/bin/python

"""
 *File: maskTests.py
 *Date: 11/14/2016
 *Written by: Daniel Zhou
 *Status: Complete

 *Description: this code tests the mask scoring methods.

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
import unittest as ut
import pandas as pd
#import matplotlib.pyplot as plt
import os
import random
import glymur
import masks
import maskMetrics as mm
from decimal import Decimal

try:
    png_compress_const=png_compress_const
except:
    try:
        png_compress_const=cv2.IMWRITE_PNG_COMPRESSION
    except:
        png_compress_const=16

class TestImageMethods(ut.TestCase):
    def test_bw(self):
        #Set existing image. Safer test.
        random.seed(1998)

        #create an image and save it
        testimg = 255*np.random.uniform(0,1,(100,100))
        testimg = testimg.astype(np.uint8)
        params=list()
        params.append(png_compress_const)
        params.append(0)
        cv2.imwrite('testImg.png',testimg,params)

        #read it back in as an image
        mytest=masks.refmask_color('testImg.png',readopt=0)
        #test if image is grayscale
        self.assertEqual(len(mytest.matrix.shape),2)
        #mytest.matrix = mytest.bw(230) #reading the image back in doesn't automatically make it 0 or 255.

        #randomize a threshold. Should behave the same for any threshold.
        th=np.random.uniform(0,1)*255
        mytestbw=mytest.bw(th)

        #test if total number of 0 and 255 pixels total to pixel number
        totalpix=10000
        self.assertEqual(np.sum(mytestbw==255)+np.sum(mytestbw==0),totalpix)

        #test for boundary case for a threshold, less than or equal to
        testimg = 255*np.ones((100,100),dtype=np.uint8)
        testimg[45:55,45:55] = 128
        cv2.imwrite('testImg.png',testimg,params)
        mytest=masks.mask('testImg.png')
        self.assertEqual(len(mytest.matrix.shape),2)
        mytest.binarize(128)
        testimg[testimg==128] = 0
        self.assertTrue(np.array_equal(mytest.bwmat,testimg))

    def test_color(self):
        #Set existing image. Safer test.
        random.seed(1998)

        #create an image and save it as color
        testimg_bins = np.round(np.random.uniform(0,1,(100,100)))
        testimg_color = 255*np.ones((100,100,3)) #RGB
        tbins = np.copy(testimg_bins)
        tbins[:,33:] = 1
        testimg_color[tbins==0] = [0,0,255]
        tbins = np.copy(testimg_bins)
        tbins[:,:33] = 1
        tbins[:,67:] = 1
        testimg_color[tbins==0] = [255,0,0]
        tbins = np.copy(testimg_bins)
        tbins[:,:67] = 1
        testimg_color[tbins==0] = [0,255,0]
        testimg_color = testimg_color.astype(np.uint8)
        params=list()
        params.append(png_compress_const)
        params.append(0)
        cv2.imwrite('testImg_color.png',testimg_color,params)

        #read it back in as a color mask image
        mytest_color=masks.refmask_color('testImg_color.png')
        self.assertEqual(len(mytest_color.matrix.shape),3) #test if image is color
        mytest_color.matrix = mytest_color.binarize(254) #fully binarize the mask for some threshold

        mytest_bw=masks.refmask_color('testImg_color.png',readopt=0)
        self.assertEqual(len(mytest_bw.matrix.shape),2) #test if image is grayscale
        mytest_bw.matrix = mytest_bw.binarize(254)

        self.assertTrue(np.array_equal(mytest_color.matrix,mytest_bw.matrix)) #test if the two matrices are equal.

    def test_noScore(self):
        #test that weight matrix is specifically equal to a pre-calculated weight matrix.
        testimg = 255*np.ones((100,100))
        testimg[61:80,31:45]=0
        testimg = testimg.astype(np.uint8)
        params=list()
        params.append(png_compress_const)
        params.append(0)
        cv2.imwrite('testImg.png',testimg,params)

        #read it back in as an image. Also test erode and dilate images.
        mytest=masks.refmask_color('testImg.png',readopt=0)
        #mytest.matrix = mytest.bw(230) #reading the image back in doesn't automatically make it 0 or 255.
        zones=mytest.boundaryNoScoreRegion(0,0,'gaussian')
        self.assertTrue(np.array_equal(zones['wimg'],np.ones((100,100))))
        self.assertTrue(np.array_equal(zones['rimg'],mytest.matrix))

        #test for nonzero kernel
        zones=mytest.boundaryNoScoreRegion(3,3,'box')
        dmat=255*np.ones((100,100),dtype=np.uint8)
        emat=255*np.ones((100,100),dtype=np.uint8)
        dmat[60:81,30:46]=0
        emat[62:79,32:44]=0
        wmat=1-(emat-dmat)/255.

        self.assertTrue(np.array_equal(zones['dimg'],dmat))
        self.assertTrue(np.array_equal(zones['eimg'],emat))
        self.assertTrue(np.array_equal(zones['wimg'],wmat))

        #test for selective dilated color
        testimg = 255*np.ones((100,100,3))
        testimg[11:20,11:20,1:3]=0 #blue
        testimg[11:20,81:90,0] = 0 #green
        testimg[11:20,81:90,2] = 0
        testimg[81:90,81:90,0:2]=0 #red
        
        testimg = testimg.astype(np.uint8)
        params=list()
        params.append(png_compress_const)
        params.append(0)

        #generate journal data
        jData = pd.DataFrame({'Color':['255 0 0','0 255 0'],'Purpose':['remove','remove'],'Evaluated':['Y','Y']})

        cv2.imwrite('testImgC.png',testimg,params)
        mytest=masks.refmask_color('testImgC.png',jData=jData,mode=0) #red,green
        
        #generate masks
        baseNoScore = mytest.boundaryNoScoreRegion(3,5,'box')['wimg']
        distractionNoScore = mytest.unselectedNoScoreRegion(5,5,'box')

        #generate comparators
        baseNScompare = np.ones((100,100),dtype=np.uint8)
        baseNScompare[9:22,79:92] = 0
        baseNScompare[12:19,82:89] = 1
        baseNScompare[79:92,79:92] = 0
        baseNScompare[82:89,82:89] = 1

        distractionNScompare = np.ones((100,100),dtype=np.uint8)
        distractionNScompare[9:22,9:22] = 0

        self.assertTrue(np.array_equal(baseNoScore,baseNScompare))
        self.assertTrue(np.array_equal(distractionNoScore,distractionNScompare))
        aggwts,_,_ = mytest.aggregateNoScore(3,5,5,'box',0)
        self.assertTrue(np.array_equal(aggwts,baseNoScore & distractionNoScore))

    #test if the no-score zone comes out exactly as expected. Box kernel for now. 
    def test_jp2_noScore(self):
        #build sample mask
#        sample_mask = np.zeros((100,100),dtype=np.uint8)
#        sample_mask[0:50,:] = 255

#        png_params = [16,0]
#        cv2.imwrite('testsysmask.png',sample_mask,png_params)
#        sImg = masks.mask('testsysmask.png')
#        os.system('rm testsysmask.png')

        journal_df = pd.DataFrame({'JournalName':['Foo','Foo','Foo'],'Color':['255 0 0','0 255 0','0 0 255'],'Operation':['PasteSplice','LocalBlur','ContentAwareFill'],'BitPlane':[3,2,1],'Sequence':[3,2,1],'Evaluated':['Y','Y','N']})

        #blank mask, erosion and dilation should yield total weighted matrix
        blank_mask = np.zeros((100,150),dtype=np.uint8)
        #glymur write to jp2
        glymur.Jp2k('testrefmask_0.jp2',blank_mask)
        rImg = masks.refmask('testrefmask_0.jp2',jData=journal_df)
        rImg.binarize(0)
        aggwts,bns,sns = rImg.aggregateNoScore(3,5,5,'box',0)

        self.assertTrue(np.array_equal(aggwts,np.ones((100,150))))
        self.assertTrue(np.array_equal(bns,np.ones((100,150))))
        self.assertTrue(np.array_equal(sns,np.ones((100,150))))
        os.system('rm testrefmask_0.jp2')

        #single mask of 1
        mask1 = np.zeros((100,100),dtype=np.uint8)
        mask1[40:60,40:60] = 1

        glymur.Jp2k('testrefmask_1.jp2',mask1)
        rImg = masks.refmask('testrefmask_1.jp2',jData=journal_df)
        rImg.binarize(0)
        aggwts,bns,sns = rImg.aggregateNoScore(3,5,5,'box',0)

        selmask = np.ones(mask1.shape)
        selmask[38:62,38:62] = 0
        self.assertTrue(np.array_equal(aggwts,selmask))
        self.assertTrue(np.array_equal(bns,np.ones(aggwts.shape)))
        self.assertTrue(np.array_equal(sns,selmask))
        os.system('rm testrefmask_1.jp2')

        #single mask of 2
        mask2 = np.zeros((100,100),dtype=np.uint8)
        mask2[40:60,40:60] = 2
        
        glymur.Jp2k('testrefmask_2.jp2',mask2)
        rImg = masks.refmask('testrefmask_2.jp2',jData=journal_df)
        rImg.binarize(0)
        aggwts,bns,sns = rImg.aggregateNoScore(3,5,5,'box',0)

        selmask = np.ones(mask1.shape)
        selmask[38:62,38:62] = 0
        selmask[41:59,41:59] = 1
        self.assertTrue(np.array_equal(aggwts,selmask))
        self.assertTrue(np.array_equal(bns,selmask))
        self.assertTrue(np.array_equal(sns,np.ones((100,100))))
        os.system('rm testrefmask_2.jp2')

        #add case with bitplane 4 set, make sure it's equal to above
        mask3 = np.copy(mask2)
        mask3[80:100,80:100] = 8
        glymur.Jp2k('testrefmask_2a.jp2',mask2)
        rImg = masks.refmask('testrefmask_2a.jp2',jData=journal_df)
        rImg.binarize(0)
        aggwts,bns,sns = rImg.aggregateNoScore(3,5,5,'box',0)

        self.assertTrue(np.array_equal(aggwts,selmask))
        self.assertTrue(np.array_equal(bns,selmask))
        self.assertTrue(np.array_equal(sns,np.ones((100,100))))
        os.system('rm testrefmask_2a.jp2')

        #add multi-layer test case
        maskML = np.zeros((100,100,2),dtype=np.uint8)
        maskML[40:60,50:60,0] = 2
        maskML[40:60,40:50,1] = 1
        journal_df = pd.DataFrame({'JournalName':['Foo','Foo','Foo'],'Color':['255 0 0','0 255 0','0 0 255'],'Operation':['PasteSplice','LocalBlur','ContentAwareFill'],'BitPlane':[3,2,1],'Sequence':[3,2,1],'Evaluated':['Y','Y','N']})
        ML_journal_df = pd.DataFrame({'JournalName':'Foo','Color':['255 0 0','0 255 0'],'Operation':['PasteSplice','Blur'],'BitPlane':[2,9],'Sequence':[2,1],'Evaluated':['Y','Y']})

        glymur.Jp2k('testrefmask_2ML.jp2',maskML)
        rImg = masks.refmask('testrefmask_2ML.jp2',jData=ML_journal_df)
        rImg.binarize(0)
        aggwts,bns,sns = rImg.aggregateNoScore(3,5,5,'box',0)
        
        self.assertTrue(np.array_equal(aggwts,selmask))
        self.assertTrue(np.array_equal(bns,selmask))
        self.assertTrue(np.array_equal(sns,np.ones((100,100))))
        os.system('rm testrefmask_2ML.jp2')
        
        #2 contained in 1
        mask2in1 = np.zeros((100,100),dtype=np.uint8)
        mask2in1[10:90,10:70] = 1
        mask2in1[30:60,30:50] = 3
        glymur.Jp2k('testrefmask_2in1.jp2',mask2in1)
        rImg = masks.refmask('testrefmask_2in1.jp2',jData=journal_df)
        rImg.binarize(0)
        aggwts,bns,sns = rImg.aggregateNoScore(3,5,5,'box',0)

        selmask = np.ones((100,100))
        selmask[8:92,8:72] = 0
        selmask[31:59,31:49] = 1
        boundmask = np.ones((100,100))
        boundmask[28:62,28:52] = 0
        boundmask[31:59,31:49] = 1

        self.assertTrue(np.array_equal(aggwts,selmask))
        self.assertTrue(np.array_equal(bns,boundmask))
        self.assertTrue(np.array_equal(sns,selmask))
        os.system('rm testrefmask_2in1.jp2')

        #1 contained in 2
        mask1in2 = np.zeros((100,100),dtype=np.uint8)
        mask1in2[10:90,10:70] = 2
        mask1in2[30:60,30:50] = 3
        glymur.Jp2k('testrefmask_1in2.jp2',mask1in2)
        rImg = masks.refmask('testrefmask_1in2.jp2',jData=journal_df)
        rImg.binarize(0)
        aggwts,bns,sns = rImg.aggregateNoScore(3,5,5,'box',0)

        selmask = np.ones((100,100))
        selmask[8:92,8:72] = 0
        selmask[11:89,11:69] = 1
        boundmask = np.ones((100,100))
        boundmask[8:92,8:72] = 0
        boundmask[11:89,11:69] = 1

        self.assertTrue(np.array_equal(aggwts,selmask))
        self.assertTrue(np.array_equal(bns,boundmask))
        os.system('rm testrefmask_1in2.jp2')

        #2 and 1 coincide
        mask1and2 = np.zeros((100,100),dtype=np.uint8)
        mask1and2[30:50,30:50] = 3
        glymur.Jp2k('testrefmask_1and2.jp2',mask1and2)
        rImg = masks.refmask('testrefmask_1and2.jp2',jData=journal_df)
        rImg.binarize(0)
        aggwts,bns,sns = rImg.aggregateNoScore(3,5,7,'box',0)

        selmask = np.ones((100,100))
        selmask[27:53,27:53] = 0
        selmask[31:49,31:49] = 1

        boundmask = np.ones((100,100))
        boundmask[28:52,28:52] = 0
        boundmask[31:49,31:49] = 1

        dismask = np.ones((100,100))
        dismask[27:53,27:53] = 0
        dismask[31:49,31:49] = 1
        
        self.assertTrue(np.array_equal(aggwts,selmask))
        self.assertTrue(np.array_equal(bns,boundmask))
        self.assertTrue(np.array_equal(sns,dismask))
        os.system('rm testrefmask_1and2.jp2')
        
        #2 and 1 intersect
        mask1x2 = np.zeros((100,100),dtype=np.uint8)
        mask1x2[40:60,20:80] = 1
        mask1x2[20:80,40:60] = 2
        mask1x2[40:60,40:60] = 3
        glymur.Jp2k('testrefmask_1x2.jp2',mask1x2)
        rImg = masks.refmask('testrefmask_1x2.jp2',jData=journal_df)
        rImg.binarize(0)
        aggwts,bns,sns = rImg.aggregateNoScore(3,5,5,'box',0)
        
        selmask = np.ones((100,100))
        selmask[38:62,18:82] = 0
        selmask[18:82,38:62] = 0
        selmask[21:79,41:59] = 1

        boundmask = np.ones((100,100))
        boundmask[18:82,38:62] = 0
        boundmask[21:79,41:59] = 1

        dismask = np.ones((100,100))
        dismask[38:62,18:82] = 0
        dismask[21:79,41:59] = 1
        
        self.assertTrue(np.array_equal(aggwts,selmask))
        self.assertTrue(np.array_equal(bns,boundmask))
        self.assertTrue(np.array_equal(sns,dismask))
        os.system('rm testrefmask_1x2.jp2')

        #2 and 1 are adjacent
        mask1n2 = np.zeros((100,100),dtype=np.uint8)
        mask1n2[40:60,40:50] = 1
        mask1n2[40:60,50:60] = 2
        glymur.Jp2k('testrefmask_1n2.jp2',mask1n2)
        rImg = masks.refmask('testrefmask_1n2.jp2',jData=journal_df)
        rImg.binarize(0)
        aggwts,bns,sns = rImg.aggregateNoScore(3,5,7,'box',0)

        selmask = np.ones((100,100))
        selmask[37:63,37:53] = 0
        selmask[38:62,48:62] = 0
        selmask[41:59,51:59] = 1

        boundmask = np.ones((100,100))
        boundmask[38:62,48:62] = 0
        boundmask[41:59,51:59] = 1

        dismask = np.ones((100,100))
        dismask[37:63,37:53] = 0
        dismask[41:59,51:59] = 1

        self.assertTrue(np.array_equal(aggwts,selmask))
        self.assertTrue(np.array_equal(bns,boundmask))
        self.assertTrue(np.array_equal(sns,dismask))
        os.system('rm testrefmask_1n2.jp2')

        #2 and 1 are separate
        mask1_2 = np.zeros((100,100),dtype=np.uint8)
        mask1_2[40:60,20:40] = 1
        mask1_2[40:60,60:80] = 2
        glymur.Jp2k('testrefmask_1_2.jp2',mask1_2)
        rImg = masks.refmask('testrefmask_1_2.jp2',jData=journal_df)
        rImg.binarize(0)
        aggwts,bns,sns = rImg.aggregateNoScore(3,5,7,'box',0)
        
#        mask1vis = np.copy(mask1_2) #debug for images
#        mask1vis[mask1_2 == 1] = 255
#        cv2.imwrite('testrefmask_1_2.png',mask1vis,[16,0])
        selmask = np.ones((100,100))
        selmask[37:63,17:43] = 0
        selmask[38:62,58:82] = 0
        selmask[41:59,61:79] = 1

        boundmask = np.ones((100,100))
        boundmask[38:62,58:82] = 0
        boundmask[41:59,61:79] = 1

        dismask = np.ones((100,100))
        dismask[37:63,17:43] = 0

        self.assertTrue(np.array_equal(aggwts,selmask))
        self.assertTrue(np.array_equal(bns,boundmask))
        self.assertTrue(np.array_equal(sns,dismask))
        os.system('rm testrefmask_1_2.jp2')

    def test_metrics(self):
        random.seed(1998)
        eps=10**-10 #account for floating point errors
        print("CASE 1: Testing metrics under the same mask shape...")
        #create an image and save it
        testimg = 255*(np.floor(np.random.uniform(0,1.9,(100,100))).astype(np.uint8))
        params=list()
        params.append(png_compress_const)
        params.append(0)
        cv2.imwrite('testImg2.png',testimg,params)

        #read it back in as an image
        rImg=masks.refmask_color('testImg2.png',readopt=0)
#        rImg.matrix = rImg.bw(230) #reading the image back in doesn't automatically make it 0 or 255.
        sImg=masks.mask('testImg2.png')  #test copy is absolutely equal
#        sImg.matrix = sImg.bw(230)
        
        def absoluteEquality(r,s):
            w=np.ones(r.get_dims())
            #get measures
            m1measures = mm.maskMetrics(r,s,w)
            m1img = m1measures.conf
            m1imgtp = m1img['TP']
            m1imgtn = m1img['TN']
            m1imgfp = m1img['FP']
            m1imgfn = m1img['FN']
            m1imgN = m1img['N']

            #assert values
            self.assertEqual(m1imgtp,np.sum(r.matrix==0))
            self.assertEqual(m1imgtn,np.sum(r.matrix==255))
            self.assertEqual(m1imgfp,0)
            self.assertEqual(m1imgfn,0)

            #randomized weights. Should be equal no matter what weights are applied.
            #self.assertEqual(r.hamming(s),0)
            self.assertEqual(m1measures.bwL1,0)
            #self.assertEqual(r.hingeL1(s,m1w,0),0)
            #self.assertEqual(r.hingeL1(s,m1w,-1),0)
            self.assertTrue(abs(m1measures.mcc - 1) < eps)
            self.assertEqual(m1measures.nmm,1)

        absoluteEquality(rImg,sImg)

#Commented out because not using grayscale test for the time being
        ##### CASE 1b: Test for grayscale ###################################
#        print("CASE 1b: Testing for grayscale cases...")
#        sImg = mask('../../data/test_suite/maskScorerTests/ref1.png')
#        sImg.matrix = np.copy(rImg.matrix)
#        sImg.matrix[rImg.matrix==0] = 127
#        w=np.ones(rImg.get_dims())
#        m1bimg = rImg.confusion_measures(sImg,w)
#        m1bimgtp = m1bimg['TP']
#        m1bimgtn = m1bimg['TN']
#        m1bimgfp = m1bimg['FP']
#        m1bimgfn = m1bimg['FN']
#        m1bimgN = m1bimg['N']
#
#        #assert values
#        self.assertTrue(abs(m1bimgtp-np.sum(rImg.matrix==0)*128./255) < eps)
#        self.assertEqual(m1bimgtn,np.sum(rImg.matrix==255))
#        self.assertEqual(m1bimgfp,0)
#        self.assertTrue(abs(m1bimgfn-np.sum(rImg.matrix==0)*127./255) < eps)
#        self.assertTrue(abs(rImg.matthews(sImg,w)-math.sqrt(128./255*(1-np.sum(rImg.matrix==0)/10000.)/(1-np.sum(rImg.matrix==0)*128./255/10000))) < eps)
#        self.assertTrue(abs(rImg.NimbleMaskMetric(sImg,w)-1./255) < eps)
#
#        #Case gray marks are completely opposite (on white instead).
#        #All black marks are marked as white
#        sImg = mask('../../data/test_suite/maskScorerTests/ref1.png')
#        sImg.matrix = np.copy(rImg.matrix)
#        sImg.matrix[sImg.matrix==255]=85
#        sImg.matrix[sImg.matrix==0]=255
#        m1bimg = rImg.confusion_measures(sImg,w)
#        m1bimgtp = m1bimg['TP']
#        m1bimgtn = m1bimg['TN']
#        m1bimgfp = m1bimg['FP']
#        m1bimgfn = m1bimg['FN']
#        m1bimgN = m1bimg['N']
#        self.assertEqual(m1bimgtp,0)
#        self.assertTrue(abs(m1bimgtn-np.sum(rImg.matrix==255)*85./255) < eps)
#        self.assertEqual(m1bimgfn,np.sum(rImg.matrix==0))
#        self.assertTrue(abs(m1bimgfp-np.sum(rImg.matrix==255)*170./255) < eps)
#        self.assertTrue(abs(rImg.NimbleMaskMetric(sImg,w)-max(-1,-(np.sum(rImg.matrix==0)+np.sum(rImg.matrix==255)*170./255)/np.sum(rImg.matrix==0))) < eps)
#
#        #gray completely opposite but black pixels are perfect match
#        sImg.matrix[sImg.matrix==255]=0
#        m1bimg = rImg.confusion_measures(sImg,w)
#        m1bimgtp = m1bimg['TP']
#        m1bimgtn = m1bimg['TN']
#        m1bimgfp = m1bimg['FP']
#        m1bimgfn = m1bimg['FN']
#        m1bimgN = m1bimg['N']
#        self.assertEqual(m1bimgtp,np.sum(rImg.matrix==0))
#        self.assertTrue(abs(m1bimgtn-np.sum(rImg.matrix==255)*85./255) < eps)
#        self.assertEqual(m1bimgfn,0)
#        self.assertTrue(abs(m1bimgfp-np.sum(rImg.matrix==255)*170./255) < eps)
#        self.assertTrue(abs(rImg.NimbleMaskMetric(sImg,w)-max(-1,(np.sum(rImg.matrix==0)-np.sum(rImg.matrix==255)*170./255)/np.sum(rImg.matrix==0))) < eps)

        ####### Case 1c: Test for rotate and flip (bw) #################################
        print("CASE 1c: Testing for equality under rotation and reflection...")

        ### rotate by 90 degrees #######################
        rImg=masks.refmask_color('testImg2.png',readopt=0)
        rImg.matrix = np.rot90(rImg.matrix)
        sImg=masks.mask('testImg2.png')
        sImg.matrix = np.rot90(sImg.matrix)
        absoluteEquality(rImg,sImg)

        ### flip horizontally
        rImg=masks.refmask_color('testImg2.png',readopt=0)
        rImg.matrix = np.fliplr(rImg.matrix)
        sImg=masks.mask('testImg2.png')
        sImg.matrix = np.fliplr(sImg.matrix)
        absoluteEquality(rImg,sImg)

        print("CASE 1 testing complete.\n")

        ##### CASE 2: Erode only. ###################################
        print("CASE 2: Testing for resulting mask having been only eroded and behavior of other library functions.")
    
        #use this rImg for all subsequent cases
        rImg = 255*np.ones((100,100))
        rImg[61:81,31:46] = 0 #small 20 x 15 square
        cv2.imwrite('testImg.png',rImg,params)

        #read it back in as an image
        rImg=masks.refmask_color('testImg.png',readopt=0)
#        rImg.matrix = rImg.bw(230).astype(np.uint8) #reading the image back in doesn't automatically make it 0 or 255.
 
        #erode by small amount so that we still get 0
        wtlist = rImg.boundaryNoScoreRegion(3,3,'disc')
        wts = wtlist['wimg'].astype(np.uint8)
        eKern=masks.getKern('disc',3)

        sImg = masks.mask('testImg.png')
        sImg.matrix=255-cv2.erode(255-rImg.matrix,eKern,iterations=1)

        m2measures = mm.maskMetrics(rImg,sImg,wts)
        self.assertEqual(m2measures.bwL1,0)
        #self.assertEqual(m2measures.hingeL1(),0)

        #erode by a larger amount.

        wtlist = rImg.boundaryNoScoreRegion(3,3,'box')
        wts = wtlist['wimg']
        eKern=masks.getKern('box',5)
        sImg = masks.mask('testImg.png')
        sImg.matrix=255-cv2.erode(255-rImg.matrix,eKern,iterations=1)

        #should throw exception on differently-sized image
        errImg = copy.deepcopy(sImg)
        errImg.matrix=np.zeros((10,100))
        with self.assertRaises(ValueError):
            mEmeasures = mm.maskMetrics(rImg,errImg,wts)
            mEmeasures.bwL1
        with self.assertRaises(ValueError):
            mEmeasures = mm.maskMetrics(rImg,sImg,wts[:,51:100])
            mEmeasures.bwL1
        with self.assertRaises(ValueError):
            wts2 = np.copy(wts)
            wts2 = np.reshape(wts2,(200,50))
            mEmeasures = mm.maskMetrics(rImg,sImg,wts2)
            mEmeasures.bwL1

        #want both to be greater than 0
        m2measures = mm.maskMetrics(rImg,sImg,wts)
        if (m2measures.bwL1 == 0):
            print("Case 2: binary weightedL1 is not greater than 0. Are you too forgiving?")
            exit(1)
#commenting out because hinge is not being used
#        if (rImg.hingeL1(sImg,wts,0.005) == 0):
#            print("Case 2: hingeL1 is not greater than 0. Are you too forgiving?")
#            exit(1)

        print("CASE 2 testing complete.\n")

        ##### CASE 3: Dilate only. ###################################
        print("CASE 3: Testing for resulting mask having been only dilated.")
        wtlist = rImg.boundaryNoScoreRegion(3,3,'disc')
        wts = wtlist['wimg']
        dKern=masks.getKern('disc',3)
        sImg = masks.mask('testImg.png')
        sImg.matrix=255-cv2.dilate(255-rImg.matrix,dKern,iterations=1)

        m3measures = mm.maskMetrics(rImg,sImg,wts)
        #dilate by small amount so that we still get 0
        self.assertEqual(m3measures.bwL1,0)
        #self.assertEqual(m3measures.hingeL1(0.5),0)

        #dilate by a larger amount.
        wtlist = rImg.boundaryNoScoreRegion(3,3,'box')
        wts = wtlist['wimg']
        dKern=masks.getKern('box',5)
        sImg.matrix=255-cv2.dilate(255-rImg.matrix,dKern,iterations=1)

        #want both to be greater than 0
        m3measures = mm.maskMetrics(rImg,sImg,wts)
        if (m3measures.bwL1 == 0):
            print("Case 3: binary weightedL1 is not greater than 0. Are you too forgiving?")
            exit(1)
#        if (rImg.hingeL1(0.005) == 0):
#            print("Case 3: hingeL1 is not greater than 0. Are you too forgiving?")
#            exit(1)

        #dilate by small amount so that we still get 0
        dKern=masks.getKern('diamond',3)
        sImg.matrix=255-cv2.dilate(255-rImg.matrix,dKern,iterations=1)
        wtlist = rImg.boundaryNoScoreRegion(3,3,'diamond')
        wts = wtlist['wimg']
        m3measures = mm.maskMetrics(rImg,sImg,wts)

        self.assertEqual(m3measures.bwL1,0)
        #self.assertEqual(rImg.hingeL1(sImg,wts,0.5),0)

        #dilate by a larger amount
        dKern=masks.getKern('box',5)
        sImg.matrix=255-cv2.dilate(255-rImg.matrix,dKern,iterations=1)
        wtlist = rImg.boundaryNoScoreRegion(3,3,'diamond')
        wts = wtlist['wimg']

        #want both to be greater than 0
        m3measures = mm.maskMetrics(rImg,sImg,wts)
        if (m3measures.bwL1 == 0):
            print("Case 3: binary weightedL1 is not greater than 0. Are you too forgiving?")
            exit(1)
#        if (rImg.hingeL1(sImg,wts,0.005) == 0):
#            print("Case 3: hingeL1 is not greater than 0. Are you too forgiving?")
#            exit(1)

        print("CASE 3 testing complete.\n")

        ##### CASE 4: Erode + dilate. ###################################
        print("CASE 4: Testing for resulting mask having been eroded and then dilated...")
        kern = masks.getKern('gaussian',3)
        sImg.matrix=cv2.erode(255-rImg.matrix,kern,iterations=1)
        sImg.matrix=cv2.dilate(sImg.matrix,kern,iterations=1)
        sImg.matrix=255-sImg.matrix
        wtlist=rImg.boundaryNoScoreRegion(3,3,'gaussian')
        wts=wtlist['wimg']

        m4measures = mm.maskMetrics(rImg,sImg,wts)
        self.assertEqual(m4measures.bwL1,0)
        #self.assertEqual(rImg.hingeL1(sImg,wts,0.5),0)
        
        #erode and dilate by larger amount
        kern = masks.getKern('gaussian',9)
        sImg.matrix=cv2.erode(255-rImg.matrix,kern,iterations=1)
        sImg.matrix=cv2.dilate(sImg.matrix,kern,iterations=1)
        sImg.matrix=255-sImg.matrix
        m4measures = mm.maskMetrics(rImg,sImg,wts)
        self.assertEqual(m4measures.bwL1,0)

        #erode and dilate by very large amount
        kern = masks.getKern('gaussian',21)
        sImg.matrix = cv2.erode(255-rImg.matrix,kern,iterations=1)
        sImg.matrix = cv2.dilate(sImg.matrix,kern,iterations=1)
        sImg.matrix = 255-sImg.matrix
        #want both to be greater than 0
        m4measures = mm.maskMetrics(rImg,sImg,wts)
        if (m4measures.bwL1 == 0):
            print("Case 4: binary weightedL1 is not greater than 0. Are you too forgiving?")
            exit(1)
#        if (rImg.hingeL1(sImg,wts,0.005) == 0):
#            print("Case 4: hingeL1 is not greater than 0. Are you too forgiving?")
#            exit(1)

        print("CASE 4 testing complete.\n")

        ##### CASE 5: Move. ###################################
        print("CASE 5: Testing for resulting mask having been moved...\n")
    
        #move close
        sImg.matrix = 255*np.ones((100,100))
        sImg.matrix[59:79,33:48] = 0 #translate a small 20 x 15 square
        wtlist=rImg.boundaryNoScoreRegion(5,5,'gaussian')
        wts=wtlist['wimg']
        m5measures = mm.maskMetrics(rImg,sImg,wts)
        self.assertEqual(m5measures.bwL1,0)

        #move further
        sImg.matrix = 255*np.ones((100,100))
        sImg.matrix[51:71,36:51] = 0 #translate a small 20 x 15 square
        m5measures = mm.maskMetrics(rImg,sImg,wts)
        if (m5measures.bwL1 == 0):
            print("Case 5: binary weightedL1 is not greater than 0. Are you too forgiving?")
            exit(1)
#        if (rImg.hingeL1(sImg,wts,0.005) == 0):
#            print("Case 5: hingeL1 is not greater than 0. Are you too forgiving?")
#            exit(1)
 
        #print (wts[55:85,25:55])
        #move completely out of range
        sImg.matrix = 255*np.ones((100,100))
        sImg.matrix[31:46,61:81] = 0 #translate a small 20 x 15 square
        m5measures = mm.maskMetrics(rImg,sImg,wts)
        self.assertEqual(m5measures.bwL1,476./9720)
#        if (rImg.hingeL1(sImg,wts,0.005) == 0):
#            print("Case 5, translate out of range: hingeL1 is not greater than 0. Are you too forgiving?")
#            exit(1)

        print("CASE 5 testing complete.")

#if __name__ == '__main__':
#    ut.main()
