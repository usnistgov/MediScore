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
import matplotlib.pyplot as plt
import os
import random
from decimal import Decimal

class TestImageMethods(ut.TestCase):
    def test_bw(self):
        #Set existing image. Safer test.
        random.seed(1998)

        #create an image and save it
        testimg = 255*np.random.uniform(0,1,(100,100))
        testimg = testimg.astype(np.uint8)
        params=list()
        params.append(cv.CV_IMWRITE_PNG_COMPRESSION)
        params.append(0)
        cv2.imwrite('testImg.png',testimg,params)

        #read it back in as an image
        mytest=refmask('testImg.png')
        #test if image is grayscale
        self.assertEqual(len(mytest.matrix.shape),2)
        mytest.matrix = mytest.bw(230) #reading the image back in doesn't automatically make it 0 or 255.

        #randomize a threshold. Should behave the same for any threshold.
        th=np.random.uniform(0,1)*255
        mytestbw=mytest.bw(th)
        mytestflat=mytest.flatten()

        #test if total number of 0 and 255 pixels total to pixel number
        totalpix=10000
        self.assertEqual(np.sum(mytestbw==255)+np.sum(mytestbw==0),totalpix)
        self.assertEqual(np.sum(mytestflat==255)+np.sum(mytestflat==0),totalpix)

    def test_noScore(self):
        #test that weight matrix is specifically equal to a pre-calculated weight matrix.
        testimg = 255*np.ones((100,100))
        testimg[61:80,31:45]=0
        testimg = testimg.astype(np.uint8)
        params=list()
        params.append(cv.CV_IMWRITE_PNG_COMPRESSION)
        params.append(0)
        cv2.imwrite('testImg.png',testimg,params)

        #read it back in as an image. Also test erode and dilate images.
        mytest=refmask('testImg.png')
        mytest.matrix = mytest.bw(230) #reading the image back in doesn't automatically make it 0 or 255.
        zones=mytest.noScoreZone(0,0,'gaussian')
        self.assertTrue(np.array_equal(zones['wimg'],np.ones((100,100))))
        self.assertTrue(np.array_equal(zones['rimg'],mytest.matrix))

        #add for nonzero kernel
        zones=mytest.noScoreZone(3,3,'box')
        dmat=255*np.ones((100,100),dtype=np.uint8)
        emat=255*np.ones((100,100),dtype=np.uint8)
        dmat[60:81,30:46]=0
        emat[62:79,32:44]=0
        wmat=1-(emat-dmat)/255.

        self.assertTrue(np.array_equal(zones['dimg'],dmat))
        self.assertTrue(np.array_equal(zones['eimg'],emat))
        self.assertTrue(np.array_equal(zones['wimg'],wmat))

    def test_metrics(self):
        random.seed(1998)
        eps=10**-10 #account for floating point errors
        print("CASE 1: Testing metrics under the same mask shape...")
        #create an image and save it
        testimg = 255*(np.random.uniform(0,2,(100,100)).round())
        testimg = testimg.astype(np.uint8)
        params=list()
        params.append(cv.CV_IMWRITE_PNG_COMPRESSION)
        params.append(0)
        cv2.imwrite('testImg2.png',testimg,params)

        #read it back in as an image
        rImg=refmask('testImg2.png')
        rImg.matrix = rImg.bw(230) #reading the image back in doesn't automatically make it 0 or 255.
        sImg=mask('testImg2.png')  #test copy is absolutely equal
        sImg.matrix = sImg.bw(230)
        
        def absoluteEquality(r,s):
            w=np.ones(r.get_dims())
            #get measures
            m1img = r.confusion_measures(s,w)
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
            m1w = np.random.uniform(0,1,r.get_dims())
            self.assertEqual(r.hamming(s),0)
            self.assertEqual(r.weightedL1(s,m1w),0)
            self.assertEqual(r.hingeL1(s,m1w,0),0)
            self.assertEqual(r.hingeL1(s,m1w,-1),0)
            self.assertTrue(abs(r.matthews(s,m1w) - 1) < eps)
            self.assertEqual(r.NimbleMaskMetric(s,w),1)

        absoluteEquality(rImg,sImg)

        ##### CASE 1b: Test for grayscale ###################################
        print("CASE 1b: Testing for grayscale cases...")
        sImg = mask('../../data/test_suite/maskScorerTests/ref1.png')
        sImg.matrix = np.copy(rImg.matrix)
        sImg.matrix[rImg.matrix==0] = 127
        w=np.ones(rImg.get_dims())
        m1bimg = rImg.confusion_measures(sImg,w)
        m1bimgtp = m1bimg['TP']
        m1bimgtn = m1bimg['TN']
        m1bimgfp = m1bimg['FP']
        m1bimgfn = m1bimg['FN']
        m1bimgN = m1bimg['N']

        #assert values
        self.assertTrue(abs(m1bimgtp-np.sum(rImg.matrix==0)*128./255) < eps)
        self.assertEqual(m1bimgtn,np.sum(rImg.matrix==255))
        self.assertEqual(m1bimgfp,0)
        self.assertTrue(abs(m1bimgfn-np.sum(rImg.matrix==0)*127./255) < eps)
        self.assertTrue(abs(rImg.matthews(sImg,w)-math.sqrt(128./255*(1-np.sum(rImg.matrix==0)/10000.)/(1-np.sum(rImg.matrix==0)*128./255/10000))) < eps)
        self.assertTrue(abs(rImg.NimbleMaskMetric(sImg,w)-1./255) < eps)

        #Case gray marks are completely opposite (on white instead).
        #All black marks are marked as white
        sImg = mask('../../data/test_suite/maskScorerTests/ref1.png')
        sImg.matrix = np.copy(rImg.matrix)
        sImg.matrix[sImg.matrix==255]=85
        sImg.matrix[sImg.matrix==0]=255
        m1bimg = rImg.confusion_measures(sImg,w)
        m1bimgtp = m1bimg['TP']
        m1bimgtn = m1bimg['TN']
        m1bimgfp = m1bimg['FP']
        m1bimgfn = m1bimg['FN']
        m1bimgN = m1bimg['N']
        self.assertEqual(m1bimgtp,0)
        self.assertTrue(abs(m1bimgtn-np.sum(rImg.matrix==255)*85./255) < eps)
        self.assertEqual(m1bimgfn,np.sum(rImg.matrix==0))
        self.assertTrue(abs(m1bimgfp-np.sum(rImg.matrix==255)*170./255) < eps)
        self.assertTrue(abs(rImg.NimbleMaskMetric(sImg,w)-max(-1,-(np.sum(rImg.matrix==0)+np.sum(rImg.matrix==255)*170./255)/np.sum(rImg.matrix==0))) < eps)

        #gray completely opposite but black pixels are perfect match
        sImg.matrix[sImg.matrix==255]=0
        m1bimg = rImg.confusion_measures(sImg,w)
        m1bimgtp = m1bimg['TP']
        m1bimgtn = m1bimg['TN']
        m1bimgfp = m1bimg['FP']
        m1bimgfn = m1bimg['FN']
        m1bimgN = m1bimg['N']
        self.assertEqual(m1bimgtp,np.sum(rImg.matrix==0))
        self.assertTrue(abs(m1bimgtn-np.sum(rImg.matrix==255)*85./255) < eps)
        self.assertEqual(m1bimgfn,0)
        self.assertTrue(abs(m1bimgfp-np.sum(rImg.matrix==255)*170./255) < eps)
        self.assertTrue(abs(rImg.NimbleMaskMetric(sImg,w)-max(-1,(np.sum(rImg.matrix==0)-np.sum(rImg.matrix==255)*170./255)/np.sum(rImg.matrix==0))) < eps)

        ####### Case 1c: Test for rotate and flip (bw) #################################
        print("CASE 1c: Testing for equality under rotation and reflection...")

        ### rotate by 90 degrees #######################
        rImgr = copy.deepcopy(rImg)
        rImgr.matrix = np.rot90(rImgr.matrix)
        sImg = mask('../../data/test_suite/maskScorerTests/ref1.png')
        sImg.matrix = np.copy(rImgr.matrix)
        absoluteEquality(rImgr,sImg)

        ### flip horizontally
        rImgf = copy.deepcopy(rImg)
        rImgf.matrix = np.fliplr(rImgf.matrix)
        sImg = mask('../../data/test_suite/maskScorerTests/ref1.png')
        sImg.matrix = np.copy(rImgf.matrix)
        absoluteEquality(rImgf,sImg)

        print("CASE 1 testing complete.\n")

        ##### CASE 2: Erode only. ###################################
        print("CASE 2: Testing for resulting mask having been only eroded and behavior of other library functions. You should expect to see ERROR messages pop up as we test the behavior of the functions, from the library functions being tested and the RUnit test package. This should happen in the test run.")
    
        #use this rImg for all subsequent cases
        rImg = 255*np.ones((100,100))
        rImg[61:81,31:46] = 0 #small 20 x 15 square
        cv2.imwrite('testImg.png',rImg,params)

        #read it back in as an image
        rImg=refmask('testImg.png')
        rImg.matrix = rImg.bw(230).astype(np.uint8) #reading the image back in doesn't automatically make it 0 or 255.
 
        #erode by small amount so that we still get 0
        wtlist = rImg.noScoreZone(3,3,'disc')
        wts = wtlist['wimg'].astype(np.uint8)
        eKern=getKern('disc',3)
        sImg = mask('../../data/test_suite/maskScorerTests/ref1.png')
        sImg.matrix=255-erode(255-rImg.matrix,eKern)

        self.assertEqual(rImg.weightedL1(sImg,wts),0)
        self.assertEqual(rImg.hingeL1(sImg,wts,0),0)

        #erode by a larger amount.

        wtlist = rImg.noScoreZone(3,3,'box')
        wts = wtlist['wimg']
        eKern=getKern('box',5)
        sImg = mask('../../data/test_suite/maskScorerTests/ref1.png')
        sImg.matrix=255-erode(255-rImg.matrix,eKern)

        #should throw exception on differently-sized image
        errImg = copy.deepcopy(sImg)
        errImg.matrix=np.zeros((10,100))
        with self.assertRaises(ValueError):
            rImg.weightedL1(errImg,wts)
        with self.assertRaises(ValueError):
            rImg.weightedL1(sImg,wts[:,51:100])
        with self.assertRaises(ValueError):
            wts2 = np.copy(wts)
            wts2 = np.reshape(wts2,(200,50))
            rImg.weightedL1(sImg,wts2)
        with self.assertRaises(ValueError):
            rImg.hingeL1(errImg,wts,0.5)

        #want both to be greater than 0
        if (rImg.weightedL1(sImg,wts) == 0):
            print("Case 2: weightedL1 is not greater than 0. Are you too forgiving?")
            exit(1)
        if (rImg.hingeL1(sImg,wts,0.005) == 0):
            print("Case 2: hingeL1 is not greater than 0. Are you too forgiving?")
            exit(1)

        print("CASE 2 testing complete.\n")

        ##### CASE 3: Dilate only. ###################################
        print("CASE 3: Testing for resulting mask having been only dilated.")
        wtlist = rImg.noScoreZone(3,3,'disc')
        wts = wtlist['wimg']
        dKern=getKern('disc',3)
        sImg = mask('../../data/test_suite/maskScorerTests/ref1.png')
        sImg.matrix=255-dilate(255-rImg.matrix,dKern)

        #dilate by small amount so that we still get 0
        self.assertEqual(rImg.weightedL1(sImg,wts),0)
        self.assertEqual(rImg.hingeL1(sImg,wts,0.5),0)

        #dilate by a larger amount.
        wtlist = rImg.noScoreZone(3,3,'box')
        wts = wtlist['wimg']
        dKern=getKern('box',5)
        sImg.matrix=255-dilate(255-rImg.matrix,dKern)

        #want both to be greater than 0
        if (rImg.weightedL1(sImg,wts) == 0):
            print("Case 3: weightedL1 is not greater than 0. Are you too forgiving?")
            exit(1)
        if (rImg.hingeL1(sImg,wts,0.005) == 0):
            print("Case 3: hingeL1 is not greater than 0. Are you too forgiving?")
            exit(1)

        #dilate by small amount so that we still get 0
        dKern=getKern('diamond',3)
        sImg.matrix=255-dilate(255-rImg.matrix,dKern)
        wtlist = rImg.noScoreZone(3,3,'diamond')
        wts = wtlist['wimg']
        self.assertEqual(rImg.weightedL1(sImg,wts),0)
        self.assertEqual(rImg.hingeL1(sImg,wts,0.5),0)

        #dilate by a larger amount
        dKern=getKern('box',5)
        sImg.matrix=255-dilate(255-rImg.matrix,dKern)
        wtlist = rImg.noScoreZone(3,3,'diamond')
        wts = wtlist['wimg']

        #want both to be greater than 0
        if (rImg.weightedL1(sImg,wts) == 0):
            print("Case 3: weightedL1 is not greater than 0. Are you too forgiving?")
            exit(1)
        if (rImg.hingeL1(sImg,wts,0.005) == 0):
            print("Case 3: hingeL1 is not greater than 0. Are you too forgiving?")
            exit(1)

        print("CASE 3 testing complete.\n")

        ##### CASE 4: Erode + dilate. ###################################
        print("CASE 4: Testing for resulting mask having been eroded and then dilated...")
        kern = getKern('gaussian',3)
        sImg.matrix=erode(255-rImg.matrix,kern)
        sImg.matrix=dilate(sImg.matrix,kern)
        sImg.matrix=255-sImg.matrix
        wtlist=rImg.noScoreZone(3,3,'gaussian')
        wts=wtlist['wimg']

        self.assertEqual(rImg.weightedL1(sImg,wts),0)
        self.assertEqual(rImg.hingeL1(sImg,wts,0.5),0)
        
        #erode and dilate by larger amount
        kern = getKern('gaussian',9)
        sImg.matrix=erode(255-rImg.matrix,kern)
        sImg.matrix=dilate(sImg.matrix,kern)
        sImg.matrix=255-sImg.matrix
        self.assertEqual(rImg.weightedL1(sImg,wts),0)

        #erode and dilate by very large amount
        kern = getKern('gaussian',21)
        sImg.matrix = erode(255-rImg.matrix,kern)
        sImg.matrix = dilate(sImg.matrix,kern)
        sImg.matrix = 255-sImg.matrix
        #want both to be greater than 0
        if (rImg.weightedL1(sImg,wts) == 0):
            print("Case 4: weightedL1 is not greater than 0. Are you too forgiving?")
            exit(1)
        if (rImg.hingeL1(sImg,wts,0.005) == 0):
            print("Case 4: hingeL1 is not greater than 0. Are you too forgiving?")
            exit(1)

        print("CASE 4 testing complete.\n")

        ##### CASE 5: Move. ###################################
        print("CASE 5: Testing for resulting mask having been moved...\n")
    
        #move close
        sImg.matrix = 255*np.ones((100,100))
        sImg.matrix[59:79,33:48] = 0 #translate a small 20 x 15 square
        wtlist=rImg.noScoreZone(5,5,'gaussian')
        wts=wtlist['wimg']
        self.assertEqual(rImg.weightedL1(sImg,wts),0)

        #move further
        sImg.matrix = 255*np.ones((100,100))
        sImg.matrix[51:71,36:51] = 0 #translate a small 20 x 15 square
        if (rImg.weightedL1(sImg,wts) == 0):
            print("Case 5: weightedL1 is not greater than 0. Are you too forgiving?")
            exit(1)
        if (rImg.hingeL1(sImg,wts,0.005) == 0):
            print("Case 5: hingeL1 is not greater than 0. Are you too forgiving?")
            exit(1)
 
        #print (wts[55:85,25:55])
        #move completely out of range
        sImg.matrix = 255*np.ones((100,100))
        sImg.matrix[31:46,61:81] = 0 #translate a small 20 x 15 square
        self.assertEqual(rImg.weightedL1(sImg,wts),476./9720)
        if (rImg.hingeL1(sImg,wts,0.005) == 0):
            print("Case 5, translate out of range: hingeL1 is not greater than 0. Are you too forgiving?")
            exit(1)

        print("CASE 5 testing complete.\n\nAll mask scorer unit tests successfully complete.")

#if __name__ == '__main__':
#    ut.main()
