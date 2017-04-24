#!/usr/bin/python

"""
 *File: masks.py
 *Date: 12/22/2016
 *Original Author: Daniel Zhou
 *Co-Author: Yooyoung Lee
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
import rawpy
import math
import copy
import numpy as np
import pandas as pd
import os
import random
from decimal import Decimal

#returns a kernel matrix
def getKern(kernopt,size):
    """
    * Description: gets the kernel to perform erosion and/or dilation
    * Input:
    *     kernopt: the shape of the kernel to be generated. Can be one of the following:
                   'box','disc','diamond','gaussian','line'
    *     size: the length of the kernel to be generated. Must be an odd integer
    * Output:
    *     the kernel to perform erosion and/or dilation 
    """
    if (size % 2 == 0):
        raise Exception('ERROR: One of your kernel sizes is not an odd integer.')
    kern = 0
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
        print("The kernel '{}' is not recognized. Please enter a valid kernel from the following: ['box','disc','diamond','gaussian','line'].".format(kernopt))
    return kern

class mask(object):
    """
    This class is used to read in and hold the system mask and its relevant parameters.
    """
    def __init__(self,n,readopt=0):
        """
        Constructor

        Attributes:
        - n: the name of the mask file
        - readopt: the option to read in the reference mask file. Choose 1 to read in
                   as a 3-channel BGR (RGB with reverse indexing) image, 0 to read as a
                   single-channel grayscale
        """
        self.name=n
        if self.name[-4:] == '.awr':
            self.matrix=rawpy.imread(n).postprocess()
        else:
            self.matrix=cv2.imread(n,readopt)  #output own error message when catching error
        if self.matrix is None:
            masktype = 'System'
            if isinstance(self,refmask):
                masktype = 'Reference'
            print("{} mask file {} is unreadable.".format(masktype,n))
        self.bwmat = 0 #initialize bw matrix to zero. Substitute as necessary.

    def get_dims(self):
        """
        * Description: gets the dimensions of the image
        * Output:
        *     a list containing the dimensions of the image
        """
        return [self.matrix.shape[0],self.matrix.shape[1]]

    def get_copy(self):
        """
        * Description: generates a copy of this mask
        * Output:
        *     a copy of the file, ending in -2.png
        """
        mycopy = copy.deepcopy(self)
        mycopy.name = self.name[-4:] + '-2.png'
        return mycopy

    def bw(self,thres):
        """
        * Description: returns binarized matrix without affecting base matrix
        * Input:
        *     thres: the threshold to binarize the matrix
        * Output:
        *     the binarized matrix
        """
        _,mymat = cv2.threshold(self.matrix,thres,255,cv2.THRESH_BINARY)
        return mymat

    #flips mask
    def binary_flip(self):
        """
        * Description: flips a [0,255] grayscale mask to a 1 to 0 matrix, 1 corresponding to black and
                       0 corresponding to white. Used for weights and aggregate color mask computation
        * Output:
        *     the flipped matrix
        """
        return 1 - self.matrix/255.

    def dimcheck(self,img):
        """
        * Description: dimensional validation against masks. Image assumed to have been entered as a matrix for flexibility
        * Input:
        *     img: the image to validate dimensionality against
        * Output:
        *     True or False, depending on whether the dimensions are equivalent
        """
        return self.get_dims() == img.shape

    @staticmethod
    def getColors(img,popt=0):
        """
        * Description: outputs the (non-white) colors of the image and their total
        * Input:
        *     img: the image to determine the distinct (non-white) colors
        *     popt: whether or not to print the colors and number of colors
        * Output:
        *     the distinct colors in the image and the total number of distinct colors
        """

        if len(img.shape) == 3:
            colors = list(set(tuple(p) for m2d in img for p in m2d))
            if (255,255,255) in colors:
                colors.remove((255,255,255))
        elif len(img.shape) == 2:
            colors = list(set(p for m2d in img for p in m2d))
            if 255 in colors:
                colors.remove(255) 

        if popt==1:
            for c in colors:
                print(c)
            print("Total: " + len(colors))

        return colors


    def overlay(self,imgName,alpha=0.7):
        """
        * Description: overlays the mask on top of the grayscale image. If you want a color mask,
                       reread the image in as a color mask
        * Input:
        *     imgName: the image on which to overlay this image
        *     alpha: the alpha value to overlay the image in the interval [0,1] (default 0.7)
        * Output:
        *     overmat: the overlayed image
        """
        mymat = np.copy(self.matrix)
        gImg = cv2.imread(imgName,0)
        gImg = np.dstack(gImg,gImg,gImg)
        if len(self.matrix.shape)==2:
            mymat = np.dstack([mymat,mymat,mymat])

        overmat = cv2.addWeighted(mymat,alpha,gImg,1-alpha,0)
        return overmat

    def intensityBinarize3Channel(self,RThresh,GThresh,BThresh,v,g):
        """
        * Description: generalized binarization based on intensity of all three channels
        * Input:
        *     RThresh: the threshold for the red channel
        *     GThresh: the threshold for the green channel
        *     BThresh: the threshold for the blue channel
        *     v: the value taken by pixels with intensity less than or equal to any of the
                 three channels
        *     g: the value taken by pixels with intensity greater than all three channels
        * Output:
        *     bimg: single-channel binarized image
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
        * Description: sets all non-white pixels to single-channel black
        * Output:
        *     bimg: single-channel binarized image
        """
        bimg = self.intensityBinarize3Channel(254,254,254,0,255)
        return bimg

    #general binarize
    def binarize(self,threshold):
        """
        * Description: sets all non-white pixels of the image to single-channel black.
                       The resulting image is also cached in a variable for the object
        * Input:
        *     threshold: the threshold at which to binarize the image
        * Output:
        *     self.bwmat: single-channel binarized image
        """
        if len(self.matrix.shape)==2:
            self.bwmat = self.bw(threshold)
        else:
            self.bwmat = self.intensityBinarize3Channel(threshold,threshold,threshold,0,255)
        return self.bwmat

    def pixelNoScore(self,pixelvalue):
        """
        * Description: this function produces a custom no-score region based on the pixel value in the function 
        * Inputs:
        *     pixelvalue: pixel value to treat as custom no-score region
        * Outputs:
        *     pns: pixel-based no-score region
        """
        dims = self.get_dims()
        pns = np.ones((dims[0],dims[1])).astype(np.uint8)
        if pixelvalue == -1:
            return pns
        pns[self.matrix==pixelvalue] = 0
        return pns

    #save mask to file
    def save(self,fname,compression=0,th=-1):
        """
        * Description: this function saves the image under a specified file name
        * Inputs:
        *     fname: the name of the file to save the image under. Must be saved as a PNG
        *     compression: the PNG compression level. 0 for no compression, 9 for maximum compression
        *     th: the threshold to binarize the image. Selecting a threshold less than 0 will not
                  binarize the image
        * Output:
        *     0 if saved with no incident, 1 otherwise
        """
        if fname[-4:] != '.png':
            print("You should only save {} as a png. Remember to add on the prefix.".format(self.name))
            return 1
        params=list()
        params.append(cv.CV_IMWRITE_PNG_COMPRESSION)
        params.append(compression)
        outmat = self.matrix
        if th >= 0:
            self.binarize(th)
            outmat = self.bwmat
        cv2.imwrite(fname,outmat,params)
        return 0

class refmask(mask):
    """
    This class is used to read in and hold the reference mask and its relevant parameters.
    It inherits from the (system output) mask above.
    """
    def __init__(self,n,readopt=1,cs=[],purposes='all'):
        """
        Constructor

        Attributes:
        - n: the name of the reference mask file
        - readopt: the option to read in the reference mask file. Choose 1 to read in
                   as a 3-channel BGR (RGB with reverse indexing) image, 0 to read as a
                   single-channel grayscale
        - cs: the list of color strings in RGB format selected based on the target
              manipulations to be evaluated (e.g. ['255 0 0','0 255 0'])
        - purposes: the target manipulations, a string if single, a list if multiple, to be evaluated.
               'all' means all non-white regions will be evaluated
        """
        super(refmask,self).__init__(n,readopt)
        #store colors and corresponding type
        self.colors = [[int(p) for p in c.split(' ')[::-1]] for c in cs]
        self.purposes = purposes

    def aggregateNoScore(self,erodeKernSize,dilateKernSize,distractionKernSize,kern,mode):
        """
        * Description: this function calculates and generates the aggregate no score zone of the mask
                       by performing a bitwise and (&) on the elements of the boundaryNoScoreRegion and the
                       unselectedNoScoreRegion functions
        * Inputs:
        *     erodeKernSize: total length of the erosion kernel matrix
        *     dilateKernSize: total length of the dilation kernel matrix
        *     distractionKernSize: total length of the dilation kernel matrix for the distraction no-score zone
        *     kern: kernel shape to be used
        *     mode: determines the task used. 0 denotes the evaluation of the  manipulation task, 1 denotes the evaluation
                    of the probe image in the splice task, 2 denotes the evaluation of the donor image in the splice task.
        """
        baseNoScore = self.boundaryNoScoreRegion(erodeKernSize,dilateKernSize,kern)['wimg']
        wimg = baseNoScore
        distractionNoScore = np.ones(self.get_dims(),dtype=np.uint8)
        if (distractionKernSize > 0) and (self.purposes is not 'all') and (mode!=1): #case 1 treat other no-scores as white regions
            distractionNoScore = self.unselectedNoScoreRegion(distractionKernSize,kern)
            wimg = cv2.bitwise_and(baseNoScore,distractionNoScore)

        return wimg,baseNoScore,distractionNoScore

    def boundaryNoScoreRegion(self,erodeKernSize,dilateKernSize,kern):
        """
        * Description: this function calculates and generates the no score zone of the mask,
                             as well as the eroded and dilated masks for additional reference
        * Inputs:
        *     erodeKernSize: total length of the erosion kernel matrix
        *     dilateKernSize: total length of the dilation kernel matrix
        *     kern: kernel shape to be used 
        """
        if (erodeKernSize==0) and (dilateKernSize==0):
            dims = self.get_dims()
            weight = np.ones(dims,dtype=np.uint8)
            return {'rimg':self.matrix,'wimg':weight}

        mymat = 0
        if (len(self.matrix.shape) == 3) and (self.purposes is not 'all'): 
            selfmat = self.matrix
            mymat = np.ones(self.get_dims(),dtype=np.uint8)
            for c in self.colors:
                mymat = mymat & (~((selfmat[:,:,0]==c[0]) & (selfmat[:,:,1]==c[1]) & (selfmat[:,:,2]==c[2]))).astype(np.uint8)
            mymat = 255*mymat
            self.bwmat = mymat
        else:
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

    def unselectedNoScoreRegion(self,dilateKernSize,kern):
        """
        * Description: this function calculates the no score zone of the unselected mask regions.
                            The resulting mask is meant to be paired with the no score zone function above to form
                            a comprehensive no-score region
                            
                            It parses the BGR color codes to determine the color corresponding to the
                            type of manipulation
        * Inputs:
        *     dilateKernSize: total length of the dilation kernel matrix for the unselected no-score region
        *     kern: kernel shape to be used
        * Output:
        *     weights: the weighted matrix computed from the distraction zones
        """

        mymat = self.matrix
        dims = self.get_dims()
        kern = kern.lower()
        dKern=getKern(kern,dilateKernSize)
        
        #take all distinct 3-channel colors in mymat, subtract the colors that are reported, and then iterate
        notcolors = mask.getColors(mymat)

        for c in self.colors:
            if tuple(c) in notcolors:
                notcolors.remove(tuple(c))
            #skip the colors that aren't present, in case they haven't made it to the mask in question
        if len(notcolors)==0:
            weights = np.ones(dims,dtype=np.uint8)
            return weights

        mybin = np.ones((dims[0],dims[1])).astype(np.uint8)
        for c in notcolors:
            #set equal to cs
            tbin = ~((mymat[:,:,0]==c[0]) & (mymat[:,:,1]==c[1]) & (mymat[:,:,2]==c[2]))
            tbin = tbin.astype(np.uint8)
            mybin = mybin & tbin

        #note: erodes relative to 0. We have to invert it twice to get the actual effects we want relative to 255.
        dImg=1-cv2.dilate(1-mybin,dKern,iterations=1)
        weights=dImg.astype(np.uint8)

        return weights

