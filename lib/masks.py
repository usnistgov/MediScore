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
import glymur
from scipy import misc
from decimal import Decimal

debug_mode=False
printq = lambda *a:None
if debug_mode:
    def printq(string):
        print(string)

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
    if (size == 0):
        printq("Kernel size 0 chosen. Returning 0.")
        return 0
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

def count_bits(n):
    """
    * Description: counts the number of bits in an unsigned integer.
    * Input:
    *     n: the unsigned integer with bits to be counted
    * Output:
    *     c: the number of bits in the integer
    """
    if n == 0:
        return 0
    c = 0
    top_bit = 1
    if n > 0:
        top_bit = int(math.floor(math.log(n,2))) + 1
    else: # n < 0
        print("The integer {} put into count_bits is not unsigned.".format(n))
        return -1
    all_bits = range(0,top_bit)
    all_bits = [ 1 << b for b in all_bits ]
    for b in all_bits:
        if b & n != 0:
            c = c + 1
    return c

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
        ext = self.name.split('.')[-1].lower()
        if (ext == 'arw') or (ext == 'nef'):
            self.matrix=rawpy.imread(n).postprocess()
            #rgb2gray this if readopt==0
            if readopt == 0:
                self.matrix=cv2.cvtColor(self.matrix,cv2.COLOR_BGR2GRAY)
        elif ext == 'bmp':
            bmpmode='L'
            if readopt==1:
                bmpmode='RGB'
            self.matrix=misc.imread(n,mode=bmpmode)
        elif ext == 'jp2':
            jp2 = glymur.Jp2k(n)
            self.matrix = jp2[:]
        else:
            self.matrix=cv2.imread(n,readopt)  #output own error message when catching error
        if self.matrix is None:
            masktype = 'System'
            if isinstance(self,refmask) or isinstance(self,refmask_color):
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
#            colors = list(set(tuple(p) for m2d in img for p in m2d))
            img1L = img[:,:,0]*65536+img[:,:,1]*256+img[:,:,2]
            colors = np.unique(img1L)
            colors = [(c//65536,(c % 65536)//256,c % 256) for c in colors]
            if (255,255,255) in colors:
                colors.remove((255,255,255))
        elif len(img.shape) == 2:
            colors = np.unique(img)
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
        #remodify to account for multiple layers (bit positions past 8)
        if self.name.split('.')[-1] == 'jp2':
#            intmat = np.zeros(self.matrix.shape)
#            intmat[self.matrix == 0] = 255
            if len(self.matrix.shape)==3:
                #collapse all matrices into a state where anything > 0 is 1.
                intmat = np.zeros(self.get_dims(),dtype=np.uint8)
                nchannels = self.matrix.shape[2]
                for j in range(nchannels):
                    intmat = intmat | (self.matrix[:,:,j] > 0)
                _,intmat = cv2.threshold(intmat,0,255,cv2.THRESH_BINARY_INV)
            else:
                _,intmat = cv2.threshold(self.matrix,0,255,cv2.THRESH_BINARY_INV)
            self.bwmat = intmat
            return self.bwmat

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
#    def __init__(self,n,readopt=1,cs=[],purposes='all'):
    def __init__(self,n,readopt=1,jData=0,mode=0):
        """
        Constructor

        Attributes:
        - n: the name of the reference mask file
        - readopt: the option to read in the reference mask file. Choose 1 to read in
                   as a 3-channel BGR (RGB with reverse indexing) image, 0 to read as a
                   single-channel grayscale
        - jData: the journal data needed for the reference mask to select manipulated regions
        - mode: evaluation mode. 0 for manipulation, 1 for splice on probe, 2 for splice on donor.
        """
        super(refmask,self).__init__(n,readopt)
        #rework the init and other functions to support bit masking
        #default to all regions if it is 0
        self.bitlist=0
        self.is_multi_layer = len(self.matrix.shape) == 3

        if jData is not 0:
#            self.colors = [[0,0,0]]
#            self.purposes = ['add']
#        else:
#            if (mode == 2) or (mode == 1):
#                #TODO: temporary measure until splice task supports individual tasks again
#                self.colors = [[0,0,0]]
#                self.purposes = ['add']
#            else:
            evalcol='Evaluated'
            if mode==1:
                evalcol='ProbeEvaluated'
            elif mode==2:
                evalcol='DonorEvaluated'

            #sort in sequence first
            self.journalData = jData.sort_values("Sequence",ascending=False)
            desired_rows = self.journalData.query("{}=='Y'".format(evalcol))

            all_bitlist = self.journalData['BitPlane'].unique().tolist()
            if '' in all_bitlist:
                all_bitlist.remove('')
            if 'None' in all_bitlist:
                all_bitlist.remove('None')
            
            #filter out bits that aren't present for that probe.
            bitmask = 0
            for b in all_bitlist:
                bitmask = bitmask + (1 << (int(b) - 1))
            printq(bitmask)
            if self.is_multi_layer:
                n_layers = self.matrix.shape[2]
                for l in range(n_layers):
                    self.matrix[:,:,l] = self.matrix[:,:,l] & (bitmask >> 8*l)
            else:
                self.matrix = self.matrix & bitmask

            bitlist = desired_rows['BitPlane'].unique().tolist()
            if '' in bitlist:
                bitlist.remove('')
            if 'None' in bitlist:
                bitlist.remove('None')
            self.bitlist = [ 1 << (int(b)-1) for b in bitlist ]

#            purposes = list(jData['Purpose'])
#            purposes_unique = []
#            [purposes_unique.append(p) for p in purposes if p not in purposes_unique]
#            self.colors = [[int(p) for p in c.split(' ')[::-1]] for c in colorlist]
#            self.purposes = purposes
        else:
            self.journalData = 0

    def getUniqueValues(self):
        """
        * Description: return unique values in the matrix
        """
        if self.is_multi_layer:
            singlematrix = np.zeros(self.get_dims(),dtype=np.uint8)
            const_factor = 1
            for i in range(self.matrix.shape[2]):
                const_factor = 1 << 8*i
                singlematrix = singlematrix + const_factor*self.matrix[:,:,i]
            unique_px = np.unique(singlematrix)
        else:
            unique_px = np.unique(self.matrix)
        return unique_px

    def regionIsPresent(self):
        """
        * Description: return True if a scoreable region is present. Does not account for no-score zones yet. False if otherwise.
        """
        presence = 0
        rmat = self.matrix
        bits = self.bitlist
        for b in bits:
            if len(rmat.shape) == 3:
                #modify to account for multi-channel
                bpos = int(math.log(b,2))
                layer = bpos//8
                presence = presence + np.sum(rmat[:,:,layer] & (b >> 8*layer))
            else:
                presence = presence + np.sum(rmat & b)
            if presence > 0:
                return True
        return False

    def getColor(self,b):
        if count_bits(b) != 1:
            raise ValueError("The input pixel number should be a power of 2.")

        cID = int(math.log(b,2)) + 1
        color = self.journalData.query("BitPlane=='{}'".format(cID)).iloc[0]['Color']
        color = [int(p) for p in color.split(' ')]
        return color

    def getAnimatedMask(self,option='all'):
        """
        * Description: return the array of color masks that can then be saved as an animated png.
                       This function will return all the colors, not just the ones scored.
        * Inputs:
        *     option: select 'all' to return all the regions in the animated mask, 'partial' to return
                      only the regions scored
        """
        dims = self.get_dims()
        base_mask = 255*np.ones((dims[0],dims[1],3),dtype=np.uint8)
        is_multi_layer = len(self.matrix.shape) == 3

        #get unique pixel values
        unique_px = self.getUniqueValues()

        if self.journalData is 0:
            top_bit = int(math.floor(math.log(max(unique_px),2))) + 1
            all_bits = [1 << b for b in range(top_bit)]
            mybitlist = all_bits
        #get all the non-intersecting regions first
        else:
            if option=='all':
                mybitlist = self.journalData['BitPlane'].unique().tolist()
                if '' in mybitlist:
                    mybitlist.remove('')
                if 'None' in mybitlist:
                    mybitlist.remove('None')
                mybitlist = [1 << (int(b) - 1) for b in mybitlist]
    
            elif option=='partial':
                mybitlist = self.bitlist
            printq((option,mybitlist))

        for p in unique_px:
            if (count_bits(p) == 1) and (p in mybitlist):
                if is_multi_layer:
                    #parse it into the appropriate layer
                    layer = int(math.log(p,2)//8)
                    pixels = self.matrix[:,:,layer] == (p >> layer*8)
                else:
                    pixels = self.matrix == p

                base_mask[pixels] = self.getColor(p)

        seq = []
        
        for b in mybitlist:
            pixel_catch = 0
            pixel_list = []
            tempmask = np.copy(base_mask) #NOTE: additive masks, so continuously overlay. But need a new copy each time

            pixel_list = [p for p in unique_px if p & b != 0]
#            for p in unique_px:
            for p in pixel_list:
                if count_bits(p) == 1:
                    continue
#                if b & p != 0:
#                    pixel_catch = pixel_catch + 1
#                    pixel_list.append(p)
#                    tempmask[self.matrix == p] = self.getColor(b)
                pixel_catch = pixel_catch + 1
                if is_multi_layer:
                    #parse it into the appropriate layer
                    layer = int(math.log(p,2)//8)
                    pixels = self.matrix[:,:,layer] == (p >> layer*8)
                else:
                    pixels = self.matrix == p

                base_mask[pixels] = self.getColor(b)
                tempmask[pixels] = self.getColor(b)

            if pixel_catch > 0:
#                print("The mask generator caught pixels at {}.".format(pixel_list))
                 seq.append(tempmask)
        if len(seq) == 0:
            seq.append(base_mask)

        return seq

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
#        wimg = baseNoScore
#        distractionNoScore = np.ones(self.get_dims(),dtype=np.uint8)
#        if (distractionKernSize > 0) and (self.purposes is not 'all') and (mode!=1): #case 1 treat other no-scores as white regions
        distractionNoScore = self.unselectedNoScoreRegion(erodeKernSize,distractionKernSize,kern)
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
#        if (len(self.matrix.shape) == 3) and (self.purposes is not 'all'):
        selfmat = self.matrix
        is_multi_layer = len(self.matrix.shape) == 3

        if self.bitlist is not 0:
            #thorough computation for individual bits 
            mymat = np.zeros(self.get_dims(),dtype=np.uint8)
            for b in self.bitlist:
#                mymat = mymat & (~((selfmat[:,:,0]==c[0]) & (selfmat[:,:,1]==c[1]) & (selfmat[:,:,2]==c[2]))).astype(np.uint8)
                if is_multi_layer:
                    layer = int(math.log(b,2)//8)
                    pixels = (self.matrix[:,:,layer] & (b >> layer*8)) > 0
                else:
                    pixels = (selfmat & b) > 0

                mymat = mymat | pixels
            mymat = 255*(1-mymat).astype(np.uint8)
        else:
            if is_multi_layer:
                for l in range(self.matrix.shape[2]):
                    mymat = mymat | (self.matrix[:,:,l] > 0)
                mymat = 255*(1-(mymat > 0))
            else:
                mymat = 255*(1-(selfmat > 0))
        self.bwmat = mymat

        #note: erodes relative to 0. We have to invert it twice to get the actual effects we want relative to 255.
        kern = kern.lower()
        if erodeKernSize > 0:
            eKern=getKern(kern,erodeKernSize)
            eImg=255-cv2.erode(255-mymat,eKern,iterations=1)
        else:
            eImg=mymat

        if dilateKernSize > 0:
            dKern=getKern(kern,dilateKernSize)
            dImg=255-cv2.dilate(255-mymat,dKern,iterations=1)
        else:
            dImg=mymat

        weight=(eImg-dImg)/255 #note: eImg - dImg because black is treated as 0.
        wFlip=1-weight

        return {'wimg':wFlip,'eimg':eImg,'dimg':dImg}

    def unselectedNoScoreRegion(self,erodeKernSize,dilateKernSize,kern):
        """
        * Description: this function calculates the no score zone of the unselected mask regions.
                            The resulting mask is meant to be paired with the no score zone function above to form
                            a comprehensive no-score region
                            
                            It parses the BGR color codes to determine the color corresponding to the
                            type of manipulation
        * Inputs:
        *     erodeKernSize: total length of the erosion kernel matrix for establishing the scored region
        *     dilateKernSize: total length of the dilation kernel matrix for the unselected no-score region
        *     kern: kernel shape to be used
        * Output:
        *     weights: the weighted matrix computed from the distraction zones
        """

        mymat = self.matrix
        dims = self.get_dims()
        
        is_multi_layer = len(self.matrix.shape) == 3
        #take all distinct 3-channel colors in mymat, subtract the colors that are reported, and then iterate
#        notcolors = mask.getColors(mymat)
#        if is_multi_layer:
#            singlematrix = np.zeros(dims,dtype=np.uint8)
#            const_factor = 1
#            for i in range(self.matrix.shape[2]):
#                const_factor = 1 << 8*i
#                singlematrix = singlematrix + const_factor*mymat[:,:,i]
#            notcolors = np.unique(singlematrix)
#        else:
#            notcolors = np.unique(mymat)

        notcolors = self.getUniqueValues()
        printq("Colors to consider: {}".format(notcolors))
        top_px = max(notcolors)

        printq(self.journalData)
        if (np.array_equal(mymat,np.zeros(dims))) or (self.bitlist is 0):
            weights = np.ones(dims,dtype=np.uint8)
            return weights

        #decompose into individual bit channels.
        #but this won't represent all colors, so max with the max bit in self.bitlist
        top_bit = max([int(math.floor(math.log(top_px,2))),int(math.floor(math.log(max(self.bitlist),2)))]) + 1
#        top_bit = int(math.floor(math.log(max(self.bitlist),2))) + 1
        notcolors = range(0,top_bit)
        notcolors = [ 1 << b for b in notcolors ]
        printq("Colors to consider: {}".format(notcolors))

#        for c in self.colors:
        scored = np.zeros((dims[0],dims[1]),dtype=np.uint8)

        for c in self.bitlist:
#            if tuple(c) in notcolors:
            if is_multi_layer:
                layer = int(math.log(c,2)//8)
                printq("Multi-layered. Accessing layer {} to get bit {}".format(layer,c))
                pixels = (mymat[:,:,layer] & (c >> layer*8)) > 0
            else:
                printq("Single-layered. Accessing bit {}".format(c))
                pixels = (mymat & c) > 0

            scored = scored | pixels
            if c in notcolors:
                notcolors.remove(c)
            #skip the colors that aren't present, in case they haven't made it to the mask in question

        printq("Excluded colors: {}".format(notcolors))
        if len(notcolors)==0:
            weights = np.ones(dims,dtype=np.uint8)
            return weights

        #edit for bit masks
        mybin = np.zeros((dims[0],dims[1]),dtype=np.uint8)
        for c in notcolors:
            #set equal to cs
#            tbin = ~((mymat[:,:,0]==c[0]) & (mymat[:,:,1]==c[1]) & (mymat[:,:,2]==c[2]))
            if is_multi_layer:
                layer = int(math.log(c,2)//8)
                pixels = (mymat[:,:,layer] & (c >> layer*8)) > 0
            else:
                pixels = (mymat & c) > 0
            mybin = mybin | pixels

        mybin = mybin.astype(np.uint8)
        #note: erodes relative to 0. We have to invert it twice to get the actual effects we want relative to 255.
        #eroded region must be set to 1 and must not be overrideen by the unselected NSR
        kern = kern.lower()
        if erodeKernSize > 0:
            eKern=getKern(kern,erodeKernSize)
            eImg = cv2.erode(scored,eKern,iterations=1)
        else:
            eImg = scored

        if dilateKernSize > 0:
            dKern=getKern(kern,dilateKernSize)
            dImg = 1 - cv2.dilate(mybin,dKern,iterations=1)
        else:
            dImg = scored

        dImg = dImg | eImg
        weights=dImg.astype(np.uint8)

        return weights

class refmask_color(mask):
    """
    This class is used to read in and hold the reference mask and its relevant parameters.
    It inherits from the (system output) mask above.
    """
#    def __init__(self,n,readopt=1,cs=[],purposes='all'):
    def __init__(self,n,readopt=1,jData=0,mode=0):
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
        super(refmask_color,self).__init__(n,readopt)
        #store colors and corresponding type
        if jData is 0:
            self.colors = [[0,0,0]]
            self.purposes = ['add']
        else:
            if (mode == 2) or (mode == 1):
                #TODO: temporary measure until splice task supports individual tasks again
                self.colors = [[0,0,0]]
                self.purposes = ['add']
            else:
                filteredJournal = jData.query("Evaluated=='Y'")
                colorlist = filteredJournal['Color'].tolist()
                colorlist = list(filter(lambda a: a != '',colorlist))
                purposes = filteredJournal['Purpose'].tolist()
                purposes_unique = []
                [purposes_unique.append(p) for p in purposes if p not in purposes_unique]
                self.colors = [[int(p) for p in c.split(' ')[::-1]] for c in colorlist]
                self.purposes = purposes

    def regionIsPresent(self):
        #return True if a scoreable region is present. False if otherwise.
        presence = 0
        rmat = self.matrix
        for c in self.colors:
            presence = presence + np.sum((rmat[:,:,0] == c[0]) & (rmat[:,:,1] == c[1]) & (rmat[:,:,2] == c[2]))
            if presence > 0:
                return True
        return False

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
            distractionNoScore = self.unselectedNoScoreRegion(erodeKernSize,distractionKernSize,kern)
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

        #note: erodes relative to 0. We have to invert it twice to get the actual effects we want relative to 255.
        kern = kern.lower()
        if erodeKernSize > 0:
            eKern=getKern(kern,erodeKernSize)
            eImg=255-cv2.erode(255-mymat,eKern,iterations=1)
        else:
            eImg = mymat
        if dilateKernSize > 0:
            dKern=getKern(kern,dilateKernSize)
            dImg=255-cv2.dilate(255-mymat,dKern,iterations=1)
        else:
            dImg = mymat

        weight=(eImg-dImg)/255 #note: eImg - dImg because black is treated as 0.
        wFlip=1-weight

        return {'wimg':wFlip,'eimg':eImg,'dimg':dImg}

    def unselectedNoScoreRegion(self,erodeKernSize,dilateKernSize,kern):
        """
        * Description: this function calculates the no score zone of the unselected mask regions.
                            The resulting mask is meant to be paired with the no score zone function above to form
                            a comprehensive no-score region
                            
                            It parses the BGR color codes to determine the color corresponding to the
                            type of manipulation
        * Inputs:
        *     erodeKernSize: total length of the erosion kernel matrix
        *     dilateKernSize: total length of the dilation kernel matrix for the unselected no-score region
        *     kern: kernel shape to be used
        * Output:
        *     weights: the weighted matrix computed from the distraction zones
        """

        mymat = self.matrix
        dims = self.get_dims()
        scoredregion = np.zeros(dims)
        
        #take all distinct 3-channel colors in mymat, subtract the colors that are reported, and then iterate
        notcolors = mask.getColors(mymat)

        #set erode region to all 1's. Must be in erode region at all times.
        for c in self.colors:
            if tuple(c) in notcolors:
                notcolors.remove(tuple(c))
                scoredregion = (mymat[:,:,0]==c[0]) & (mymat[:,:,1]==c[1]) & (mymat[:,:,2]==c[2])
        scoredregion = scoredregion.astype(np.uint8)
            #skip the colors that aren't present, in case they haven't made it to the mask in question
        if len(notcolors)==0:
            weights = np.ones(dims,dtype=np.uint8)
            return weights

        mybin = np.ones((dims[0],dims[1]),dtype=np.uint8)
        for c in notcolors:
            #set equal to cs
            tbin = ~((mymat[:,:,0]==c[0]) & (mymat[:,:,1]==c[1]) & (mymat[:,:,2]==c[2]))
            tbin = tbin.astype(np.uint8)
            mybin = mybin & tbin

        #note: erodes relative to 0. We have to invert it twice to get the actual effects we want relative to 255.
        kern = kern.lower()
        printq(erodeKernSize)
        if erodeKernSize > 0:
            eKern=getKern(kern,erodeKernSize)
            eImg=cv2.erode(scoredregion,eKern,iterations=1)
        else:
            eImg = scoredregion
        printq(dilateKernSize)
        if dilateKernSize > 0:
            dKern=getKern(kern,dilateKernSize)
            dImg=1-cv2.dilate(1-mybin,dKern,iterations=1)
        else:
            dImg = scoredregion
        dImg=dImg | eImg
        weights=dImg.astype(np.uint8)

        return weights
