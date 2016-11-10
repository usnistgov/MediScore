# *File: mycolortest.py
# *Date: 10/12/2016
# *Author: Daniel Zhou
# *Status: Complete
#
# *Description: this code produces a sample aggregate color mask. 
#
# *Disclaimer:
# This software was developed at the National Institute of Standards
# and Technology (NIST) by employees of the Federal Government in the
# course of their official duties. Pursuant to Title 17 Section 105
# of the United States Code, this software is not subject to copyright
# protection and is in the public domain. NIST assumes no responsibility
# whatsoever for use by other parties of its source code or open source
# server, and makes no guarantees, expressed or implied, about its quality,
# reliability, or any other characteristic."
#################################################################

import numpy as np,cv2
#import img
execfile('img.py')

maniImgName = '../doc/maskImgs/sample_flower.jpg'
refImgName = '../doc/maskImgs/myimg.png'
rImg = image(refImgName)
rImg.matrix = rImg.bw(230)

sysImgName = '../doc/maskImgs/sImg.png'
sImg = cv2.imread(sysImgName,0)

myns = rImg.noScoreZone(5,5,'box')

rImg.coloredMask_opt1(sysImgName,maniImgName,sImg,myns['wimg'],myns['eimg'],myns['dimg'],'.')
