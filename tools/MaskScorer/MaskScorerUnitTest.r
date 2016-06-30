# *File: SSDMaskScorerUnitTest.r
# *Date: 5/24/2016
# *Author: Daniel Zhou
# *Status: In Progress
#
# *Description:
# This function tests the output of the library scripts of the MediScore package.
# If the library scripts operate as intended, the test will run and return 0.
# Otherwise, it exits with status 1.
#
# *Inputs: none
#
# *Outputs:
# Returns 0 if all libary scripts work as intended. Otherwise prints an error
# message and exits with status 1.
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

###Daniel: please update this test with the current version. it now includes
### both SSD and DSD in MaskScorer.r ...thanks

#setwd('~/Desktop/Medifor/MediScore/tools/MaskScorer')

lib_path <- "../../lib"
source(file.path(lib_path, "maskMetrics.r"))

unit.test <- function()
{
  suppressWarnings(suppressMessages(require(RUnit)))
  suppressWarnings(suppressMessages(require(EBImage)))
  suppressWarnings(suppressMessages(require(useful)))
  suppressWarnings(suppressMessages(require(png)))

  cat("Beginning unit test of mask scoring metrics...\n")

  samplepng = matrix(rep(1,10000),ncol=100)
  samplepng[51:80,33:67] <- 127/255
  writePNG(samplepng,target="sample.png")
  readpng <- readImage('sample.png')
  #readpng <- readPNG('sample.png')
  checkEquals(sum(t(readpng)!= samplepng),0) #CHECK: readImage reads as transpose. Correct it accordingly?
  system('rm sample.png')

  #library("RUnit", lib.loc="~/Library/R/3.2/library")
  ##### CASE 1: Same image. Should yield perfect accuracy. ################

  cat("CASE 1: Testing metrics under the exact same mask...\n")
  rImg = round(matrix(rexp(10000,rate=1),ncol=100),digits=0) #exponentially random 0s and 1s.
  rImg[rImg > 0] = 1
  sImg = rImg

  absoluteEquality <- function(r,s) {
    m1img <- measures_no_ns(r,s)
    m1imgfpr <- m1img$FPR
    m1imgtpr <- m1img$TPR
    m1imgacc <- m1img$ACC
    m1imgmcc <- m1img$MCC
    checkEquals(m1imgfpr,0)
    checkEquals(m1imgtpr,1)
    checkEquals(m1imgacc,1)
    checkEquals(m1imgmcc,1)
    m1imgloss <- hamming(r,s)
    checkEquals(m1imgloss,0)
    testwts <- generateNoScoreZone(r,0, 0,'Gaussian')
    testwts <- testwts$wimg
    dims <- dim(r)
    onewts <- matrix(rep(1,dims[1]*dims[2]),ncol=dims[2])
    checkEquals(testwts,onewts)

    m1weights <- round(rexp(length(r))) #random weights
    m1weights[m1weights > 2] = 1
    checkEquals(weightedL1(r,s,m1weights),0) #should be zero regardless
    checkEquals(hingeL1(r,s,m1weights,0),0)
    checkEquals(hingeL1(r,s,m1weights,-1),0)
  }

  absoluteEquality(rImg,sImg)

  dims <- dim(rImg)
  onewts <- matrix(rep(1,dims[1]*dims[2]),ncol=dims[2])
  m2img <- measures_ns_new(rImg,sImg,onewts)
  m2imgfp <- m2img$FP
  m2imgfn <- m2img$FN
  m2imgtp <- m2img$TP

  checkEquals(m2imgfp,0)
  checkEquals(m2imgfn,0)
  checkEquals(m2imgtp,sum(rImg==0))
  checkEquals(m2img$TN,sum(rImg==1))
  checkEquals(matthews(m2img$TP,m2img$FN,m2img$FP,m2img$N),1)

  checkEquals(NimbleMaskMetric(rImg, onewts, m2imgtp, m2imgfn, m2imgfp),1)
  cat("CASE 1 testing complete.\n\n")

  ##### CASE 1b: Test for grayscale ###################################
  cat("CASE 1b: Testing for grayscale cases...\n")
  sImg[sImg==0] = 127/255
  m1bimg <- measures_ns_new(rImg,sImg,onewts)
  checkEquals(m1bimg$TP,sum(rImg==0)*127/255)
  checkEquals(m1bimg$TN,sum(rImg==1))
  checkEquals(m1bimg$FN,sum(rImg==0)*128/255)
  checkEquals(m1bimg$FP,0)
  checkEquals(matthews(m1bimg$TP,m1bimg$FN,m1bimg$FP,m1bimg$N),sqrt(127/255*(1-sum(rImg==0)/10000)/(1-sum(rImg==0)*127/255/10000)))
  checkEquals(NimbleMaskMetric(rImg,onewts,m1bimg$TP,m1bimg$FN,m1bimg$FP),-1/255)
  #Case gray marks are completely opposite (on white instead).
  #All black marks are marked as white
  sImg = rImg
  sImg[sImg==1] = 85/255
  sImg[sImg==0] = 1
  m1bimg <- measures_ns_new(rImg,sImg,onewts)
  checkEquals(m1bimg$TP,0)
  checkEquals(m1bimg$TN,sum(rImg==1)*85/255)
  checkEquals(m1bimg$FN,sum(rImg==0))
  checkEquals(m1bimg$FP,sum(rImg==1)*170/255)
  checkEquals(NimbleMaskMetric(rImg,onewts,m1bimg$TP,m1bimg$FN,m1bimg$FP),max(-1,-(sum(rImg==0)+sum(rImg==1)*170/255)/sum(rImg==0)))

  cat("CASE 1b testing complete\n\n")

  #gray completely opposite but black pixels are perfect match
  sImg = rImg
  sImg[sImg==1] = 85/255
  m1bimg <- measures_ns_new(rImg,sImg,onewts)
  checkEquals(m1bimg$TP,sum(rImg==0))
  checkEquals(m1bimg$TN,sum(rImg==1)*85/255)
  checkEquals(m1bimg$FN,0)
  checkEquals(m1bimg$FP,sum(rImg==1)*170/255)
  checkEquals(NimbleMaskMetric(rImg,onewts,m1bimg$TP,m1bimg$FN,m1bimg$FP),max(-1,(sum(rImg==0)-sum(rImg==1)*170/255)/(sum(rImg==0))))

  ####### Case 1c: Test for rotate and flip (bw) #################################
  cat("CASE 1c: Testing for equality under rotation and reflection...\n")

  ### rotate by 90 degrees #######################
  rImgr = rotate(rImg,90)
  sImg = rImgr
  absoluteEquality(rImgr,sImg)

  ### flip horizontally
  rImgf = apply(rImg,2,rev)
  sImg = rImgr
  absoluteEquality(rImgr,sImg)

  cat("CASE 1c: testing complete.\n\n")

  ##### CASE 2: Erode only. ###################################
  cat("CASE 2: Testing for resulting mask having been only eroded and behavior of other library functions. You should expect to see ERROR messages pop up as we test the behavior of the functions, from the library functions being tested and the RUnit test package. This should happen in the test run.\n")

  #use this rImg for all subsequent cases
  rImg = matrix(rep(1,10000),ncol=100)
  rImg[61:80,31:45] = 0 #small 20 x 15 square

  #erode by small amount so that we still get 0
  kern <- makeBrush(3, shape='disc')
  sImg <- erode(1-rImg, kern) #result of an erosion, invert image for brush temporarily
  sImg <- 1-sImg
  wtlist <- generateNoScoreZone(rImg,3, 3,'disc')
  wts <- wtlist$wimg
  checkEquals(weightedL1(rImg,sImg,wts),0)
  err <- 0
  checkEquals(hingeL1(rImg,sImg,wts,err),0)

  #erode by a larger amount.
  kern <- makeBrush(5, shape='box')
  sImg <- erode(1-rImg, kern) #result of an erosion
  sImg <- 1-sImg
  wtlist <- generateNoScoreZone(rImg,3,3,'box')
  wts <- wtlist$wimg
  #should throw exception on differently-sized image
  #EDIT: implement message check?
  errImg = matrix(rep(0,1000),ncol=100)
  checkException(weightedL1(rImg,errImg,wts))
  checkException(weightedL1(rImg,sImg,wts[,51:100]))
  wts2 <- wts
  dim(wts2) <- c(200,50)
  checkException(weightedL1(rImg,sImg,wts2))
  checkException(hingeL1(rImg,errImg,wts,0.5))

  #want both to be greater than 0
  if (weightedL1(rImg,sImg,wts) == 0) {
    print("Case 2: weightedL1 is not greater than 0. Are you too forgiving?")
    quit(status=1)
  }
  if (hingeL1(rImg,sImg,wts,0.005) == 0) {
    print("Case 2: hingeL1 is not greater than 0. Are you too forgiving?")
    quit(status=1)
  }

  cat("CASE 2 testing complete.\n\n")

  ##### CASE 3: Dilate only. ###################################

  cat("CASE 3: Testing for resulting mask having been only dilated\n")

  #erode by small amount so that we still get 0
  kern <- makeBrush(3, shape='disc')
  sImg <- dilate(1-rImg, kern) #result of an erosion
  sImg <- 1-sImg
  wtlist <- generateNoScoreZone(rImg,3,3,'disc')
  wts <- wtlist$wimg
  checkEquals(weightedL1(rImg,sImg,wts),0)
  checkEquals(hingeL1(rImg,sImg,wts,0.5),0)

  #erode by a larger amount.
  kern <- makeBrush(5, shape='box')
  sImg <- erode(1-rImg, kern) #result of an erosion
  sImg <- 1-sImg
  wtlist <- generateNoScoreZone(rImg,3,3,'box')
  wts <- wtlist$wimg
  #want both to be greater than 0
  if (weightedL1(rImg,sImg,wts) == 0) {
    print("Case 3: weightedL1 is not greater than 0. Are you too forgiving?")
    quit(status=1)
  }
  if (hingeL1(rImg,sImg,wts,0.005) == 0) {
    print("Case 3: hingeL1 is not greater than 0. Are you too forgiving?")
    quit(status=1)
  }

  #dilate by small amount so that we still get 0
  kern <- makeBrush(3, shape='diamond')
  sImg <- dilate(1-rImg, kern) #result of an erosion
  sImg <- 1-sImg
  wtlist <- generateNoScoreZone(rImg,3,3,'diamond')
  wts <- wtlist$wimg
  checkEquals(weightedL1(rImg,sImg,wts),0)
  checkEquals(hingeL1(rImg,sImg,wts,0.5),0)

  #dilate by a larger amount
  kern <- makeBrush(5, shape='box')
  sImg <- erode(1-rImg, kern) #result of an erosion
  sImg <- 1-sImg
  wtlist <- generateNoScoreZone(rImg,3,3,'box')
  wts <- wtlist$wimg
  #want both to be greater than 0
  if (weightedL1(rImg,sImg,wts) == 0) {
    print("Case 3: weightedL1 is not greater than 0. Are you too forgiving?")
    quit(status=1)
  }
  if (hingeL1(rImg,sImg,wts,0.005) == 0) {
    print("Case 3: hingeL1 is not greater than 0. Are you too forgiving?")
    quit(status=1)
  }

  cat("CASE 3 testing complete.\n\n")

  ##### CASE 4: Erode + dilate. ###################################
  cat("CASE 4: Testing for resulting mask having been eroded and then dilated...\n")

  kern <- makeBrush(3, shape='Gaussian')
  sImg <- erode(1-rImg, kern)
  sImg <- dilate(sImg, kern)
  sImg <- 1-sImg
  wtlist <- generateNoScoreZone(rImg,3,3,'Gaussian')
  wts <- wtlist$wimg
  checkEquals(weightedL1(rImg,sImg,wts),0)
  checkEquals(hingeL1(rImg,sImg,wts,0.5),0)

  #erode and dilate by larger amount
  kern <- makeBrush(9, shape='Gaussian')
  sImg <- erode(1-rImg, kern)
  sImg <- dilate(sImg, kern)
  sImg <- 1-sImg
  checkEquals(weightedL1(rImg,sImg,wts),0)

  #erode and dilate by very large amount
  kern <- makeBrush(21, shape='Gaussian')
  sImg <- erode(1-rImg, kern)
  sImg <- dilate(sImg, kern)
  sImg <- 1-sImg
  #want both to be greater than 0
  if (weightedL1(rImg,sImg,wts) == 0) {
    print("Case 4: weightedL1 is not greater than 0. Are you too forgiving?")
    quit(status=1)
  }
  if (hingeL1(rImg,sImg,wts,0.05) == 0) {
    print("Case 4: hingeL1 is not greater than 0. Are you too forgiving?")
    quit(status=1)
  }

  cat("CASE 4 testing complete.\n\n")

  ##### CASE 5: Move. ###################################

  cat("CASE 5: Testing for resulting mask having been moved...\n")

  #move close
  sImg = matrix(rep(1,10000),ncol=100)
  sImg[59:78,33:47] = 0 #translate a small 20 x 15 square
  wtlist <- generateNoScoreZone(rImg,5,5,'Gaussian')
  wts <- wtlist$wimg
  checkEquals(weightedL1(rImg,sImg,wts),0)

  #move further
  sImg = matrix(rep(1,10000),ncol=100)
  sImg[51:70,36:50] = 0 #translate a small 20 x 15 square
  if (weightedL1(rImg,sImg,wts) == 0) {
    print("Case 5, translate more: weightedL1 is not greater than 0. Are you too forgiving?")
    quit(status=1)
  }
  if (hingeL1(rImg,sImg,wts,0.5) == 0) {
    print("Case 5, translate more: hingeL1 is not greater than 0. Are you too forgiving?")
    quit(status=1)
  }

  #move completely out of range
  sImg = matrix(rep(1,10000),ncol=100)
  sImg[31:45,61:80] = 0 #translate a small 20 x 15 square
  checkEquals(weightedL1(rImg,sImg,wts),476/9720)
  if (hingeL1(rImg,sImg,wts,0.5) == 0) {
    print("Case 5, translate out of range: hingeL1 is not greater than 0. Are you too forgiving?")
    quit(status=1)
  }
  cat("CASE 5 testing complete.\n\nAll mask scorer unit tests successfully complete.\n")

  quit(status=0)
}

unit.test()
