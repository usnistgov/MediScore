# *File: detectionMetrics.r
# *Date: 5/20/2016
# *Author: Yooyoung Lee
# *Status: Complete
#
# *Description: this code calculates performance measures (for ROC, AUC, and EER) 
# on system outputs (confidence scores) and return the metrics
#
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

# *Description: this function generates the weighted table for no score zone
#
#* Inputs
#    * rImg: reference mask image
#    * erodekernSize: Smoothing kernel size for Erosion
#    * dilatekernSize: Smoothing kernel size for Dilation
#    * myShape: Shape has options of box, disc, diamond, Gaussian (default) or line
#
#* Outputs
#    * weighted table data
generateNoScoreZone <- function(rImg, erodeKernSize, dilateKernSize, myShape)
{
  # provide size and shape for morphological process (erode and dilate)
  if(FALSE) #delete this section
  {
    if (kernSize == 0) {
      dims <- dim(rImg)
      weight <- matrix(rep(1,dims[1]*dims[2]),ncol=dims[2])
      return(list(rimg = imageData(rImg), wimg = weight))
    }
    
    if(kernSize %% 2 == 0)
    {
      stop("ERROR: The kernel size must be an odd integer")
    }
  }
  
  if (erodeKernSize == 0 & dilateKernSize == 0) {
    dims <- dim(rImg)
    weight <- matrix(rep(1,dims[1]*dims[2]),ncol=dims[2])
    return(list(rimg = imageData(rImg), wimg = weight))
  }
  
  if((erodeKernSize %% 2 == 0) &(dilateKernSize %% 2 == 0))
  {
    stop("ERROR: The kernel size must be an odd integer")
  }
  
  eKern <- makeBrush(erodeKernSize, shape=myShape)
  dKern <- makeBrush(dilateKernSize, shape=myShape)
  eImg <- erode(rImg, eKern) #This is actually dilation
  dImg <- dilate(rImg, dKern) #This is actually erosion
  #display(eimg, title='Erosion of reference mask')
  #display(dilate, title='Dilatation of reference mask')
  weight <- abs(imageData(eImg) - imageData(dImg))
  #0's flipped to 1's and 1's flipped to 0's
  wFlip <- binary.flip(weight)
  
  return(list(wimg = wFlip, eimg = imageData(dImg), dimg = imageData(eImg))) # intentially exchanged dImg and eImg here
}

#*Metric: measure_ns_new
#*Description: this function calculates an area of TP, TN, FP, and FN
#               between the reference mask and the system output mask,
#               accommodating the no score zone
#
#*Inputs
#* ref: binary data for reference mask
#* sys: binary or grayscale (normalized as [0,1]) data for system output mask
#* wts: binary data from weigthed table generated
#
# *Outputs
#* MET: list of the TP, TN, FP, and FN area, and N (total score region)

measures_ns_new <- function(r, s, w)
{
  x <- w*(r - s)
  
  #if the x values are equal to zero, count as 1 (0 is a mask area)
  sum_same_values <- length(x[x==0 & r==0 & s != 1 & w ==1])
  #if the x values are negative values, make a absolute value and sum them all
  sum_neg_values <- sum(abs(x[x<0 & r==0 & s != 1 & w ==1]))
  tp <- sum_same_values + sum_neg_values
  tp_len <-length(x[x<=0 & r == 0 & s != 1 & w == 1])
  
  fp_len <- length(x[x>0 & r==1 & w == 1])
  fp <- sum(x[x>0 & r==1 & w == 1])
  
  fn_len <- length(x[x==-1 & r==0 & w == 1])
  #Substact the TP area from the reference mask area
  fn <- length(r[r==0 & w == 1]) - tp
  
  tn_len <- length(x[x==0 & r==1 & w == 1])
  tn <- length(r[r==1 & w == 1]) - fp
  
  n <- tp_len + tn_len + fp_len + fn_len # this should be equal to (total pixels - length(w[w==1]))
  
  #checkNum <- length(s) - length(w[w==0])
  #cat("N: ", n, "Checknum: ", checkNum, "\n")
  #if(n != checkNum)
  #  print ("Error with pixel numbers")
  
  #s<-(tp+fn)/n
  #p<-(tp+fp)/n
  
  # Matthews correlation coefficient (MCC)
  #if ((s==1) | (p==1) | (s==0) | (p==0)){
  #  mcc <- 0
  #} else {
  #  mcc<-(tp/n-s*p)/sqrt(p*s*(1-s)*(1-p))
  #}
  
  #cat("TP: ", tp, " TN: ", tn, " FP: ", fp, " FN: ", fn, "\n")
  #cat("TPLEN: ", tp.len, " TNLEN: ", tn.len, " FPLEN: ", fp.len, " FNLEN: ", fn.len, "\n")
  return(list(TP = tp, TN = tn, FP = fp, FN = fn, N = n))
}

######this is the same with measures_ns_new(), please ignore this function ########
measures_ns <- function(ref, sys, wts)
{
  #ref <- rData
  #sys <- sData
  #wts <- wData

  tp.sum<-function(r, s, w)
  {
    x <- w*(r - s)
    #tp <- length(x[x==0 & ref==0 & sys != 1 & wts ==1]) + sum(abs(x[x <0 & ref==0 & sys != 1 & wts ==1]))
    #if the x values are equal to zero, count as 1 (0 is a mask area)
    sum_same_values <- length(x[x==0 & r==0 & s != 1 & w ==1])
    #if the x values are negative values, make a absolute value and sum them all
    sum_neg_values <- sum(abs(x[x<0 & r==0 & s != 1 & w ==1]))
    tp <- sum_same_values + sum_neg_values
    tp_len <-length(x[x<=0 & r == 0 & s != 1 & w == 1])
    #cat("TP: ", tp, "Length: ", tp_len, "\n")
    return(list(TP=tp, TPLEN=tp_len))
  }
  
  fp.sum<-function(r, s, w)
  {
    x <- w*(r - s)
    fp_len <- length(x[x>0 & r==1 & w == 1])
    fp <- sum(x[x>0 & r==1 & w == 1])
    #cat("FP: ", fp, "Length:", fp_len, "\n")
    return(list(FP=fp, FPLEN=fp_len))
  }
  
  
  fn.sum<-function(r, s, w, tpp)
  {
    x <- w*(r - s)
    fn_len <- length(x[x==-1 & r==0 & w == 1])
    #Substact the TP area from the reference mask area
    fn <- length(r[r==0 & w == 1]) - tpp
    #cat("FN: ", fn, "Length: ", fn_len, "\n")
    return(list(FN=fn, FNLEN=fn_len))
  }
  

  tn.sum<-function(r, s, w, fpp)
  {
    x <- w*(r - s)
    tn_len <- length(x[x==0 & r==1 & w == 1])
    tn <- length(r[r==1 & w == 1]) - fpp
    #cat("TN: ", tn, "Length: ", tn_len, "\n")
    return(list(TN=tn, TNLEN=tn_len))
  }
  
  #diff <- wts*(ref - sys) # argument "diff" is missing problem
  tp_list<-tp.sum(ref, sys, wts)
  fp_list<-fp.sum(ref, sys, wts)
  tn_list<-tn.sum(ref, sys, wts, fp_list$FP)
  fn_list<-fn.sum(ref, sys, wts, tp_list$TP)
  
  tp<-tp_list$TP
  tn<-tn_list$TN
  fp<-fp_list$FP
  fn<-fn_list$FN
  
  n <- tp_list$TPLEN + tn_list$TNLEN + fp_list$FPLEN + fn_list$FNLEN # this should be equal to (total pixels - length(w[w==1]))
  checkNum <- length(sys) - length(wts[wts==0])
  cat("N: ", n, "Checknum: ", checkNum, "\n")
  if(n != checkNum)
    print ("WARNING: pixel numbers don't match.")
  
  #cat("TP: ", tp, " TN: ", tn, " FP: ", fp, " FN: ", fn, "\n")
  #cat("TPLEN: ", tp.len, " TNLEN: ", tn.len, " FPLEN: ", fp.len, " FNLEN: ", fn.len, "\n")
  return(list(TP = tp, TN = tn, FP = fp, FN = fn, N = n))
}


#*Metric: measure_wh_no_ns
# *Description: this function calculates FPR, TPR, ACC, and MCC
#               between the reference mask and the system output mask,
#               without the no score zone
#
# *Inputs
#* ref: binary data for reference mask
#* sys: binary or grayscale (normalized as [0,1]) data for system output mask
#
# *Outputs
# * MET: list of FPR, TPR, ACC, and MCC

measures_no_ns <- function(ref, sys)
{
  if (length(ref) != length(sys))
  {
    stop("ERROR: The length between reference and system output is different")
  }
  
  tp.sum<-function(r,s)
  {
    #print(sum(r & s))
    return(sum(r & s))
  }
  
  tn.sum<-function(r,s)
  {
    #print(length(ref)-sum(r | s))
    return(length(r)-sum(r | s))
  }
  
  fp.sum<-function(r,s)
  {
    x<-r - s
    #print(abs(sum(x[x==-1])))
    return(abs(sum(x[x==-1])))
  }
  
  fn.sum<-function(r,s)
  {
    x<-r - s
    #print(sum(x[x==1]))
    return(sum(x[x==1]))
  }
  
  tp<-tp.sum(ref, sys)
  tn<-tn.sum(ref, sys)
  fp<-fp.sum(ref, sys)
  fn<-fn.sum(ref, sys)
  
  n<-tp+tn+fp+fn
  s<-(tp+fn)/n
  p<-(tp+fp)/n
  
  # False positive rate (FPR)
  fpr<-fp/(fp+tn)
  
  # Sensitivity or true positive rate (TPR)
  tpr<-tp/(tp+fn)
  
  # Accuracy (ACC)
  acc<-(tp+tn)/n
  
  # Matthews correlation coefficient (MCC)
  if ((s==1) | (p==1) | (s==0) | (p==0)){
    mcc <- 0
  } else {
    mcc<-(tp/n-s*p)/sqrt(p*s*(1-s)*(1-p))
  }
  
  #cat("TP: ", tp, " TN: ", tn, " FP: ", fp, " FN: ", fn, "\n")
  return(list(FPR = fpr, TPR = tpr, ACC = acc, MCC = mcc))
}

#*Metric: NMM
#*Description: this function calculates the system mask score
#               based on the measures_ns function
#* Inputs
#*   r: binary data for reference mask
#*   w: binary data from weigthed table generated
#*   tp: true positive
#*   fn: false negative
#*   fp: false positive
#*   c: forgiveness value
#* Outputs
#*   Score range [-1, 1]

NimbleMaskMetric <- function(r, w, tp, fn, fp, c=-1)
{

  Rgt <-length(r[r==0 & w == 1])
  score <- max(c, (tp - fn - fp)/Rgt)
  return(score)
}

#*Metric: MCC (Matthews correlation coefficient)
#*Description: this function calculates the system mask score
#               based on the MCC function
#* Inputs
#*   tp: true positive
#*   fn: false negative
#*   fp: false positive
#*   n: total number of pixels except the no score zone
#* Outputs
#*   Score range [0, 1]

matthews <- function(tp, fn, fp, n)
{  
  s<-(tp+fn)/n
  p<-(tp+fp)/n
  
  # Matthews correlation coefficient (MCC)
  if ((s==1) | (p==1) | (s==0) | (p==0)){
    score <- 0.0
  } else {
    score<-(tp/n-s*p)/sqrt(p*s*(1-s)*(1-p))
  }
  return(score)
}
# *Description: this function calculates Hamming distance
#               between the reference mask and the system output mask
#
# *Inputs
#* r: binary data for reference mask
#* s: binary data for system output mask
#
#* Outputs
#    * Hamming distance value
hamming <- function(r, s)
{
  if (length(r) != length(s))
  {
    stop("The length between reference and system output is different")
  }
  val <- sum(xor(r, s))/length(r)
  #hamming <- sum(r != s)/length(r)
  return (val)
}


# *Description: this function calculates Weighted L1 loss
#               and normalize the value with the no score zone
#
#* Inputs
#    * r: binary data fro reference mask
#    * s: binary data fro system output mask
#    * w: binary data from weighted table generated
#
#* Outputs
#    * Normalized WL1 value
weightedL1 <- function(r, s, w)
{
  if ((length(w) != length(r)) | (length(w) != length(s)))
  {
    stop("ERROR: The length of reference, system output, and weight is different")
  }
  wL1 <- sum(w * abs(r-s))
  #cat("r:", "\n", r, "\n")
  #cat("s:", "\n", s, "\n")
  #cat("r-s:", "\n", abs(r-s), "\n")
  #cat("r-s:", "\n", abs(r-s), "\n")
  #cat("weight:", "\n", w, "\n")
  #cat("final:", "\n", w * abs(r-s))
  n <- length(w[w==1])
  norm_wL1 <- wL1/n
  return(norm_wL1)
}

# *Description: this function calculates Hinge L1 loss
#               and normalize the value with the no score zone
#
#* Inputs
#    * r: binary data fro reference mask
#    * s: binary data fro system output mask
#    * w: binary data from weighted table generated
#    * e: epsilon (default = 0.1)
#
#* Outputs
#    * Normalized HL1 value"

hingeL1 <- function(r, s, w, e)
{
  if ((length(w) != length(r)) | (length(w) != length(s)))
  {
    stop("ERROR: The length of reference, system output, and weight is different")
  }
  if (e < 0) {
    print("Your chosen epsilon is negative. Setting to 0.")
    e = 0
  }
  wL1 <- sum(w * abs(r-s))
  # L1 Hinge Loss
  rArea <- length(r[r==0]) # 0 reprents the mask area
  hL1 <- max(0, wL1 - e * rArea )
  #cat("Dh: ", hDm, "\n")
  n <- length(w[w==1])
  norm_hL1 <- hL1/n
  return(norm_hL1)
}

