#!/usr/local/bin/Rscript
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1)
{
  print("Usage: at least one argument must be supplied (see --help)")
  q(status=1)
}

#"* File: MaskScorer.r
#* Date: 03/04/2016
#* Author: Yooyoung Lee
#* Status: In progress
#
#* Description: This calculates performance scores for localizing mainpulated area 
#                between reference mask and system output mask 
#* Example metrics:
#    * NIMBLE mask metric (NMM)
#    * Matthews correlation coefficient (MCC)
#    * Hamming Loss (HAM)
#    * Weighted L1 (WL1)
#    * Hinge weighted L1 (HL1)
#
#* Requirements: This code requires the following packages:
#
#    - require(data.table)   
#    - require(optparse)
#    - require(useful)
#    - require(RUnit)
#    - require(EBImage)
#    * EBImage package installation
#    - source('http://bioconductor.org/biocLite.R')
#    - biocLite('EBImage')
#    - require(RMySQL)
#
#* Disclaimer: 
#This software was developed at the National Institute of Standards 
#and Technology (NIST) by employees of the Federal Government in the
#course of their official duties. Pursuant to Title 17 Section 105 
#of the United States Code, this software is not subject to copyright 
#protection and is in the public domain. NIST assumes no responsibility 
#whatsoever for use by other parties of its source code or open source 
#server, and makes no guarantees, expressed or implied, about its quality, 
#reliability, or any other characteristic."
#
##############Scoring##############################################

# loading libraries
lib_path <- "../../lib"
source(file.path(lib_path, "maskMetrics.r"))

### Calculate all metrics
#* Description: This loads the image file names and calculates all metrics
#* Inputs
#    * refImgName: reference mask file name
#    * sysImgName: system output mask file name
#    * erodekernSize: Kernel size for Erosion
#    * dilatekernSize: Kernel size for Dilation
#    * thres: threshold input[0,1] for grayscale mask
#    * metricOption: options [nmm, mcc, ham, wL1, hL1, and all]
#* Outputs
#    * List of NMM, MCC, HAM, WL1, and HL1

getMetrics <- function(refImgName, sysImgName, erodeKernSize, dilateKernSize, thres=0, metricOption="nmm")
{
  #refImgName = "../../data/SystemOutputs/dct/remaskscorer/NC2016_7876.png"
  #sysImgName = "../../data/SystemOutputs/dct/remaskscorer/NC2016_5749.png"
  
  # initialize metric values
  nmm <- -1
  mcc <- 0
  ham <- 0
  wL1 <- 0
  hL1 <- 0
  

  #Read a reference mask and a system output mask
  #Note that "ReadImage" from EBImage package automatically read in grayscale [0 to 1]
  rImg <- readImage(refImgName)
  #cat("Ref Mask:", refImgName, ": ", dim(rImg), "\n")
  sImg <- readImage(sysImgName)
  #cat("Sys Mask:", sysImgName, ": ", dim(sImg), "\n")
  
  #if(FALSE) #opencv is much faster
  #{
  #  orImg <- readImg(refImgName);
  #  osImg <- readImg(sysImgName);
  #  rImg = img2r(orImg)[,,1]/255;
  #  sImg = img2r(osImg)[,,1]/255;
  #}

  #sImg[which(imageData(sImg) > 0.5)] <- 1
  
  # for grayscale system output, 
  # if the threshold value is not equal to 0, then convert it to the binary file
  if(thres != 0)
  {
    print("converting to binary")
    sImg[which(imageData(sImg) > thres)] <- 1
    sImg[which(imageData(sImg) <= thres)] <- 0  
    #print(unique(imageData(sImg)))
  }
  
  # This section is for testing
  #kern <- makeBrush(25, shape="Gaussian")
  #sImg <- erode(sImg, kern)
  #sImg <- dilate(sImg, kern)
  
 
  
  #TO BE DELETED: if the image dimension is different, transpose the reference dimension.
  #Note: no longer need this part
  #rImgData <- imageData(rImg)
  #if(dim(rImg)[1] != dim(sImg)[1]) 
  #{
  #  rImgData <- t(imageData(rImg))
  #}
  #w <- generateNoScoreZone(rImgData, erodeKernSize, dilateKernSize, "Gaussian")
  w <- generateNoScoreZone(rImg, erodeKernSize, dilateKernSize, "Gaussian")
  
  # Convert data to vector for comparison
  rData <- as.vector(imageData(rImg))
  #rData <- as.vector(rImgData)
  sData <- as.vector(imageData(sImg))
  wData <- as.vector(w$wimg)
  

  #TO BE DELETED later once runing validators 
  if ((length(wData) != length(rData)) | (length(wData) != length(sData)))
  {
    cat("Ref: ", length(rData), "Sys: ", length(sData), "Weight: ", length(wData), "\n")
    print("The length of reference and system output masks is different", call.=FALSE)
    #q(status=1)
    return(list(R = NULL, S = NULL, W = NULL, E = NULL, D = NULL, NMM=nmm, MCC=mcc, HAM=ham, WL1=wL1, HL1=hL1))
  }
  else
  {
    
    #cat("Ref data length: ", length(rData), ", Sys data length: ", length(sData), ", Weighted data length: ",length(wData), "\n") 
    
    #t<-system.time(met <- measures_ns(rData, sData, wData)) #with score zone
    #t<-system.time(met <- measures_ns_new(rData, sData, wData)) #with score zone
    #print(t)
    
    
    
    #NIMBLE Mask Metric
    if(metricOption == "nmm")
    {
      met <- measures_ns_new(rData, sData, wData)
      forgiven <- -1.0 # the lower number will provide more forgiveness to the system
      nmm <- NimbleMaskMetric(rData, wData, met$TP, met$FN, met$FP, forgiven)
      cat("NMM: ", nmm, "\n")
    }
    
    # Matthews correlation coefficient
    if(metricOption == "mcc")
    {
      met <- measures_ns_new(rData, sData, wData)
      mcc <- matthews(met$TP, met$FN, met$FP, met$N)
      cat("MCC (Matthews correlation coeff.): ", mcc, "\n")
    }
    
    # Hamming distance
    if(metricOption == "ham")
    {
      ham <- hamming(rData, sData)
      cat("Hamming Loss: ", ham, "\n")
    }
    
    # Weighted L1
    if(metricOption == "wL1")
    {
      wL1 <- weightedL1(rData, sData, wData)
      cat("Weighted L1: ", wL1, "\n")
    }
    
    # Hinge loss L1
    if(metricOption == "hL1")
    {
      e <- 0.1
      hL1 <- hingeL1(rData, sData, wData, e)
      cat("Hinge Loss L1: ", hL1, "\n")
    }
    
    # all metrics
    if(metricOption == "all")
    {
      met <- measures_ns_new(rData, sData, wData)
      forgiven <- -1.0 # the lower number will provide more forgiveness to the system
      nmm <- NimbleMaskMetric(rData, wData, met$TP, met$FN, met$FP, forgiven)
      cat("NMM: ", nmm, "\n")
    
      # Matthews correlation coefficient
      mcc <- matthews(met$TP, met$FN, met$FP, met$N)
      cat("MCC (Matthews correlation coef.): ", mcc, "\n")
      
      # Hamming distance
      ham <- hamming(rData, sData)
      cat("Hamming Loss: ", ham, "\n")
      
      # Weighted L1
      wL1 <- weightedL1(rData, sData, wData)
      cat("Weighted L1: ", wL1, "\n")
      
      # Hinge loss L1
      e <- 0.1
      hL1 <- hingeL1(rData, sData, wData, e)
      cat("Hinge Loss L1: ", hL1, "\n")
    }
    
    #return(list(R = imageData(rImg), S = imageData(sImg), W = w$wimg, E= w$eimg, D = w$dimg, NMM=nmm, MCC=mcc, HAM=ham, WL1=wL1, HL1= hL1))
    return(list(R = rImgData, S = imageData(sImg), W = w$wimg, E= w$eimg, D = w$dimg, NMM=nmm, MCC=mcc, HAM=ham, WL1=wL1, HL1= hL1))
  }
}

### Colored error area in the manipulated image
#* Description: This saves the output image illustrating the error area in the manipulated image
#* Inputs
#    * refImgName: reference mask file name
#    * sysImgName: system output mask file name
#    * erodekernSize: Kernel size for Erosion
#    * dilatekernSize: Kernel size for Dilation
#    * rData: reference image data
#    * sData: system image data
#    * wData: no score zone image data
#    * eData: image data after erosion
#    * rData: image data after dilation
#* Outputs
#    * List of NMM, MCC, HAM, WL1, and HL1
#
coloredMask_opt1 <- function(sysImgName, maniImgName, erodeKernSize, dilateKernSize, rData, sData, wData, eData, dData, outputMaskPath)
{
  if(FALSE)
  {
    #refImgName = "../../data/NC2016_Test0601/reference/splice/mask/NC2016_2228.png"
    #sysImgName = "../../data/SystemOutputs/splice0601/mask/NC2016_0048_0_1.png"
    #maniImgName = "../../data/NC2016_Test0601/probe/NC2016_0048.jpg"
    refImgName = "../../data/SystemOutputs/dct/remaskscorer/NC2016_7876.png"
    sysImgName = "../../data/SystemOutputs/dct/remaskscorer/NC2016_5749.png"
    maniImgName = "../../data/SystemOutputs/dct/remaskscorer/NC2016_5749.jpg"
    rImg = readImage(refImgName)
    sImg = readImage(sysImgName)
    #require(png)
    #rImg = readPNG(refImgName)
    #sImg = readPNG(sysImgName)
    rData = imageData(rImg)
    sImg[which(imageData(sImg) > 0.5)] <- 1
    sData = imageData(sImg)
    m <- generateNoScoreZone(rImg, 15, 9, "Gaussian")
    wData <- m$wimg
    eData <- m$eimg
    dData <- m$dimg
    
    dim(rData)
    dim(sData)
    dim(wData)
    dim(eData)
    dim(dData)
    
    if(FALSE)
    {
      rData <- t(rData)
      sData <- t(sData)
      wData <- t(wData)
      eData <- t(eData)
      dData <- t(dData)
      o_maniImg = readImg(maniImgName)
      mData = img2r(o_maniImg)[,,1]/255;
      mData = t(mData) #the opencv dim is opposite to EBImage
    }
  }
  
  maniImg = readImage(maniImgName)
  mData <-imageData(maniImg)[,,1] # reading as one channel (grayscale)
  #dim(mData)

  # values are fliped becasue the black area is the ground-truth area.
  b.rImg <- binary.flip(rData)
  b.sImg <- binary.flip(sData)
  b.wImg <- binary.flip(wData)
  b.eImg <- binary.flip(eData)
  b.dImg <- binary.flip(dData)
  
  b.sImg[which(imageData(b.sImg) != 0)] <- 1
  #b.rImg[which(imageData(b.rImg) != 0)] <- 10
  
  b.eImg[which(imageData(b.eImg) != 0)] <- 2
  b.wImg[which(imageData(b.wImg) != 0)] <- 5
  
  #length(b.eImg[which(imageData(b.eImg) == 2)])
  #length(b.wImg[which(imageData(b.wImg) == 5)])
  
  ewImg <- b.eImg + b.wImg
  #ewImg[which(imageData(ewImg) != 0)]
  #length(ewImg[which(imageData(ewImg) != 0)])
  
  mImg <- ewImg + b.sImg #sys error
  #errorImg = mImg
  #unique(errorImg[which(imageData(errorImg) != 0)])
  mImg[which(imageData(mImg) != 3)] <- 0
  #display(mImg, method="raster")
  
  col_fa <- 'blue'
  col_ns <- 'yellow'
  col_gt_br <- 'black' #ground-true boundary
  col_fn <- 'red'
  col_tp <- 'green'
  myopac <- 0.5
  
  res <- paintObjects(b.sImg, toRGB(mData), opac=c(myopac, myopac), col=c(col_fa, col_fa))
  res <- paintObjects(b.wImg, res, opac=c(myopac, myopac), col=c(col_ns, col_ns))
  res <- paintObjects(b.rImg, res, opac=c(myopac, myopac), col=c(col_gt_br, col_ns), thick =TRUE)
  res <- paintObjects(b.eImg, res, opac=c(myopac, myopac), col=c(col_fn, col_fn))
  res <- paintObjects(mImg, res, opac=c(myopac, myopac), col=c(col_tp, col_tp))
  #display(res, method="raster")
  
  #grep the system mask output name
  outputMaskName <- strsplit(basename(sysImgName), '[.]')[[1]][1]
  finalMaskName <- paste0(outputMaskName,"_colored.jpg")
  
  #cat("colMaskName: ", finalMaskName, "\n")
  writeImage(res, file.path(outputMaskPath, finalMaskName))
  #writeImg(finalMaskName, res)
  return(file.path(outputMaskPath, finalMaskName))
}

### Mask performance per image pair
# *Description: this function calculates metrics for a pair of mask images
# *Inputs
#   *numOfRows: the number of pair images 
#   *refMaskFName: vector of reference mask file names
#   *sysMaskFName: vector of system output mask file names
#   *maniImageFName: vector of manipulated image file names
#   *refDir: reference mask file directory
#   *sysDir: system output mask file directory
#   *erodekernSize: Kernel size for Erosion
#   *dilatekernSize: Kernel size for Dilation
#   *isMaskOut: option for mask outputs
#   *thres: threshold input[0,1] for grayscale mask
#   *metricOption: options [nmm, mcc, ham, wL1, hL1, and all]
# *Outputs
#   * report dataframe

scores_4_mask_pairs <- function(numOfRows, refMaskFName, sysMaskFName, maniImageFName, refDir, sysDir, erodeKernSize, dilateKernSize, isMaskOut, thres, metricOption, outputMaskPath)
{
  # Create a dataframe
  df<-data.frame("NMM"=rep(-1, numOfRows),"MCC"=rep(0, numOfRows),"HAM"=rep(0, numOfRows),
                 "WL1"=rep(0, numOfRows),"HL1"=rep(0, numOfRows), "ColMaskFileName"=rep(NA, numOfRows)) 
  mydigits <- 5 #digits for round

  for(i in 1: nrow(df))
  #for(i in 1:6)
  {
    refMaskName <- file.path(refDir, refMaskFName[i])
    #print(refMask)
    sysMaskName <- file.path(sysDir, sysMaskFName[i]) #change it to exp path (inputs)
    #print(sysMask)

      if(is.null(sysMaskFName[i]))
      {
        print("Empty mask file")
        df$NMM[i] <- -1
        df$MCC[i] <- 0
        df$HAM[i] <- 0
        df$WL1[i] <- 0
        df$HL1[i] <- 0
        
      } else if (is.na(sysMaskFName[i])) {
        print("Empty mask file")
        df$NMM[i] <- -1
        df$MCC[i] <- 0
        df$HAM[i] <- 0
        df$WL1[i] <- 0
        df$HL1[i] <- 0
      } else if (sysMaskFName[i] == "") {
        print("Empty mask file")
        df$NMM[i] <- -1
        df$MCC[i] <- 0
        df$HAM[i] <- 0
        df$WL1[i] <- 0
        df$HL1[i] <- 0
      } else
      {
        metric <- getMetrics(refMaskName, sysMaskName, erodeKernSize, dilateKernSize, thres, metricOption)
        #metric <- getMetrics(refMask, sysMask, 15, 9) #for testing    
        df$NMM[i] <- round(metric$NMM, digits = mydigits)
        df$MCC[i] <- round(metric$MCC, digits = mydigits)
        df$HAM[i] <- round(metric$HAM, digits = mydigits)
        df$WL1[i] <- round(metric$WL1, digits = mydigits)
        df$HL1[i] <- round(metric$HL1, digits = mydigits)
        
        # if there is no system ouput masks, then do not perform this function
        if(isMaskOut == "y" & !is.null(metric$R)) 
        {
          colMaskFName <- file.path(refDir, maniImageFName[i]) 
          df$ColMaskFileName[i] <- coloredMask_opt1(sysMaskName, colMaskFName, erodeKernSize, dilateKernSize, metric$R, metric$S, metric$W, metric$E, metric$D, outputMaskPath)
        }
    }
  }
  return(df)
}

### SSD average mask performance by a factor
# *Description: this function returns a CSV report with the average mask performance by a factor
# *Inputs
#   *df: the dataframe with 
#   *taskType: [manipulation, removal, splice]
#   *bySet: the reports will be seperated by DatasetID(e.g., y/n, default: n)
#   *byPost: the reports will be seperated by PostProcessingID(e.g., y/n, default: n)
# *Outputs
#   *report dataframe

avg_scores_by_factors_SSD <- function(df, taskType, bySet, byPost)
{
  set.id <- "*"
  post.id <- "*"
  
  if(bySet == "y")
  {
    set.id <- unique(df$Collection)
  }  
  
  if(byPost == "y")
  {
    post.id <- unique(df$ProbePostProcessed)
  }
  
  #num.runs <- length(task.id) * length(set.id) * length(post.id) 
  num.runs <- length(set.id) * length(post.id)  
  
  df_avg<-data.frame("runID"=rep(NA, num.runs), "TaskID"=rep(taskType, num.runs), "Collection"=rep(NA, num.runs),"PostProcessed"=rep(NA, num.runs),
                     "NMM"=rep(NA, num.runs),"MCC"=rep(NA, num.runs),"HAM"=rep(NA, num.runs),"WL1"=rep(NA, num.runs),"HL1"=rep(NA, num.runs))
  
  sub.d <- df
  idx <- 1 
  mydigits <- 5
  
  # todo: for optimization
  # df_avg$runID <- seq(1,length(set.id)*length(post.id),by=1)
  # df_avg$Collection <- as.character(rep(set.id,each=length(post.id)))
  # df_avg$PostProcessed <-
  # aggDF <- aggregate(cbind(NMM,MCC,HAM,WL1,HL1) ~ Collection + ProbePostProcessed,data=df,FUN=mean, na.rm=TRUE)
  
  for (j in 1:length(set.id))
  {
    for (k in 1:length(post.id))
    { 
      #sub.d <- subset(df, grepl(task.id[i], TaskID) & grepl(set.id[j], Collection) & grepl(post.id[k], ProbePostProcessed))
      sub.d <- subset(df, grepl(set.id[j], Collection) & grepl(post.id[k], ProbePostProcessed))
      
      cat(idx, " - Length of sub.df: ", nrow(sub.d), "\n")
      
      df_avg$runID[idx] <- idx
      df_avg$Collection[idx] <- as.character(set.id[j])
      df_avg$PostProcessed[idx] <- as.character(post.id[k])
      df_avg$NMM[idx] <- round(mean(na.omit(sub.d$NMM)), digits = mydigits)
      df_avg$MCC[idx] <- round(mean(na.omit(sub.d$MCC)), digits = mydigits)
      df_avg$HAM[idx] <- round(mean(na.omit(sub.d$HAM)), digits = mydigits)
      df_avg$WL1[idx] <- round(mean(na.omit(sub.d$WL1)), digits = mydigits)
      df_avg$HL1[idx] <- round(mean(na.omit(sub.d$HL1)), digits = mydigits)
      idx <- idx + 1
    }
  }
  return(df_avg)
}

### DSD average mask performance by a factor
# *Description: this function returns a CSV report with the average mask performance by a factor
# *Inputs
#   *df: the dataframe with 
#   *taskType: [manipulation, removal, splice]
#   *bySet: the reports will be seperated by DatasetID(e.g., y/n, default: n)
#   *byPost: the reports will be seperated by PostProcessingID(e.g., y/n, default: n)
# *Outputs
#   *report dataframe
avg_scores_by_factors_DSD <- function(df, taskType, bySet, byPost)
{
  
  set.id <- "*"
  post.id <- "*"
  
  if(bySet == "y")
  {
    set.id <- unique(df$Collection)
  }  
  
  if(byPost == "y")
  {
    post.id <- unique(paste(df$ProbePostProcessed, df$DonorPostProcessed))
  }
  
  num.runs <- length(set.id) * length(post.id)  
  
  df_avg<-data.frame("runID"=rep(NA, num.runs), "TaskID"=rep(taskType, num.runs), "Collection"=rep(NA, num.runs),"PostProcessed"=rep(NA, num.runs),
                     "pNMM"=rep(NA, num.runs),"pMCC"=rep(NA, num.runs),"pHAM"=rep(NA, num.runs),"pWL1"=rep(NA, num.runs),"pHL1"=rep(NA, num.runs),
                     "dNMM"=rep(NA, num.runs),"dMCC"=rep(NA, num.runs),"dHAM"=rep(NA, num.runs),"dWL1"=rep(NA, num.runs),"dHL1"=rep(NA, num.runs))
  
  sub.d <- df
  idx <- 1 
  mydigits <- 5
  
  for (j in 1:length(set.id))
  {
    for (k in 1:length(post.id))
    { 
      #sub.d <- subset(df, grepl(task.id[i], TaskID) & grepl(set.id[j], ImageCollection) & grepl(post.id[k], ProbePostProcessed))
      sub.d <- subset(df, grepl(set.id[j], Collection) & grepl(post.id[k], paste(ProbePostProcessed, DonorPostProcessed)))
      
      cat(idx, " - Length of sub.df: ", nrow(sub.d), "\n")
      
      df_avg$runID[idx] <- idx    
      df_avg$Collection[idx] <- as.character(set.id[j])
      df_avg$PostProcessed[idx] <- as.character(post.id[k])
      df_avg$pNMM[idx] <- round(mean(na.omit(sub.d$pNMM)), digits = mydigits)
      df_avg$pMCC[idx] <- round(mean(na.omit(sub.d$pMCC)), digits = mydigits)
      df_avg$pHAM[idx] <- round(mean(na.omit(sub.d$pHAM)), digits = mydigits)
      df_avg$pWL1[idx] <- round(mean(na.omit(sub.d$pWL1)), digits = mydigits)
      df_avg$pHL1[idx] <- round(mean(na.omit(sub.d$pHL1)), digits = mydigits)
      
      df_avg$dNMM[idx] <- round(mean(na.omit(sub.d$dNMM)), digits = mydigits)
      df_avg$dMCC[idx] <- round(mean(na.omit(sub.d$dMCC)), digits = mydigits)
      df_avg$dHAM[idx] <- round(mean(na.omit(sub.d$dHAM)), digits = mydigits)
      df_avg$dWL1[idx] <- round(mean(na.omit(sub.d$dWL1)), digits = mydigits)
      df_avg$dHL1[idx] <- round(mean(na.omit(sub.d$dHL1)), digits = mydigits)
      idx <- idx + 1
    }
  }
  
  return(df_avg)
}

### Create a CSV report for single source detection
# *Description: this function calls each metric function and 
#               return the metric value and the colored mask output as a report
# *Inputs
# *ref: reference dataframe
# *sys: system output dataframe
# *index: index dataframe
#   *refDir: reference mask file directory
#   *sysDir: system output mask file directory
#   *erodekernSize: Kernel size for Erosion
#   *dilatekernSize: Kernel size for Dilation
#   *isMaskOut: option for mask outputs [y/n]
#   *taskType: [manipulation, removal, splice]
#   *bySet: the reports will be seperated by DatasetID(e.g., y/n, default: n)
#   *byPost: the reports will be seperated by PostProcessingID(e.g., y/n, default: n)
#   *thres: threshold input[0,1] for grayscale mask
#   *metricOption: options [nmm, mcc, ham, wL1, hL1, and all]
# *Outputs
#   *report dataframe

createReportSSD <- function(ref, sys, index, refDir, sysDir, erodeKernSize, dilateKernSize, isMaskOut, taskType, bySet, byPost, thres, metricOption, outputMaskPath)
{
   
  if(FALSE)
  {
    #TO BE DELETED: rename system csv column names
    #colnames(sys)<-c("ProbeFileID", "ConfidenceScore", "ProbeOutputMaskFileName")
    
    ref <- myRef # for testing
    
    #colnames(ref)<-c("TaskID", "ProbeFileID", "ProbeFileName", "ProbeMaskFileName", 
    #    "DonorFileID", "DonorFileName", "DonorMaskFileName", "IsTarget", 
    #    "ProbePostProcessed", "DonorPostProcessed", "ManipulationQuality", 
    #    "IsManipulationTypeRemoval", "IsManipulationTypeSplice", "IsManipulationTypeCopyClone", 
    #    "Collection", "BaseFileName", "Lighting", "IsControl", "CorrespondingControlFileName", "SemanticConsistency")
    sys <- mySys # for testing
    colnames(sys)<-c("ProbeFileID", "ConfidenceScore", "ProbeOutputMaskFileName")
    refDir <- myRefDir
    sysDir <- mySysDir
    index <- myIndex
    #colnames(index) <- c("TaskID", "ProbeFileID", "ProbeFileName", "ProbeWidth", "ProbeHeight")
    erodeKernSize <- 11
    dilateKernSize <- 5
    isMaskOut <- "y"
    #Test
    taskType <- "manipulation"
    bySet <- "y"
    byPost <- "y"
  }

  sub_ref <- ref[ref$IsTarget=="Y",] # grep only the target class
  index <- subset(index, select =c("ProbeFileID","ProbeWidth", "ProbeHeight")) #due to ProbeFileName duplication
  # subset for the target class
  sub_index <- index[match(sub_ref$ProbeFileID, index$ProbeFileID),] 
  sub_sys <- sys[match(sub_ref$ProbeFileID, index$ProbeFileID),]

  # merge the ref csv with the index csv (indicated col names due to the duplicated col names between ref and index csv files)  
  idx_df <- merge(x=sub_ref, y=sub_index, by="ProbeFileID", all.x = TRUE)
  # merge the ref+index file with the system csv file
  m_df <- merge(x=idx_df, y=sub_sys, by="ProbeFileID", all.x = TRUE)
  
  f_df <- m_df
  #NOTE: f_df should be deleted after the validataion process
  #f_df <- m_df[!(m_df$ProbeMaskFileName=="" | is.na(m_df$ProbeMaskFileName)),]

  df <- scores_4_mask_pairs(nrow(f_df), f_df$ProbeMaskFileName, f_df$ProbeOutputMaskFileName, f_df$ProbeFileName, refDir, sysDir, erodeKernSize, dilateKernSize, isMaskOut, thres, metricOption, outputMaskPath)
  merged_df <-cbind(f_df, df)
  df_avg <- avg_scores_by_factors_SSD(merged_df, taskType, bySet, byPost)  
  return (list(DFILE = merged_df, DFAVG = df_avg))
}

### Create a CSV report for double source detection
# *Description: this function calls each metric function and 
#               return the metric value and the colored mask output as a report
# *Inputs
# *ref: reference dataframe
# *sys: system output dataframe
# *index: index dataframe
#   *refDir: reference mask file directory
#   *sysDir: system output mask file directory
#   *erodekernSize: Kernel size for Erosion
#   *dilatekernSize: Kernel size for Dilation
#   *isMaskOut: option for mask outputs [y/n]
#   *taskType: [manipulation, removal, splice]
#   *bySet: the reports will be seperated by DatasetID(e.g., y/n, default: n)
#   *byPost: the reports will be seperated by PostProcessingID(e.g., y/n, default: n)
#   *thres: threshold input[0,1] for grayscale mask
#   *metricOption: options [nmm, mcc, ham, wL1, hL1, and all]
# *Outputs
#   *report dataframe

createReportDSD <- function(ref, sys, index, refDir, sysDir, erodeKernSize, dilateKernSize, isMaskOut, taskType, bySet, byPost, thres, metricOption, outputMaskPath)
{
   
  if(FALSE)
  {
    #colnames(sys)<-c("ProbeFileID", "DonorFileID", "ConfidenceScore", "ProbeOutputMaskFileName", "DonorOutputMaskFileName")
    
    ref <- myRef # for testing
    #colnames(ref)<-c("TaskID", "ProbeFileID", "ProbeFileName", "ProbeMaskFileName", 
    #                 "DonorFileID", "DonorFileName", "DonorMaskFileName", "IsTarget", 
    #                 "ProbePostProcessed", "DonorPostProcessed", "ManipulationQuality", 
    #                 "IsManipulationTypeRemoval", "IsManipulationTypeSplice", "IsManipulationTypeCopyClone", 
    #                 "Collection", "BaseFileName", "Lighting", "IsControl", "CorrespondingControlFileName", "SemanticConsistency")
    sys <- mySys # for testing
    colnames(sys)<-c("ProbeFileID", "DonorFileID", "ConfidenceScore", "ProbeOutputMaskFileName", "DonorOutputMaskFileName")
    refDir <- myRefDir
    sysDir <- mySysDir
    index <- myIndex
    #colnames(index) <- c("TaskID", "ProbeFileID", "ProbeFileName", "ProbeWidth", "ProbeHeight",
    #                     "DonorFileID", "DonorFileName", "DonorWidth", "DonorHeight")
    erodeKernSize <- 11
    dilateKernSize <- 5
    isMaskOut <- "y"

    taskType <- "splice"
    bySet <- "y"
    byPost <- "y"
  }
   
  sub_ref <- ref[ref$IsTarget=="Y",] # grep only the target class
  index <- subset(index, select = c("ProbeFileID", "DonorFileID", "ProbeWidth", "ProbeHeight", "DonorWidth", "DonorHeight")) ##due to ProbeFileName/DonorFileName duplication
  
  # grep the scores where IsTarget is equal to "Y"
  sub_index <- index[match(paste(sub_ref$ProbeFileID, sub_ref$DonorFileID), paste(index$ProbeFileID, index$DonorFileID)),]
  sub_sys <- sys[match(paste(sub_ref$ProbeFileID, sub_ref$DonorFileID), paste(sys$ProbeFileID, sys$DonorFileID)),]
  
  # merge the ref csv with the index csv (indicated col names due to the duplicated col names between ref and index csv files)
  idx_df <- merge(x=sub_ref, y=sub_index, by=c("ProbeFileID", "DonorFileID"), all.x = TRUE)
  m_df <- merge(x=idx_df, y=sub_sys, by=c("ProbeFileID", "DonorFileID"), all.x = TRUE)
 
  
  f_df <- m_df
  #NOTE: f_df should be deleted after the validataion process
  #f_df <- m_df[!(m_df$ProbeMaskFileName=="" | is.na(m_df$ProbeMaskFileName) | m_df$DonorMaskFileName=="" | is.na(m_df$DonorMaskFileName)), ]

  probe_df <- scores_4_mask_pairs(nrow(f_df), f_df$ProbeMaskFileName, f_df$ProbeOutputMaskFileName, f_df$ProbeFileName, refDir, sysDir, erodeKernSize, dilateKernSize, isMaskOut, thres, metricOption, outputMaskPath)
  colnames(probe_df) <- c("pNMM", "pMCC", "pHAM", "pWL1", "pHL1", "ProbeColMaskFileName")
  
  # TODO: change this line to Donor after fixing the database
  #donor_df <- scores_4_mask_pairs(nrow(f_df), f_df$ProbeMaskFileName, f_df$ProbeOutputMaskFileName, f_df$ProbeFileName, refDir, sysDir, erodeKernSize, dilateKernSize, isMaskOut, thres) #for test
  donor_df <- scores_4_mask_pairs(nrow(f_df), f_df$DonorMaskFileName, f_df$DonorOutputMaskFileName, f_df$DonorFileName, refDir, sysDir, erodeKernSize, dilateKernSize, isMaskOut, thres, metricOption, outputMaskPath)
  colnames(donor_df) <- c("dNMM", "dMCC", "dHAM", "dWL1", "dHL1", "DonorColMaskFileName")
  
  pd_df <- cbind(probe_df, donor_df)
  merged_df <- cbind(f_df, pd_df)
  
  df_avg <- avg_scores_by_factors_DSD(merged_df, taskType, bySet, byPost)  
  return (list(DFILE = merged_df, DFAVG = df_avg))
}


########### packages ########################################################
suppressWarnings(suppressMessages(require(EBImage)))
suppressWarnings(suppressMessages(require(data.table)))
suppressWarnings(suppressMessages(require(useful)))
suppressWarnings(suppressMessages(require(optparse)))
#require (videoplayR)

########### Command line interface ########################################################

if(TRUE)
{
  #setwd('/Users/yunglee/YYL/MEDIFOR/MediScore/tools/MaskScorer')
  #setwd('/Users/yooyoung/Documents/NIST/MediScore/tools/MaskScorer')
  data_path <- "../../data"
  nc_path <- "NC2016_Test0601"
  refFname <- "reference/manipulation/NC2016-manipulation-ref.csv"
  indexFname <- "indexes/NC2016-manipulation-index.csv"
  #refFname <- "reference/remove/NC2016-removal-ref.csv"
  #indexFname <- "indexes/NC2016-removal-index.csv"
  #refFname <- "reference/splice/NC2016-splice-ref.csv"
  #indexFname <- "indexes/NC2016-splice-index.csv"
  
  #sysFname <- file.path(data_path, "SystemOutputs/copymove01_byLee_manipulation.csv") #this file name need to be changed 
  #sysFname <- file.path(data_path, "SystemOutputs/rajiv_splice.csv") #this file name need to be changed 
  #sysFname <- file.path(data_path, "SystemOutputs/splice_results_cleaned_up.csv") #this file name need to be changed 
  #sysFname <- file.path(data_path, "SystemOutputs/splice0601/results_cleaned_up.csv") #this file name need to be changed 
  #sysFname <- file.path(data_path, "SystemOutputs/splice0608/results.csv") #this file name need to be changed
  #sysFname <- file.path(data_path, "SystemOutputs/dct/dct_cleaned_up.csv")
  sysFname <- file.path(data_path, "SystemOutputs/dct0608/dct02.csv")
  
  if(FALSE)
  {

    #myRef <- read.table(file.path(data_path, nc_path, refFname), header = TRUE, row.names = NULL, sep="|")
    
    system.time(myRef <- fread(file.path(data_path, nc_path, refFname), sep="|", header=TRUE, 
                               colClasses=c("character","character","character","character","character",
                                            "character","character","character","character","character",
                                            "character","character","character","character","character",
                                            "character","character","character","character","character")))
    #system.time(myIndex <- read.table(file.path(data_path, nc_path,indexFname), header = TRUE, row.names = NULL, sep="|"))
    
    myIndex <- fread(file.path(data_path, nc_path, indexFname), sep="|", header=TRUE, 
                               colClasses=c("character","character","character","numeric","numeric",
                                            "character","character","numeric","numeric"))
    
    system.time(mySys <- read.table(sysFname, header = TRUE, row.names = NULL, sep="|"))
    
    mySys <- fread(file.path(data_path, nc_path, sysFname), sep="|", header=TRUE, 
                                 colClasses=c("character","character","numeric","character","character"))
    
    mySys <- fread(file.path(data_path, nc_path, sysFname), sep="|", header=TRUE, 
                   colClasses=c("character","numeric","character"))
    mySysDir <- dirname(sysFname)
    myRefDir <- file.path(data_path, nc_path)
  }
}

####### Command-line interface ##############

option_list = list(
  make_option(c("-i", "--inType"), type = "character", default = "file", 
              help = "Two different types of inputs: [file] and [DAPR]", metavar = "character"),
  
  # if "file"
  make_option(c("-t", "--task"), type = "character", default = "manipulation", 
              help = "Three different types of taks: [manipulation], [removal], and [splice]", metavar = "character"),
  
  make_option(c("-d", "--ncDir"), type="character", default="../../data/NC2016_Test0601", 
              help="NC2016_Test directory path: [e.g., ../../data/NC2016_Test]", metavar="character"),
  
  make_option(c("-r", "--inRef"), type="character", default=refFname, 
              help="Reference csv file name: [e.g., reference/manipulation/NC2016-manipulation-ref.csv]", metavar="character"),
  
  make_option(c("-s", "--inSys"), type="character", default=sysFname, 
              help="System output csv file name: [e.g., ~/expid/system_output.csv]", metavar="character"),
  
  make_option(c("-x", "--inIndex"), type="character", default=indexFname, 
              help="Task Index csv file name: [e.g., indexes/NC2016-manipulation-index.csv]", metavar="character"),
  
  make_option(c("-m", "--metric"), type="character", default = "nmm", 
              help="Metric option: [all], [nmm], [mcc], [ham], [wL1], and [hL1]", metavar="character"),
  
  make_option(c("-o", "--out"), type="character", default="report.csv", 
              help="Report output file name [default= %default], option format: CSV", metavar="character"),
  
  make_option("--eks", type="integer", default= 15, 
              help="Erosion kernel size number must be odd, [default= %default]", metavar="number"),
  
  make_option("--dks", type="integer", default= 9, 
              help="Dilation  kernel size number must be odd, [default= %default]", metavar="number"),
  
  make_option("--maskout", type="character", default= "y", 
              help="Error indicated (by color) mask, option: ['y'] or ['n'], [default= %default]", metavar="character"),
  
  make_option("--bySet", type="character", default= "n", 
              help="Evaluation by DatasetID [default= %default]", metavar="character"),
  
  make_option("--byPost", type="character", default= "n", 
              help="Evaluation by PostProcessing conditions [default= %default]", metavar="character"),
  
  make_option("--th", type="numeric", default= 0, 
              help="threshold value [0,1] for grayscale, [default= %default]", metavar="numeric"),
  
  make_option("--maskpath", type="character", default="maskoutputs", 
              help="Input the folder name for colored mask outputs [default= %default], option format: CSV", metavar="character")
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

if (is.null(opt$inType)){
  print_help(opt_parser)
  print("ERROR: Input file type (either [file] or [DAPR]) must be supplied.n", call.=FALSE)
  q(status=1)
}

if (!(opt$inType == "file" | opt$inType == "DAPR"))
{
  print_help(opt_parser)
  print("ERROR: Input file type must be either [file] or [DAPR]", call.=FALSE)
  q(status=1)
}

#################################################
## command-line arguments for "file"
#################################################

if(opt$inType == "file")
{
 
  if (is.null(opt$task)){
    print_help(opt_parser)
    print("ERROR: Task type must be supplied.n", call.=FALSE)
    q(status=1)
  }else
  {
    if(!(opt$task == "manipulation" | opt$task == "removal" | opt$task == "splice"))
    {
      print("ERROR: Task type should be one of [manipulation], [removal], and [splice]", call.=FALSE)
      q(status=1)
    }
  }
  
  if (is.null(opt$ncDir)){
    print_help(opt_parser)
    print("ERROR: NC2016_Test directory path must be supplied.n", call.=FALSE)
    q(status=1)
  }else
  {
    myRefDir <- file.path(opt$ncDir)
  }
  
  if (is.null(opt$inRef)){
    print_help(opt_parser)
    print("ERROR: Input file name for reference must be supplied.n", call.=FALSE)
    q(status=1)
  }else
  {
    #myRef <- read.csv(opt$inRef, header = TRUE, row.names = NULL)
    #myRef <- read.table(file.path(myRefDir, opt$inRef), header = TRUE, row.names = NULL, sep="|")
    myRef <- fread(file.path(myRefDir, opt$inRef), sep="|", header=TRUE, 
                   colClasses=c("character","character","character","character","character",
                                "character","character","character","character","character",
                                "character","character","character","character","character",
                                "character","character","character","character","character"))

    #print(nrow(myRef))
  }
  
  if (is.null(opt$inSys)){
    print_help(opt_parser)
    print("ERROR: Input file name for system output must be supplied.n", call.=FALSE)
    q(status=1)
  }else
  {
    #mySys <- read.csv(opt$inSys, header = TRUE, row.names = NULL)
    #mySys <- read.table(opt$inSys, header = TRUE, row.names = NULL, sep="|")
    mySysDir <- dirname(opt$inSys)
    
    if(TRUE)
    {
      if(opt$task == "manipulation" | opt$task == "removal")
      {
        mySys <- fread(file.path(opt$inSys), sep="|", header=TRUE, colClasses=c("character","numeric","character"))
       
      }
      else if(opt$task == "splice")
      {
        mySys <- fread(file.path(opt$inSys), sep="|", header=TRUE, 
                       colClasses=c("character","character","numeric","character","character"))
      }
    }
    
    #print(mySysDir)
    #print(nrow(mySys))
  }
  
  if (is.null(opt$inIndex)){
    print_help(opt_parser)
    print("ERROR: Input file name for index files  must be supplied.n", call.=FALSE)
    q(status=1)
  }else
  {
    #myIndex <- read.table(file.path(myRefDir, opt$inIndex), header = TRUE, row.names = NULL, sep="|")
    if(TRUE)
    {
      if(opt$task == "manipulation" | opt$task == "removal")
      {
        myIndex <- fread(file.path(myRefDir, opt$inIndex), sep="|", header=TRUE, 
                         colClasses=c("character","character","character","numeric","numeric"))
      }
      else if(opt$task == "splice")
      {
        myIndex <- fread(file.path(myRefDir, opt$inIndex), sep="|", header=TRUE, 
                                   colClasses=c("character","character","character","numeric","numeric",
                                                "character","character","numeric","numeric"))
      }
    }
    #print(nrow(myIndex))
  }


  if (is.null(opt$eks) | is.null(opt$dks)){
    print_help(opt_parser)
    print("ERROR: erosion and dilation kernel size  must be supplied.n", call.=FALSE)
    q(status=1)
  }
  
  if (is.null(opt$th) | opt$th < 0 | opt$th > 1){
    print_help(opt_parser)
    print("ERROR: Threshold for grayscale must be range [0,1].n", call.=FALSE)
    q(status=1)
  }
  
  #cat("eKernSize: ", opt$eks, "dKernSize: ", opt$dks, "\n")
  
 
  
  if (is.null(opt$maskout) | !(opt$maskout=="y"|opt$maskout=="n")){
    print_help(opt_parser)
    print("ERROR: y/n  must be supplied for maskout", call.=FALSE)
    q(status=1)
  }
  
  #create the folder and save the mask outputs
  #set.seed(1)
  #ranx <-stringi::stri_rand_strings(1, 10) #generate random number
  #output_path <- paste0("maskoutputs_",ranx)
  #if(!dir.exists(output_path))
  #  dir.create(file.path(output_path), FALSE)
  
  if (is.null(opt$maskpath)){
    print_help(opt_parser)
    print("ERROR: the folder name for mask ouputs must be supplied.n", call.=FALSE)
    q(status=1)
  }else
  {
    if(!dir.exists(opt$maskpath))
      dir.create(file.path(opt$maskpath), FALSE)
  }
  
  if(opt$task == "manipulation" | opt$task == "removal")
  {
    r_df <- createReportSSD(myRef, mySys, myIndex, myRefDir, mySysDir, opt$eks, opt$dks, 
                            opt$maskout, opt$task, opt$bySet, opt$byPost, opt$th, opt$metric, opt$maskpath) # default 15, 9
  }
  else if(opt$task == "splice")
  {
    r_df <- createReportDSD(myRef, mySys, myIndex, myRefDir, mySysDir, opt$eks, opt$dks, 
                            opt$maskout, opt$task, opt$bySet, opt$byPost, opt$th, opt$metric, opt$maskpath) # default 15, 9
  }
  
  a_df <- r_df$DFAVG
  
  print("Starting a report ...")
  
  if (!(opt$metric == "all" | opt$metric == "nmm"| opt$metric == "mcc"| opt$metric == "ham"| opt$metric == "wL1"| opt$metric == "hL1"))
  {
    print_help(opt_parser)
    print("ERROR: Input metric type must be one of [all], [nmm], [mcc], [ham], [wL1], and [hL1]", call.=FALSE)
    q(status=1)
  }
  
  if(opt$task == "manipulation" | opt$task == "removal")
  {
    if(opt$metric =="all")
    {
      cat("Avg NMM: ", a_df$NMM,", Avg MCC: ", a_df$MCC,", Avg HAM: ", a_df$HAM, ", Avg WL1: ", a_df$WL1,", Avg HL1: ", a_df$HL1,"\n")
      
    }else if(opt$metric =="nmm")
    {
      cat("Avg NMM: ", a_df$NMM, "\n")
      
    }else if(opt$metric =="mcc")
    {
      cat("Avg MCC: ", a_df$MCC, "\n")
      
    }else if(opt$metric =="HAM")
    {
      cat("Avg HAM: ", a_df$HAM, "\n")
      
    }else if(opt$metric =="wL1")
    {
      cat("Avg WL1: ", a_df$WL1, "\n")
      
    }else if(opt$metric =="hL1")
    {
      cat("Avg HL1: ", a_df$HL1, "\n")
    }
  }
  
  if(opt$task == "splice")
  {
    if(opt$metric =="all")
    {
      #cat("NMM (mean): ", mean(a_df$pNMM, a_df$dNMM),", MCC (mean): ", mean(a_df$pMCC, a_df$dMCC),", HAM (mean): ", mean(a_df$pHAM, a_df$dHAM), 
      #    ", WL1 (mean): ", mean(a_df$pWL1, a_df$dWL1),", HL1 (mean): ", mean(a_df$pHL1,a_df$dHL1),"\n") # got trim error here
      cat("  Avg pNMM: ", a_df$pNMM, "\n Avg pMCC: ", a_df$pMCC,"\n Avg pHAM: ", a_df$pHAM, "\n Avg pWL1: ", a_df$pWL1, "\n Avg pHL1: ", a_df$pHL1,"\n")
      cat("  Avg dNMM: ", a_df$dNMM, "\n Avg dMCC: ", a_df$dMCC,"\n Avg dHAM: ", a_df$dHAM, "\n Avg dWL1: ", a_df$dWL1, "\n Avg dHL1: ", a_df$dHL1,"\n")
    }else if(opt$metric =="nmm")
    {
      cat("Avg pNMM: ", a_df$pNMM, "Avg dNMM: ", a_df$dNMM, "\n")
      
    }else if(opt$metric =="mcc")
    {
      cat("Avg pMCC: ", a_df$pMCC, "Avg dMCC: ", a_df$dMCC, "\n")
      
    }else if(opt$metric =="HAM")
    {
      cat("Avg pHAM: ", a_df$pHAM, "Avg dHAM: ", a_df$dHAM,"\n")
      
    }else if(opt$metric =="wL1")
    {
      cat("Avg pWL1: ", a_df$pWL1, "Avg dWL1: ", a_df$dWL1,"\n")
      
    }else if(opt$metric =="hL1")
    {
      cat("Avg pHL1: ", a_df$pHL1, "Avg dHL1: ", a_df$dHL1, "\n")
      
    }
  }
  
  
  if (is.null(opt$out)){
    print_help(opt_parser)
    print("ERROR: Output file name for the report must be supplied.n", call.=FALSE)
    q(status=1)
  }else
  {
    write.csv(r_df$DFILE, file=opt$out, row.names=F)
    dirname <-dirname(opt$out)
    newname <-basename((unlist(opt$out))) #
    extname <- strsplit(newname, '[.]')[[1]][1]
    #write.csv(r_df$DFAVG, file=paste0("avg_",opt$out), row.names=F)
    write.csv(r_df$DFAVG, file=file.path(dirname, paste0(extname, "_avg.csv")), row.names=F)
  }
  
}


##TODO
if(FALSE)
{
  avg_scores_by_factors_todo <- function(df, factors)
  {
    
    factors <- 'TaskID|ImageCollection|ProbePostProcessed'
    
    #df <- f_df
    
    if(is.null(factors))
    {
      sub.d <- df
      df_avg$runID <- 1
      df_avg$pNMM <- round(mean(sub.d$pNMM), digits = mydigits)
      df_avg$pMCC <- round(mean(sub.d$pMCC), digits = mydigits)
      df_avg$pHAM <- round(mean(sub.d$pHAM), digits = mydigits)
      df_avg$pWL1 <- round(mean(sub.d$pWL1), digits = mydigits)
      df_avg$pHL1 <- round(mean(sub.d$pHL1), digits = mydigits)
    }
    else
    {
      arr_f <- unlist(strsplit(factors, "[|]"))
      list_f <- list()
      runs = 1
      for (i in 1: length(arr_f))
      {
        list_f[[i]] <- unique(na.omit(df[,arr_f[i]]))
        print(list_f[[i]])
        runs = runs * length(list_f[[i]])
      }
      
      df_avg<-data.frame("runID"=rep(NA, runs))
      
      idx <- 1         
      
      for(i in 1: length(arr_f))
      {
        for(j in 1: length(list_f[[i]]))
        {
          sub.d <- subset(df, grepl(as.character(list_f[[i]][j]), arr_f[i]))
          print(list_f[[i]][j])
          if(length(list_f[[i]]) > 1)
          {
            df_avg$runID[idx] <- idx
            
            df_avg$pNMM[idx] <- round(mean(sub.d$pNMM), digits = mydigits)
            df_avg$pMCC[idx] <- round(mean(sub.d$pMCC), digits = mydigits)
            df_avg$pHAM[idx] <- round(mean(sub.d$pHAM), digits = mydigits)
            df_avg$pWL1[idx] <- round(mean(sub.d$pWL1), digits = mydigits)
            df_avg$pHL1[idx] <- round(mean(sub.d$pHL1), digits = mydigits)
            df_avg[idx,arr_f[i]] <- as.character(list_f[[i]][j])
            idx = idx + 1
          }
        }
      }
    }
    
    write.csv(df_avg, file=paste0(list_f[[i]][j], "-test.csv"), row.names=F)
  }
}

