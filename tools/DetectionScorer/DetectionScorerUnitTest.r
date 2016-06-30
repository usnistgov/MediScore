### Unit Test
# * please refer the unittest markdown file (TODO)
# *File: SSDDetectionScorerUnitTest.r
# *Date: 5/25/2016
# *Author: Daniel Zhou
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

# *Description: this function calculates the area under ROC curve
# *Inputs
# *none
#
# *Outputs
# *0 if test is a success. Otherwise throws an error.

lib_path <- "../../lib"
source(file.path(lib_path, "detectionMetrics.r"))

unit.test <- function()
{
  cat("Beginning unit test for detection scorer...\n\n")
# Daniel: this need to be changed to "jsonlite"
  suppressWarnings(suppressMessages(require("jsonlite")))
  suppressWarnings(suppressMessages(require("RUnit")))
  
  #assign fake data directly
  d1 <- c('F18.jpg',0.034536792,"Y",
          'F166.jpg',0.020949942,"N",
          'F86.jpg',0.016464296,"Y",
          'F172.jpg',0.014902585,"N")
  dim(d1) <- c(3,length(d1)/3)
  d1 <- t(d1)
  df1 <- data.frame(d1)
  names(df1) <- c("fname","score","gt")
  df1$fname <- as.character(df1$fname)
  df1$score <- as.numeric(as.character(df1$score))
  df1$gt <- as.character(df1$gt)

  df1fpr<-c(0,0.5,0.5,1)
  df1tpr<-c(0.5,0.5,1,1)
  drocs <- myroc(df1$score,df1$gt)

  checkEquals(drocs$fpr,df1fpr)
  checkEquals(drocs$tpr,df1tpr)

  dauc <- 0.75
  checkEquals(myauc(drocs$fpr,drocs$tpr),dauc)

  #check partial AUC for following thresholds
  threshold_list <- c(0.1,0.3,0.5,0.7,0.9)
  partial_aucs <- c(0,0,0.25,0.25,0.25)

  total_tc <- length(threshold_list)
  for (j in 1:total_tc) {
    checkEquals(myauc.partial(df1fpr,df1tpr,threshold_list[j]),partial_aucs[j])
  }

  df1eer <- 0.5
  checkEquals(myeer(df1fpr,1-df1tpr),df1eer)

  #fake data #2. Contains duplicate scores.
  d2 <- c('F18.jpg',0.034536792,"Y",
          'F166.jpg',0.020949942,"N",
          'F165.jpg',0.020949942,"N",
          'F86.jpg',0.016464296,"Y",
          'F87.jpg',0.016464296,"N",
          'F88.jpg',0.016464296,"Y",
          'F172.jpg',0.014902585,"N")
  dim(d2) <- c(3,length(d2)/3)
  d2 <- t(d2)
  df2 <- data.frame(d2)
  names(df2) <- c("fname","score","gt")
  df2$fname <- as.character(df2$fname)
  df2$score <- as.numeric(as.character(df2$score))
  df2$gt <- as.character(df2$gt)
  drocs2 <- myroc(df2$score,df2$gt)

  df2fpr<-c(0,0.5,0.5,0.75,0.75,0.75,1)
  df2tpr<-c(1/3,1/3,1/3,1,1,1,1)

  checkEquals(drocs2$fpr,df2fpr)
  checkEquals(drocs2$tpr,df2tpr)

  #TEST myroc_new
  drocs2 <- myroc_new(df2$score,df2$gt)
  df2fpr<-c(0,0.5,0.75,1)
  df2tpr<-c(1/3,1/3,1,1)
  checkEquals(drocs2$fpr,df2fpr)
  checkEquals(drocs2$tpr,df2tpr)

  dauc2 <- 7/12
  checkEquals(myauc(df2fpr,df2tpr),dauc2)

  #check partial AUC for following thresholds
  threshold_list <- c(0.2,0.4,0.6,0.8,1)
  partial_aucs <- c(0,0,1/6,1/3,7/12)
  total_tc <- length(threshold_list)
  for (j in 1:total_tc) {
    checkEquals(myauc.partial(df2fpr,df2tpr,threshold_list[j]),partial_aucs[j])
  }

  df1eer <- 7/12
  checkEquals(myeer(df2fpr,1-df2tpr),df1eer)
  cat("All detection scorer unit test successfully complete.\n\n")
  quit(status=0)
}

unit.test()
