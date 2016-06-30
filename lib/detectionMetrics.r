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

# *Description: this function calculates the area under ROC curve
# *Inputs
# *score: system output (e.g., confidence score)
# *gt: ground-truth class labels (e.g., Y and N)
#
# *Outputs
# *fpr: false positive rate
# *tpr: true positive rate
myroc <- function(score, gt)
{
  score.srt <- sort(score, decreasing = TRUE, index.return = TRUE)
  val <- unlist(score.srt$x)
  idx <- unlist(score.srt$ix)

  binary = gt[idx];

  #accounts for (rare) case where the next group of scores is equal.
  total <- length(binary)
  fp <- rep(0,times = total)
  tp <- rep(0,times = total)


  if(TRUE)
  {
    #loop over indices from 1 to total
    for (i in 1:total) { # TODO: change this to apply() later for speed
      #counts
      fp[i] <- sum(binary[val >= val[i]] == "N")
      tp[i] <- sum(binary[val >= val[i]] == "Y")
      #cat("FOR LOOP: FP: ", fp[i], "TP:", tp[i] , "\n")
    }
  }

  if(FALSE)
  {
    lapply(1:total, function(i){
      fp[i] <- sum(binary[val >= val[i]] == "N")
      tp[i] <- sum(binary[val >= val[i]] == "Y")})

  }

  sumN <- sum(binary == "N")
  sumY <- sum(binary== "Y")
  #divide at the very end to get the rate
  fpr <- fp/sumN;
  tpr <- tp/sumY;

  return(list(fpr = fpr, tpr = tpr))
}

# *Description: this function calculates the area under ROC curve
# *Inputs
# *score: system output (e.g., confidence score)
# *gt: ground-truth class labels (e.g., Y and N)
# *n : number of thresholds
#
# *Outputs
# *fpr: false positive rate
# *tpr: true positive rate
myroc_new <- function(score, gt, n=100)
{

  tpr <- function(score, gt, threshold)
  {
    sum(score >= threshold & gt == "Y") / sum(gt == "Y")
  }

  fpr <- function(score, gt, threshold) {
    sum(score >= threshold & gt == "N") / sum(gt == "N")
  }

  score.srt <- sort(score, decreasing = TRUE, index.return = TRUE)
  #val <- norm0to1(unlist(score.srt$x))
  val <- unlist(score.srt$x)
  idx <- unlist(score.srt$ix)

  binary = gt[idx];


  # Use unique values of scores as a threshod
  # ("n" can be used to divide threshold values as well if needs)
  threshold = unique(val)
  tpr <- sapply(threshold, function(th) tpr(val, binary, th))
  fpr <- sapply(threshold, function(th) fpr(val, binary, th))

  return(list(fpr = fpr, tpr = tpr))
}

# *Description: this function calculates ROC (Receiver Operating Characteristic) curve points
#               and returns fpr (false positive rate) and tpr (true positive rate)
#
# *Inputs
# *fpr: false positive rate for x-axis
# *tpr: true positive rate for y-axis
#
# *Outputs
# *AUC (Area under the ROC curve) value

myauc <-function(fpr, tpr)
{
  height = (tpr[-1] + tpr[-length(tpr)]) / 2
  width = diff(fpr)
  auc <- sum(height * width)
  #print(round(auc, digits =3))
  return (auc)
}

# *Description: this function calculates the  partial areaunder the ROC curve
#               in agiven fpr value (e.g, fpr_stop =0.5)
#
# *Inputs
# *fpr: false positive rate for x-axis
# *tpr: true positive rate for y-axis
# *fpr_stop: range 0 to 1
#
# *Outputs
# *Partial AUC value
myauc.partial <-function(fpr, tpr, fpr_stop)
{
  #fpr_stop <- 0.5
  height = (tpr[-1] + tpr[-length(tpr)]) / 2
  width = diff(fpr[fpr <= fpr_stop])
  p.height <- height[1:length(width)]
  partial.auc <- sum(p.height * width)
  #print(round(auc, digits =3))
  return (partial.auc)
}

# *Description: this function calculates the point at which both fpr and fnr are equal
#
# *Inputs
# *fpr: false positive rate
# *fnr: false negative rate
#
# *Outputs
# *EER (Equal Error Rate) value

myeer <- function(fpr, fnr)
{
  idx <-which.min(abs(fnr-fpr))
  eer <-mean(c(fpr[idx], fnr[idx]))
  #print(round(eer, digits =3))
  return (eer)
}
