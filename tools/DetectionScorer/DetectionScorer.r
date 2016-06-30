#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) 
{
  print("usage: at lemast one argument must be supplied(see --help)")
  q(status=1)
}


# *File: DetectionScorer.r
# *Date: 5/20/2016
# *Author: Yooyoung Lee
# *Status: In progress
#
# *Description: this code calculates performance measures (for ROC, AUC, and EER) 
# on system outputs (confidence scores) and return the report table
#
# *Requirement: This code requires the following packages:
#    - require(data.table)
#    - require(optparse)
#    - require(jsonlite)
#    - require(scales)
#    - require(ggplot2)
#    - require(RUnit)
#    - require(RMySQL)
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
##########################################################################

# loading libraries
lib_path <- "../../lib"
source(file.path(lib_path, "detectionMetrics.r"))
source(file.path(lib_path, "json2df.r"))
source(file.path(lib_path, "plots.r"))

### Create a CSV report for Single Signal detection (SSD) score metrics
# *Description: this function calls each metric fucntion and return the metric value as a report
# *Inputs
# *ref: reference dataframe
# *sys: system output dataframe
# *index: index dataframe
# *byTask: the reports will be seperated by TaskID(e.g., y/n, default: n)
# *bySet: the reports will be seperated by DatasetID(e.g., y/n, default: n)
# *byPost: the reports will be seperated by PostProcessingID(e.g., y/n, default: n)
# *Outputs
# * report dataframe

createReportSSD <- function(ref, sys, index, taskType, bySet, byPost)
{
  if(FALSE)
  {
    ref <- myRef # for testing
    sys <- mySys # for testing
    index <- myIndex
    #colnames(index) <- c("TaskID", "ProbeFileID", "ProbeFileName", "ProbeWidth", "ProbeHeight")
    #colnames(index) <- c("TaskID", "ProbeFileID", "ProbeFileName", "ProbeWidth", "ProbeHeight",
    #                     "DonorFileID", "DonorFileName", "DonorWidth", "DonorHeight")
    #runid <- "0" # for testing
    bySet <- "y"
    byPost <- "y"
    
    cat("Which task type? ", taskType, "by dataset? ", bySet, "by post-processing? ", byPost, "\n")
  }

  # covert the ProbeFileID type to a character
  index <- subset(index, select =c("ProbeFileID","ProbeWidth", "ProbeHeight"))
  
  if(FALSE)
  {
    #m_df <- merge(x=sub_ref, y=sub_sys, by=c("ProbeFileID", "DonorFileID"))
    idx_df <- merge(x=ref, y=index, by="ProbeFileID", all.x = TRUE)
    # validator - add an error message for the number of rows (nrows(idx_df) != nrows(sys))
    m_df <- merge(x=idx_df, y=sys, by="ProbeFileID", all.x = TRUE)
  }
    
  ref_dt = data.table(ref, key="ProbeFileID")
  index_dt = data.table(index, key="ProbeFileID")
  sys_dt = data.table(sys, key="ProbeFileID")

  idx_dt <- merge(ref_dt, index_dt, all.x=TRUE)
  m_df <- merge(idx_dt, sys_dt, all.x=TRUE)

  #NOTE: the filtering is not necessary after the validation process
  #f_df <- m_df[!(m_df$ConfidenceScore=="" | is.na(m_df$ConfidenceScore) | is.null(m_df$ConfidenceScore)),] 
  
  m_df$ConfidenceScore[is.na(m_df$ConfidenceScore)] <- 0
  m_df$ConfidenceScore[(m_df$ConfidenceScore=="")] <- 0
  f_df <- m_df
  
  #Test
  #f_df[f_df$ImageCollection =="Nimble-WEB" & f_df$ProbePostProcessed == "rescale" 
  #     & f_df$DonorPostProcessed == "rescale", c("ConfidenceScore", "IsTarget")]
  
  set.id <- "*"
  post.id <- "*" 
  
  # get unique settings for each factor
  if(bySet == "y")
  {
    set.id <- unique(f_df$Collection)
  }  
  
  if(byPost == "y")
  {
    post.id <- unique(f_df$ProbePostProcessed)
  }
  
  # calculate the number of rows for factors
  num.runs <- length(set.id) * length(post.id)
  
  df<-data.frame("runID"=rep(NA, num.runs), "taskID"=rep(taskType, num.runs), "Collection"=rep(NA, num.runs),"PostProcessed"=rep(NA, num.runs),
                 "AUC"=rep(NA, num.runs), "partialAUC"=rep(NA, num.runs), "EER"=rep(NA, num.runs), "ROCpoints"=rep(NA, num.runs))
  
  myDigits <- 4 # round value condition
  fpr_stop <- 0.9 # for partial AUC
  idx <- 1
  
  sub.d <- f_df # use entire data if factors are not required
  #print(nrow(sub.d))

  #system.time(
  for (j in 1:length(set.id))
  {
    for (k in 1:length(post.id)) 
    {
      # sample subset based on the indicated factors (Collection, PostProcessed)
      sub.d <- subset(f_df, grepl(set.id[j], Collection) & grepl(post.id[k], ProbePostProcessed))
      
      cat(idx, " - Length of sub.df: ", nrow(sub.d), "\n")
      
      # calculate ROC points
      #roclist <- myroc(sub.d$ConfidenceScore, sub.d$IsTarget)
      roclist <- myroc_new(sub.d$ConfidenceScore, sub.d$IsTarget, 100)
      pt.tpr <- unlist(roclist$tpr)
      pt.fpr <- unlist(roclist$fpr)
      pt.fnr <- 1.0 - pt.tpr
      
      df$runID[idx] <- idx
      df$Collection[idx] <- as.character(set.id[j])
      df$PostProcessed[idx] <- as.character(post.id[k])
      df$AUC[idx] <- round(myauc(pt.fpr, pt.tpr), digits = myDigits)
      df$partialAUC[idx] <- round(myauc.partial(pt.fpr, pt.tpr, fpr_stop), digits = myDigits)
      df$EER[idx] <- round(myeer(pt.fpr, pt.fnr), digits = myDigits) #faster 
      df$ROCpoints[idx] <- toJSON(roclist, digits = 4, flatten = TRUE) # NOTE: Total number of characters in cell that can contain 32,767 characters
      idx <- idx + 1
    }
  }
  # )
  
  return (df)
}

### Create a CSV report for Double Signal detection (DSD) score metrics
# *Description: this function calls each metric fucntion and return the metric value as a report
# *Inputs
# *ref: reference dataframe
# *sys: system output dataframe
# *index: index dataframe
# *byTask: the reports will be seperated by TaskID(e.g., y/n, default: n)
# *bySet: the reports will be seperated by DatasetID(e.g., y/n, default: n)
# *byPost: the reports will be seperated by PostProcessingID(e.g., y/n, default: n)
# *Outputs
# * report dataframe

createReportDSD <- function(ref, sys, index, taskType, bySet, byPost)
{

  if(FALSE)
  {
    ref <- myRef # for testing
    #colnames(ref)<-c("TaskID", "ProbeFileID", "ProbeFileName", "ProbeMaskFileName", 
    #                 "DonorFileID", "DonorFileName", "DonorMaskFileName", "IsTarget", 
    #                 "ProbePostProcessed", "DonorPostProcessed", "ManipulationQuality", 
    #                 "IsManipulationTypeRemoval", "IsManipulationTypeSplice", "IsManipulationTypeCopyClone", 
    #                 "Collection", "BaseFileName", "Lighting", "IsControl", "CorrespondingControlFileName", "SemanticConsistency")
    
    sys <- mySys # for testing
    index <- myIndex
    #colnames(index) <- c("TaskID", "ProbeFileID", "ProbeFileName", "ProbeWidth", "ProbeHeight",
    #                     "DonorFileID", "DonorFileName", "DonorWidth", "DonorHeight")
    #runid <- "0" # for testing
    #taskType <- "splice"
    bySet <- "y"
    byPost <- "y"
    cat("Which task type? ", taskType, "by dataset? ", bySet, "by post-processing? ", byPost, "\n")
  }
  
 
  # to avoid duplicated fields (e.g., ProbeFileName)
  index <- subset(index, select = c("ProbeFileID", "DonorFileID", "ProbeWidth", "ProbeHeight", "DonorWidth", "DonorHeight")) 
  
  if(FALSE) #slower
  {
    system.time(idx_df <- merge(x=ref, y=index, by=c("ProbeFileID", "DonorFileID"), all.x = TRUE))
    # merge the system output and reference file using both ProbeFileID and DonorFileID
    system.time(m_df <- merge(x=idx_df, y=sys, by=c("ProbeFileID", "DonorFileID"), all.x = TRUE))
  }
  
  # Merging in this way is much faster
  ref_dt = data.table(ref, key=c("ProbeFileID", "DonorFileID"))
  index_dt = data.table(index, key=c("ProbeFileID", "DonorFileID"))
  sys_dt = data.table(sys, key=c("ProbeFileID", "DonorFileID"))
    
  idx_dt <- merge(ref_dt, index_dt, all.x=TRUE)
  m_df <- merge(idx_dt, sys_dt, all.x=TRUE)
  
  #NOTE: the filering process is not necessary after the validation process
  m_df$ConfidenceScore[(m_df$ConfidenceScore=="" | is.na(m_df$ConfidenceScore))] <- 0
  f_df <- m_df
  #f_df <- m_df[!(m_df$ConfidenceScore=="" | is.na(m_df$ConfidenceScore)),] 
  
  #Test
  #f_df[f_df$ImageCollection =="Nimble-WEB" & f_df$ProbePostProcessed == "rescale" 
  #     & f_df$DonorPostProcessed == "rescale", c("ConfidenceScore", "IsTarget")]
 
  set.id <- "*"
  post.id <- "*"

  if(bySet == "y")
  {
    set.id <- unique(f_df$Collection)
  }  
  
  if(byPost == "y") # get unique settings using both Probe/Donor PostProcessed
  {
    post.id <- unique(paste(f_df$ProbePostProcessed, f_df$DonorPostProcessed))
    #test <- subset(m_df, grepl(post.id[1], paste(ProbePostProcessed, DonorPostProcessed)))
  }
  
  # calculate the number of rows for factors
  num.runs <- length(set.id) * length(post.id)
  
  df<-data.frame("runID"=rep(NA, num.runs), "taskID"=rep(taskType, num.runs), "Collection"=rep(NA, num.runs),"PostProcessed"=rep(NA, num.runs),
                 "AUC"=rep(NA, num.runs), "partialAUC"=rep(NA, num.runs), "EER"=rep(NA, num.runs), "ROCpoints"=rep(NA, num.runs))
  
  myDigits <- 4 # round value condition
  fpr_stop <- 0.9 # for partial AUC
  idx <- 1  
  sub.d <- f_df
 
  #print(nrow(sub.d))
  #system.time(
  for (j in 1:length(set.id))
  {
    for (k in 1:length(post.id)) 
    {
      
      # for splice task, grab subsets based on Collection and both Probe/Donor PostProcessed
      sub.d <- subset(f_df, grepl(set.id[j], Collection) & grepl(post.id[k], paste(ProbePostProcessed, DonorPostProcessed)))
      
      cat(idx, " - Length of sub.df: ", nrow(sub.d), "\n")
      
      #roclist <- myroc(sub.d$ConfidenceScore, sub.d$IsTarget)
      roclist <- myroc_new(sub.d$ConfidenceScore, sub.d$IsTarget, 100) # more efficient
      pt.tpr <- unlist(roclist$tpr)
      pt.fpr <- unlist(roclist$fpr)
      pt.fnr <- 1.0 - pt.tpr
      
      df$runID[idx] <- idx
      df$Collection[idx] <- as.character(set.id[j])
      df$PostProcessed[idx] <- as.character(post.id[k])
      df$AUC[idx] <- round(myauc(pt.fpr, pt.tpr), digits = myDigits)
      df$partialAUC[idx] <- round(myauc.partial(pt.fpr, pt.tpr, fpr_stop), digits = myDigits)
      df$EER[idx] <- round(myeer(pt.fpr, pt.fnr), digits = myDigits)
      df$ROCpoints[idx] <- toJSON(roclist, digits = 4, flatten = TRUE) #Note: Total number of characters in cell that can contain 32,767 characters
      idx <- idx + 1
    }
  }
 # )

  return (df)
}

####### Test CSV files ############################

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
  #sysFname <- file.path(data_path, "SystemOutputs/results_cleaned_up.csv") #this file name need to be changed 
  #sysFname <- file.path(data_path, "SystemOutputs/splice0601/results_cleaned_up.csv") #this file name need to be changed 
  #sysFname <- file.path(data_path, "SystemOutputs/dct01_v3.csv")
  #sysFname <- file.path(data_path, "SystemOutputs/copymove01.csv")
  #sysFname <- file.path(data_path, "SystemOutputs/splice0608/results.csv")
  sysFname <- file.path(data_path, "SystemOutputs/dct0608/dct02.csv")
  
  if(FALSE) # for testing
  {
    #sysFname <- file.path(data_path, "SystemOutputs/Metadata0519.csv")
    #refFname <- file.path(data_path, nc_path, "reference/manipulation/NC2016-manipulation-ref.csv")
    system.time(myRef1 <- read.table(file.path(data_path, nc_path, refFname), header = TRUE, row.names = NULL, sep="|",
                                    stringsAsFactors=FALSE, comment.char=""))
    system.time(myRef2 <- read.table(file.path(data_path, nc_path, refFname), header = TRUE, row.names = NULL, sep="|",
                                     colClasses=c("character","character","character","character","character",
                                                  "character","character","character","character","character",
                                                  "character","character","character","character","character",
                                                  "character","character","character","character","character")))
    require(data.table)
    system.time(myRef <- fread(file.path(data_path, nc_path, refFname), sep="|", header=TRUE, 
                                colClasses=c("character","character","character","character","character",
                                             "character","character","character","character","character",
                                             "character","character","character","character","character",
                                             "character","character","character","character","character")))
    myIndex <- read.table(file.path(data_path, nc_path,indexFname), header = TRUE, row.names = NULL, sep="|", quote="", stringsAsFactors=FALSE, comment.char="")
    mySys <- read.table(sysFname, header = TRUE, row.names = NULL, sep="|", stringsAsFactors=FALSE, comment.char="")
    mySysDir <- dirname(sysFname)
    myRefDir <- file.path(data_path, nc_path)
  }
}


####### Loading packages ############################
suppressWarnings(suppressMessages(require(jsonlite)))
suppressWarnings(suppressMessages(require(data.table))) # for boosting speed
suppressWarnings(suppressMessages(require(optparse)))

####### Command-line interface ############################

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
  
  make_option(c("-m", "--metric"), type="character", default = "all", 
              help="Metric option: [all], [auc], [pauc], [eer], and [roc]", metavar="character"),
  
  make_option("--bySet", type="character", default= "n", 
              help="Evaluation by DatasetID [default= %default]", metavar="character"),
  
  make_option("--byPost", type="character", default= "n", 
              help="Evaluation by PostProcessing conditions [default= %default]", metavar="character"),
  
  make_option(c("-o", "--out"), type="character", default="report.csv", 
              help="Report output file name [default= %default], option format: CSV", metavar="character"),
  
  make_option("--whichplot", type="character", default= "roc", 
              help="plot options: ['det'] and ['roc'], [default= %default]", metavar="character"),
  
  make_option("--title", type="character", default= "", 
              help="Input plot title, [default= %default]", metavar="character"),
  
  make_option("--size", type="numeric", default= 1, 
              help="Input plot line size[ < 5.0], [default= %default]", metavar="character"),
  
  make_option(c("-l", "--linetype"), type="character", default="solid", 
              help="Plot curve line options ['solid'], ['dotted'], ['dashed'] and ['longdash'], [default= %default]", metavar="character"),

  make_option(c("-p", "--plot"), type="character", default="plot.pdf", 
              help="DET or ROC plot output file name [default= %default], option format:PDF", metavar="character")
  
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

if (is.null(opt$inType)){
  print_help(opt_parser)
  print("Error: Input file type (either [file] or [DAPR]) must be supplied.n", call.=FALSE)
  q(status=1)
}

if (!(opt$inType == "file" | opt$inType == "DAPR"))
{
  print_help(opt_parser)
  print("Error: Input file type must be either [file] or [DAPR]", call.=FALSE)
  q(status=1)
}

#######################################################################
## command-line arguments for "file"
#######################################################################

if(opt$inType == "file")
{  
  
  if (is.null(opt$task)){
    print_help(opt_parser)
    print("Error: Task type must be supplied.n", call.=FALSE)
    q(status=1)
  }else
  {
    if(!(opt$task == "manipulation" | opt$task == "removal" | opt$task == "splice"))
    {
      print("Error: Task type should be one of [manipulation], [removal], and [splice]", call.=FALSE)
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
    print("Error: Input file name for reference must be supplied.n", call.=FALSE)
    q(status=1)
  }else
  {
    # slower
    #myRef <- read.table(file.path(myRefDir, opt$inRef), header = TRUE, row.names = NULL, sep="|", quote="", stringsAsFactors=FALSE, comment.char="")
    # much faster
    myRef <- fread(file.path(myRefDir, opt$inRef), sep="|", header=TRUE, 
            colClasses=c("character","character","character","character","character",
                         "character","character","character","character","character",
                         "character","character","character","character","character",
                         "character","character","character","character","character"))

    #print(nrow(myRef))
  }
  
  if (is.null(opt$inSys)){
    print_help(opt_parser)
    print("Error: Input file name for system output must be supplied.n", call.=FALSE)
    q(status=1)
  }else
  {
    #mySys <- read.table(opt$inSys, header = TRUE, row.names = NULL, sep="|", stringsAsFactors=FALSE, comment.char="")
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
    print("Input file name for index files  must be supplied.n", call.=FALSE)
    q(status=1)
  }else
  {
    #myIndex <- read.table(file.path(myRefDir, opt$inIndex), header = TRUE, row.names = NULL, sep="|", stringsAsFactors=FALSE, comment.char="")
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

  
  #r_df <- createReport(myRef, mySys, opt$byTask, opt$bySet, opt$byPost)
  # call different report function depending on the task type
  if(opt$task == "manipulation" | opt$task == "removal")
  {
    #add index file here
    r_df <- createReportSSD(myRef, mySys, myIndex, opt$task, opt$bySet, opt$byPost)
  }
  
  if(opt$task == "splice")
  {
    r_df <- createReportDSD(myRef, mySys, myIndex, opt$task, opt$bySet, opt$byPost) # default 15, 9
  }
    
  cat("Starting a report: \n")
  if (!(opt$metric == "all" | opt$metric == "auc"| opt$metric == "pauc"| opt$metric == "eer"| opt$metric == "roc"))
  {
    print_help(opt_parser)
    print("Error: Input metric type must be one of [all], [auc], [pauc], [eer], and [roc]", call.=FALSE)
    q(status=1)
  }
  
  if(opt$metric =="all")
  {
    cat("AUC: ", r_df$AUC,", partialAUC: ", r_df$partialAUC,", EER: ", r_df$EER, "\n", "ROCpoints: ", r_df$ROCpoints,"\n")
  }else if(opt$metric =="auc")
  {
    cat("AUC: ", r_df$AUC, "\n")
    
  }else if(opt$metric =="pauc")
  {
    cat("Partial AUC: ", r_df$partialAUC, "\n")
    
  }else if(opt$metric =="eer")
  {
    cat("EER: ", r_df$EER, "\n")
  }else if(opt$metric =="roc")
  {
    cat("ROC points: ", r_df$ROCpoints, "\n")
  }
  
  if (is.null(opt$out)){
    print_help(opt_parser)
    print("Error: Output file name for the report must be supplied.n", call.=FALSE)
    q(status=1)
  }else
  {
    write.csv(r_df, file=opt$out, row.names=F)
    #write.table(r_df, file=paste0(datapath,"/Reports/",opt$out), row.names=F, sep="|")
  }

  
  if (is.null(opt$linetype) | (!(opt$linetype == "solid" | opt$linetype == "dotted" | opt$linetype == "dashed"| opt$linetype == "longdash"))){
    print_help(opt_parser)
    print("Error: The linetype for plot curve must be [solid], [dotted], and [dashed]", call.=FALSE)
    q(status=1)
  }
  
  if (is.null(opt$size) | opt$size > 5){
    print_help(opt_parser)
    print("Error: The line size for plot curve must be less than 5", call.=FALSE)
    q(status=1)
  }
  
  
  if (is.null(opt$plot)){
    print_help(opt_parser)
    print("Output file name for the report must be supplied.n", call.=FALSE)
    q(status=1)
  }else
  {
    
    j_df <- jsonToDataframe(r_df)
    
    label_title_opt <- "RunID: TaskID : SetID : PostID"
    label_opt <- do.call(paste, c(r_df[,c("runID", "taskID", "Collection", "PostProcessed")], sep=" : "))
    
    if(opt$whichplot =='det')
    {
      #detplot_opt(j_df, paste0(opt$title, " (DET)"), opt$plot, "RunID", r_df$runID)
      detplot_opt(j_df, paste0(opt$title, " (DET)"), opt$plot, label_title_opt, label_opt, opt$linetype, opt$size)
    } else if(opt$whichplot =='roc')
    {
      rocplot_opt(j_df, paste0(opt$title, " (ROC)"), opt$plot, label_title_opt, label_opt, opt$linetype, opt$size)
    }
    else
    {
      print_help(opt_parser)
      print("Plot type option must be one of 'det' and 'roc'", call.=FALSE)
      q(status=1)
    }
  }
}

#######################################################################
## command-line arguments for "DAPR" (Please ignore this part -IN PROGRESS)
#######################################################################

if(opt$inType == "DAPR")
{
  cat("In progress ... ", "\n")
}

