# *File: json2df.r
# *Date: 5/20/2016
# *Author: Yooyoung Lee
# *Status: Complete
#
# *Description: this function converts JSON format to dataframe for ROC points
#
# *Inputs
# *df: report dataframe
#
# *Outputs
# * dataframe
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

jsonToDataframe <- function(df)
{
  #require(jsonlite)
  #df <- all_df
  mylist <- list()
  for(i in 1:nrow(df))
  {
    roc_points <- fromJSON(as.character(df[i, "ROCpoints"]), flatten = TRUE)
    mytpr <- unlist(roc_points$tpr)
    myfpr <- unlist(roc_points$fpr)
    
    runid <- paste0('runid-',df[i, "runID"])
    tmp <- list(RUNID = runid, FPR=myfpr, TPR=mytpr)
    mylist[[runid]] <- tmp
  }
  #convert this to data frame  
  mydf = do.call("rbind", lapply(mylist, as.data.frame))
  return(mydf)
}


