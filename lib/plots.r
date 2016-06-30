# *File: plots.r
# *Date: 5/20/2016
# *Author: Yooyoung Lee
# *Status: Complete
#
# *Description: this code visualizes the performance using ROC and DET
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

suppressWarnings(suppressMessages(require(scales)))
suppressWarnings(suppressMessages(require(ggplot2)))
### tdeviate transform for axes
tdeviate_trans <- function() 
{
  trans_new("tdeviate",qnorm,pnorm)
}

# *Description: this function reads a dataframe and plots DET curves with a label option
#              
# *Inputs
# * df: dataframe
# * myTitle: title for plot
# * plotFName: plot file name to be saved
# * labelTitle: legend title
# * myLabel: legend label
#
# *Outputs
# * DET plot (pdf)
detplot_opt <- function(df, myTitle, plotFName, labelTitle, myLabel, myLineType, mySize)
{
  
  breaks_vec <-c(0, 0.0001, 0.001, 0.004, .01, .02, .05, .10, .20, .40, .60, .80, .90, .95, .98, .99, .995, .999, 0.9999)
  gp <- ggplot(df, aes(x = FPR, y = (1.0 - TPR), color = factor(RUNID, labels = myLabel))) + 
    #geom_line(aes(linetype = RUNID), size=2) + 
    geom_line(aes(linetype = myLineType), size=mySize) + 
    geom_point(alpha=0.5, size=mySize) +
    geom_abline(intercept=0, slope=-1,linetype='dashed', color="black") +
    scale_linetype_discrete(guide = FALSE) +
    scale_x_continuous(trans="tdeviate",breaks=breaks_vec, limits = c(0.0001,.9999), labels=breaks_vec*100) +
    scale_y_continuous(trans="tdeviate",breaks=breaks_vec, limits = c(0.0001,.9999), labels=breaks_vec*100) +
    theme_bw() +
    theme(legend.position=c(.75, .9)) +
    labs(title = myTitle, size = 20, x = "False Positive Rate", y = "False Negative Rate", color = labelTitle)
  #plot(gp)
  ggsave(plotFName, plot = gp, width = 8, height = 7.5, units = "in")
}

# *Description: this function reads a dataframe and plots ROC curves with a label option
# *             
# *Inputs
# * df: dataframe
# * myTitle: title for plot
# * plotFName: plot file name to be saved
# * labelTitle: legend title
# * myLabel: legend label
#
# *Outputs
# * ROC plot (pdf)
rocplot_opt <- function(df, myTitle, plotFName, labelTitle, myLabel, myLineType, mySize)
{
  #breaks_vec <-c(0, 0.0001, 0.001, 0.004, .01, .02, .05, .10, .20, .40, .60, .80, .90, .95, .98, .99, .995, .999, 0.9999)
  gp <- ggplot(df, aes(x = FPR, y = TPR, color = factor(RUNID, labels = myLabel))) + 
    #geom_line(aes(linetype = RUNID), size=1) + 
    geom_line(linetype = myLineType, size=mySize) + 
    geom_point(alpha=0.5, size=mySize+.4) +
    geom_abline(linetype='dashed', color="black") +
    scale_linetype_discrete(guide = FALSE) +
    #scale_x_log10() +
    #scale_x_continuous(trans="tdeviate",breaks=breaks_vec, limits = c(0.0001, 0.9999), labels=breaks_vec*100) +
    #scale_y_continuous(trans="tdeviate",breaks=breaks_vec, limits = c(0.0001, 0.9999), labels=breaks_vec*100) +
    scale_x_continuous(breaks=round(seq(0, 1, length.out=11),digits=1)) +
    scale_y_continuous(breaks=round(seq(0, 1, length.out=11),digits=1)) +
    theme_bw() +
    theme(legend.position=c(.75, .25)) +
    labs(title = myTitle, size = 20, x = "False Positive Rate", y = "True Positive Rate", color = labelTitle)
  #plot(gp)
  ggsave(plotFName, plot = gp, width = 8, height = 7.5, units = "in")
}

