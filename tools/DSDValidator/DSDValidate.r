#* File: DSDValidate.r
#* Date: 05/26/2016
#* Author: Daniel Zhou
#* Status: In progress
#
#* Description: This validates the format of the input of the Single
#  Source Detection system output along with the index file.
#
#* Requirements: This code requires the following packages:
#    - require(RMySQL)
#    - require("optparse")
#    - require ("EBImage")
#
#* Inputs
#    * -x, inIndex: index file name
#    * -s, inSys: system output file name
#    * -q, quiet: whether or not to silence output. Selecting this option silences output
#
#* Outputs
#    * List of NMM, MCC, HAM, WL1, and HL1
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

args <- commandArgs(trailingOnly = TRUE)

suppressMessages(require("optparse"))
suppressMessages(require("EBImage"))

#suppress warning about RMySQL being built in R version 3.2.3
h <- function(w) {
  if (any(grepl("R version",w))) {
    invokeRestart("muffleWarning")
  }
}

withCallingHandlers(
  {
    suppressMessages(require(RMySQL))
    
    option_list = list(
      make_option(c("-x", "--inIndex"), type="character", default=NULL, 
                  help="required index file", metavar="character"),
      make_option(c("-s", "--inSys"), type="character", default=NULL, 
                  help="required system output file", metavar="character"),
      make_option(c("-q","--quiet"),type="integer",default=NULL,
                  help="Suppress printed output to standard output. Type 0 to suppress all printed output, 1 to suppress only non-error output.")
      #CHECK: comment out everything having to do with database
      #make_option("--user", type="character", default= "simo", 
      #            help="Username to connect to DB [default= %default]", metavar="character"),
      #make_option("--pw", type="character", default= "123", 
      #            help="Password to connect to DB [default= %default]", metavar="character"),
      #make_option(c("-d","--dbname"), type="character", default= "mediforYooyoung2", 
      #            help="Database name [default= %default]", metavar="character"),
      #make_option("--host", type="character", default= "borg07.ncsl.nist.gov", 
      #            help="Host server [default= %default]", metavar="character")
    )
    
    opt_parser = OptionParser(option_list=option_list);
    opt = parse_args(opt_parser);
    
    printq <- function(somestring,iserr = FALSE) {
      if (is.null(opt$quiet)) {
        cat(paste(somestring,"\n"))
      } else if ((opt$quiet == 1) & iserr) {
        cat(paste(somestring,"\n"))
      }
    }
    
    if (is.null(opt$inIndex)){
      print_help(opt_parser)
      printq("ERROR: Index table must be supplied via -x.",TRUE)
      quit(status=1)
    }
    if (is.null(opt$inSys)){
      print_help(opt_parser)
      printq("ERROR: System output table must be supplied via -s.",TRUE)
      quit(status=1)
    }
    
    queryToDb <- function(q,db) {
      tryCatch({
        mydb <- dbConnect(MySQL(), user='simo', password='123', dbname=db, host='borg07.ncsl.nist.gov')
        myQuery <- dbGetQuery(mydb,q)
      }, error = function(e) {
        printq(paste("Error in query:",e),TRUE)
      },
      finally={
        if (dbIsValid(mydb)) {
          dbDisconnect(mydb)
          return(myQuery)
        }
      }
      )
    }
    
    #takes the name of the system file, including path
    nameCheck <- function(sysfilename) {
      printq("Validating the name of the system file...")
      sys.pieces <- strsplit(sysfilename,".",fixed=TRUE)
      sys.pieces <- unlist(sys.pieces)
      sys.ext <- sys.pieces[length(sys.pieces)]
      if (sys.ext != 'csv') {
        printq('ERROR: Your system output is not a csv!',TRUE)
        quit(status=1)
      }
      
      fileExpid <- unlist(strsplit(sys.pieces[length(sys.pieces)-1],"/",fixed=TRUE))
      dirExpid <- fileExpid[length(fileExpid)-1]
      fileExpid <- fileExpid[length(fileExpid)]
      if (fileExpid != dirExpid) {
        printq("ERROR: Please follow the naming convention. The system output should follow the naming <EXPID>/<EXPID>.csv.",TRUE)
        quit(status=1)
      }
      
      taskFlag <- 0
      teamFlag <- 0
      sysPath <- dirname(sysfilename)
      sysfName <- basename(sysfilename)
      arrSplit <- unlist(strsplit(sysfName,'_'))
      team <- arrSplit[1]
      ncid <- arrSplit[2]
      task <- arrSplit[3]
      condition <- arrSplit[4]
      sys <- arrSplit[5]
      version <- arrSplit[6]
      if (grepl('\\+',team)) {
        printq("ERROR: The team name must not include characters + or _",TRUE)
        teamFlag <- 1
      }
      if (task != 'Splice') {
        printq(paste('ERROR: What kind of task is ',task,'? It should be Splice!',sep=""),TRUE)
        taskFlag <- 1
      }
      if ((taskFlag == 0) && (teamFlag == 0)) {
        printq('The name of this file is valid!')
      } else {
        printq('The name of the file is not valid. Please review the requirements.',TRUE)
        quit(status=1) 
      }
    }
    
    #takes the system file and index file as data frames
    contentCheck <- function(sysfilename,idxfilename) {
      printq("Validating the syntactic content of the system output...")
      idxfile <- read.csv(idxfilename,head=TRUE,sep="|")
      sysfile <- read.csv(sysfilename,head=TRUE,sep="|")
      
      headerFlag <- 0
      dupFlag <- 0
      xrowFlag <- 0
      scoreFlag <- 0
      maskFlag <- 0
      
      if (ncol(sysfile) != 5) {
        printq("ERROR: The number of columns of the system output file must be equal to 5. Are you using '|' to separate your columns?",TRUE)
        quit(status=1)
      }
      
      sysHeads <- colnames(sysfile)
      if ((sysHeads[1] != "ProbeFileID") || (sysHeads[2] != "DonorFileID") || (sysHeads[3] != "ConfidenceScore") || 
          (sysHeads[4] != "ProbeOutputMaskFileName") || (sysHeads[5] != "DonorOutputMaskFileName")) {
        headlist <- c()
        properhl <- c()
        if (sysHeads[1] != "ProbeFileID") {
          headlist <- c(headlist,sysHeads[1])
          properhl <- c(properhl,"ProbeFileID")
        }
        if (sysHeads[2] != "DonorFileID") {
          headlist <- c(headlist,sysHeads[2])
          properhl <- c(properhl,"DonorFileID")
        }
        if (sysHeads[3] != "ConfidenceScore") {
          headlist <- c(headlist,sysHeads[3])
          properhl <- c(properhl,"ConfidenceScore")
        }
        if (sysHeads[4] != "ProbeOutputMaskFileName") {
          headlist <- c(headlist,sysHeads[4])
          properhl <- c(properhl,"ProbeOutputMaskFileName")
        }
        if (sysHeads[5] != "DonorOutputMaskFileName") {
          headlist <- c(headlist,sysHeads[5])
          properhl <- c(properhl,"DonorOutputMaskFileName")
        }
        
        printq(paste("ERROR: Your header(s)",paste(headlist,collapse=', '),"should be",paste(properhl,collapse=', '),"respectively."),TRUE)
        headerFlag <- 1
      }
      if (nrow(sysfile) != nrow(unique(sysfile))) {
        rowlist <- 1:nrow(sysfile)
        printq(paste("ERROR: Your system output contains duplicate rows for pairs: ",paste("(",as.character(sysfile[duplicated(sysfile),1]),
                    ",",as.character(sysfile[duplicated(sysfile),2]),")",collapse=', ',sep="")," at row(s): ",
                     as.character(rowlist[duplicated(sysfile)])," after the header. I recommended you delete these row(s).",sep=""),TRUE)
        dupFlag <- 1
      }
      
      if (nrow(sysfile) != nrow(idxfile)) {
        printq("ERROR: The number of rows in the system output does not match the number of rows in the index file.",TRUE)
        xrowFlag <- 1
      }
      
      if (!((headerFlag == 0) && (dupFlag == 0) && (xrowFlag == 0))) {
        printq("The contents of your file are not valid!",TRUE)
        quit(status = 1)
      }
      sysfile$ProbeFileID <- as.character(sysfile$ProbeFileID)
      sysfile$DonorFileID <- as.character(sysfile$DonorFileID)
      sysfile$ConfidenceScore <- as.numeric(sysfile$ConfidenceScore)
      sysfile$ProbeOutputMaskFileName <- as.character(sysfile$ProbeOutputMaskFileName)
      sysfile$DonorOutputMaskFileName <- as.character(sysfile$DonorOutputMaskFileName)
      
      idxfile$ProbeFileID <- as.character(idxfile$ProbeFileID)
      idxfile$DonorFileID <- as.character(idxfile$DonorFileID)
      idxfile$ProbeHeight <- as.numeric(idxfile$ProbeHeight)
      idxfile$ProbeWidth <- as.numeric(idxfile$ProbeWidth)
      idxfile$DonorHeight <- as.numeric(idxfile$DonorHeight)
      idxfile$DonorWidth <- as.numeric(idxfile$DonorWidth)

      sysPath <- dirname(sysfilename)
      for (i in 1:nrow(sysfile)) {
        #First get all the matching probe rows
        rowset <- idxfile[idxfile$ProbeFileID == sysfile$ProbeFileID[i],]
        if (nrow(rowset) == 0) {
          printq(paste(sysfile$ProbeFileID[i]," does not exist in the index file."),TRUE)
          printq("The contents of your file are not valid!",TRUE)
          quit(status = 1)
        }
        
        #search in these rows for DonorFileID matches. If empty, pair does not exist. Quit status 1
        if (is.na(match(rowset$DonorFileID,sysfile$DonorFileID[i]))) {
          printq(paste("ERROR: The pair (",sysfile$ProbeFileID[i],",",sysfile$DonorFileID[i],")"," does not exist in the index file.",sep=""),TRUE)
          printq("The contents of your file are not valid!",TRUE)
        }
        
        if (!is.numeric(sysfile$ConfidenceScore[i])) {
          printq(paste("ERROR: Score for pair (",sysfile$ProbeFileID[i],",",sysfile$DonorFileID[i],
                       ") is not numeric. Check row ",i,".",sep=""),TRUE)
          scoreFlag <- 1
        }
        #check mask validation
        if ((is.null(sysfile$ProbeOutputMaskFileName[i])) | (is.null(sysfile$DonorOutputMaskFileName[i]))) {
          printq(paste("At least one mask for pair (",sysfile$ProbeFileID[i],",",sysfile$DonorFileID[i],") appears to be absent. Skipping it.",sep=""))
          next
        } else if ((is.na(sysfile$ProbeOutputMaskFileName[i])) | (is.na(sysfile$DonorOutputMaskFileName[i]))) {
          printq(paste("At least one mask for pair (",sysfile$ProbeFileID[i],",",sysfile$DonorFileID[i],") appears to be absent. Skipping it.",sep=""))
          next
        } else if ((sysfile$ProbeOutputMaskFileName[i]=="") | (sysfile$DonorOutputMaskFileName[i]=="")) {
          printq(paste("At least one mask for pair (",sysfile$ProbeFileID[i],",",sysfile$DonorFileID[i],") appears to be absent. Skipping it.",sep=""))
          next
        }
        
        maskFlag <- maskFlag | maskCheck(paste(sysPath,sysfile$ProbeOutputMaskFileName[i],sep="/"),
                                         paste(sysPath,sysfile$DonorOutputMaskFileName[i],sep="/"),
                                         sysfile$ProbeFileID[i],sysfile$DonorFileID[i],idxfile,i)
      }
      
      #final validation
      if ((scoreFlag == 0) && (maskFlag == 0)) {
        printq("The contents of your file are valid!")
      } else {
        printq("The contents of your file are not valid!",TRUE)
        quit(status = 1)
      }
    }
    
    get.dims <- function(maskname) {
      #imgData <- system(paste('identify',maskname),intern=TRUE)
      #regData <- regexpr("[0-9]+x[0-9]+",imgData)
      #dim <- regmatches(imgData,regData)
      #dims <- as.numeric(unlist(strsplit(dim,'x')))
      dims <- dim(readImage(maskname))
      return(dims)
    }
    
    maskCheck <- function(pmaskname,dmaskname,probeid,donorid,indexfile,rownum) {
      #check to see if index file input image files are consistent with system output
      flag = 0
      printq(paste("Validating probe and donor mask pair (",pmaskname,",",dmaskname,") for (",probeid,",",donorid,") pair at row ",rownum,"...",sep=""))
      
      pmask.pieces <- strsplit(pmaskname,".",fixed=TRUE)
      pmask.pieces <- unlist(pmask.pieces)
      pmask.ext <- pmask.pieces[length(pmask.pieces)]
      if (pmask.ext != 'png') {
        printq(paste('ERROR: Probe mask image ',pmaskname,' for pair (',probeid,',',donorid,') at row ',rownum,' is not a png. Make it into a png!',sep=""),TRUE)
        return(1)
      }
      dmask.pieces <- strsplit(dmaskname,".",fixed=TRUE)
      dmask.pieces <- unlist(dmask.pieces)
      dmask.ext <- dmask.pieces[length(dmask.pieces)]
      if (dmask.ext != 'png') {
        printq(paste('ERROR: Probe mask image ',dmaskname,' for pair (',probeid,',',donorid,') at row ',rownum,' is not a png. Make it into a png!',sep=""),TRUE)
        return(1)
      }
      
      #check to see if png files exist before doing anything with them.
      if (!file.exists(pmaskname)) {
        printq(paste("ERROR:",pmaskname,"does not exist! Did you name it wrong?"),TRUE)
      }
      if (!file.exists(dmaskname)) {
        printq(paste("ERROR:",dmaskname,"does not exist! Did you name it wrong?"),TRUE)
      }
      if ((!file.exists(pmaskname)) | (!file.exists(dmaskname))) {
        return(1)
      }
      
      pdims <- get.dims(pmaskname)
      pbaseHeight <- indexfile$ProbeHeight[(indexfile$ProbeFileID == probeid) & (indexfile$DonorFileID == donorid)]
      pbaseWidth <-indexfile$ProbeWidth[(indexfile$ProbeFileID == probeid) & (indexfile$DonorFileID == donorid)]
      
      #To be deleted:
      if ((pbaseHeight == pdims[1]) & (pbaseWidth == pdims[2])) #EDIT: for flip cases
      {
        temp = pbaseHeight
        pbaseHeight = pbaseWidth
        pbaseWidth = temp
        printq("Mask is flipped.",TRUE)
      }
      if ((pbaseHeight != pdims[2]) || (pbaseWidth != pdims[1])) {
        printq(paste("Dimensions for ProbeImg of pair (",probeid,",",donorid,"): ",pbaseHeight,",",pbaseWidth,sep=""),TRUE)
        printq(paste("Dimensions of probe mask ",pmaskname,": ",pdims[2],",",pdims[1],sep=""),TRUE)
        printq("ERROR: The mask image's length and width do not seem to be the same as the base image's.",TRUE)
        flag = 1
      }
      
      ddims <- get.dims(dmaskname)
      dbaseHeight <- indexfile$DonorHeight[(indexfile$ProbeFileID == probeid) & (indexfile$DonorFileID == donorid)]
      dbaseWidth <-indexfile$DonorWidth[(indexfile$ProbeFileID == probeid) & (indexfile$DonorFileID == donorid)]
      
      if ((dbaseHeight != ddims[2]) || (dbaseWidth != ddims[1])) {
        printq(paste("Dimensions for DonorImg of pair (",probeid,",",donorid,"): ",dbaseHeight,",",dbaseWidth,sep=""),TRUE)
        printq(paste("Dimensions of donor mask ",dmaskname,": ",ddims[2],",",ddims[1],sep=""),TRUE)
        printq("ERROR: The mask image's length and width do not seem to be the same as the base image's.",TRUE)
        flag = 1
      }
      
      #pmaskImg <- readPNG(pmaskname) #EDIT: expensive. Need alternate option.
      #dmaskImg <- readPNG(dmaskname)
      
      if(!is.na(pdims[3])) {
        printq(paste("ERROR: Probe mask image ",pmaskname," of pair (",probeid,",",donorid,") should be single-channel.",sep=""),TRUE)
        flag = 1
      }
      if(!is.na(ddims[3])) {
        printq(paste("ERROR: Donor mask image ",dmaskname," of pair (",probeid,",",donorid,") should be single-channel.",sep=""),TRUE)
        flag = 1
      }
      
      if(flag == 0) {
        printq("These masks are valid.")
      }
      
      return(flag)
    }
    
    ############################ VALIDATION BEGINS HERE ###################################################
    db = NULL#opt$dbname
    
    ## Check if files exist
    if (!file.exists(opt$inSys)) {
      printq(paste("ERROR: I can't find your system output ",opt$inSys,"! Where is it?",sep=""),TRUE)
    }
    if (!file.exists(opt$inIndex)) {
      printq(paste("ERROR: I can't find your index file ",opt$inIndex,"! Where is it?",sep=""),TRUE)
    }
    if ((!file.exists(opt$inSys))| (!file.exists(opt$inIndex))) {
      quit(status=1)
    }
    
    nameCheck(opt$inSys)
    
    ## Checking if index file is a pipe-separated csv.
    printq("Checking if index file is a pipe-separated csv...")
    idx.pieces <- strsplit(opt$inIndex,".",fixed=TRUE)
    idx.pieces <- unlist(idx.pieces)
    idx.ext <- idx.pieces[length(idx.pieces)]
    if (idx.ext != 'csv') {
      printq("ERROR: Your index file should have csv as an extension! (It's separated by '|', I know...)",TRUE)
      quit(status=1)
    }
    printq("Your index file appears to be a pipe-separated csv, for now. Hope it isn't separated by commas.")
    
    contentCheck(opt$inSys,opt$inIndex)
    
    if (is.null(db)) {
      quit(status = 0)
    }
    
    ############################# POPULATE SYSTEM OUTPUT DB ###############################################
    
    sysPath <- dirname(opt$inSys)
    sysfName <- basename(opt$inSys)
    sysBase <- unlist(strsplit(sysfName,'.',fixed=TRUE))
    arrSplit <- unlist(strsplit(sysBase[1],'_'))
    
    team <- arrSplit[1]
    ncid <- arrSplit[2]
    task <- arrSplit[3]
    condition <- arrSplit[4]
    sys <- arrSplit[5]
    version <- arrSplit[6]
    
    #System Table
    createSys <- "CREATE TABLE if not exists `System` (
    `sysID` int(11) NOT NULL AUTO_INCREMENT,
    `teamID` int(11) DEFAULT NULL,
    `taskID` varchar(255) DEFAULT NULL,
    `conditionID` varchar(255),
    `sys` varchar(255),
    `version` int,
    `systemPath` varchar(255),
    `expid` varchar(255),
    PRIMARY KEY (`sysID`),
    KEY `teamID` (`teamID`),
    KEY `taskID` (`taskID`),
    CONSTRAINT `System_ibfk_1` FOREIGN KEY (`teamID`) REFERENCES `Team` (`teamID`) on update cascade
) ENGINE=InnoDB DEFAULT CHARSET=latin1"

    createSSD <- "create table if not exists SystemFileResponseSSD(
    `sysID` int(11) NOT NULL DEFAULT '0',
    `taskID` varchar(255),
    `fileID` varchar(255) NOT NULL,
    `newScore` float,
    `mask` varchar(255)
    ) ENGINE=InnoDB DEFAULT CHARSET=latin1"

    invisible({
      queryToDb(createSys,db)
      queryToDb(createSSD,db)
    })
    
    qTeamID <- paste("select teamID from ",db,".Team where teamName='",team,"'",sep="")
    invisible(
      resTeamID <- queryToDb(qTeamID,db)
    )
    
    teamID = as.integer(unlist(resTeamID)[1])
    
    #CHECK: still add all this into database?
    qSys = paste("insert into ",db,".System (teamID,taskID,conditionID,sys,version,systemPath,expid)
             select * from (select '",teamID,"' as teamID, 'NC2016_",task,"' as taskID, '",condition,"' as conditionID,'",sys,"' as sys, '",version,"' as version, '",sysPath,"' as systemPath, '",sysfName,"' as expid) as tmp
             where not exists (
             select teamID, taskID, systemPath, expid from ",db,".System where teamID='",teamID,"' and expid='",sysfName,"' and systempath='",sysPath,"')",sep="")
    
    invisible(queryToDb(qSys,db))
    
    sys.csv <- read.csv(opt$inSys,head=TRUE,sep="|")
    idx.csv <- read.csv(opt$inIndex,head=TRUE,sep="|")
    
    for (i in 1:nrow(sys.csv)) {
      qSysID <- paste("select sysID from ",db,".System s inner join ",db,".Team t on s.teamID=t.teamID where teamName='",team,"' and s.sys='",sys,"'",sep="")
      invisible(resSysID <- queryToDb(qSysID,db))
      
      sysID <- as.integer(unlist(resSysID)[1])
      fID <- as.character(sys.csv[i,1])
      
      task <- as.character(idx.csv[as.character(idx.csv$FileID) == fID,]$TaskID)
      
      newScore <- as.numeric(sys.csv[i,2])
      mask <- as.character(sys.csv[i,3])
      
      qSFR <- paste("insert into ",db,".SystemFileResponseSSD (sysID,taskID,fileID,newScore,mask) values ('",sysID,"','",task,"','",fID,"','",newScore,"','",mask,"')",sep="")
      invisible(queryToDb(qSFR,db))
    }
  }
  ,warning=h)
