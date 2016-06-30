suppressMessages(require(RUnit))

cat("BASIC FUNCTIONALITY validation of SSDValidator.r beginning...\n")
invisible(checkEquals(system("Rscript ../../../tools/SSDValidator/SSDValidate.r -q 1 -s foo_NC2016_Manipulation_ImgOnly_p-baseline_1/foo_NC2016_Manipulation_ImgOnly_p-baseline_1.csv -x NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv"),0))
cat("BASIC FUNCTIONALITY validated.\n")

h <- function(w) {
  if (any(grepl("running command",w))) {
    invokeRestart("muffleWarning")
  }
}

errmsg<-""
cat("\nBeginning experiment ID naming error validations. Expect ERROR printouts for the next couple of cases. This is normal here.\n")
cat("CASE 0: Validating behavior when files don't exist.\n")
withCallingHandlers(
  {
    invisible({
      checkEquals(system("Rscript ../../../tools/SSDValidator/SSDValidate.r -q 1 -s emptydir_NC2016_Splice_ImgOnly_p-baseline_1/foo__NC2016_Manipulation_ImgOnly_p-baseline_1.csv -x NC2016_Test0516_dfz/indexes/NC2016-manipulation-index0.csv"),1)
      errmsg <- system("Rscript ../../../tools/SSDValidator/SSDValidate.r -q 1 -s emptydir_NC2016_Splice_ImgOnly_p-baseline_1/foo__NC2016_Manipulation_ImgOnly_p-baseline_1.csv -x NC2016_Test0516_dfz/indexes/NC2016-manipulation-index0.csv",intern=TRUE)
      checkEquals(grepl("ERROR: I can't find your system output",errmsg),TRUE)
    })

    cat("CASE 0 validated.\n")

    cat("\nCASE 1: Validating behavior when detecting consecutive underscores ('_') in name...\n")
    invisible({
      checkEquals(system("Rscript ../../../tools/SSDValidator/SSDValidate.r -q 1 -s foo__NC2016_Manipulation_ImgOnly_p-baseline_1/foo__NC2016_Manipulation_ImgOnly_p-baseline_1.csv -x NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv"),1)
      errmsg <- system("Rscript ../../../tools/SSDValidator/SSDValidate.r -q 1 -s foo__NC2016_Manipulation_ImgOnly_p-baseline_1/foo__NC2016_Manipulation_ImgOnly_p-baseline_1.csv -x NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv",intern=TRUE)
      checkEquals(grepl("ERROR: What kind of task is",errmsg[1]),TRUE)
    })
    cat("CASE 1 validated.\n")
    
    cat("\nCASE 2: Validating behavior when detecting excessive underscores elsewhere...\n")
    invisible({
      checkEquals(system("Rscript ../../../tools/SSDValidator/SSDValidate.r -q 1 -s fo_o_NC2016_Manipulation_ImgOnly_p-baseline_1/fo_o_NC2016_Manipulation_ImgOnly_p-baseline_1.csv -x NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv"),1)
      errmsg <- system("Rscript ../../../tools/SSDValidator/SSDValidate.r -q 1 -s fo_o_NC2016_Manipulation_ImgOnly_p-baseline_1/fo_o_NC2016_Manipulation_ImgOnly_p-baseline_1.csv -x NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv",intern=TRUE)
      checkEquals(grepl("ERROR: What kind of task is",errmsg[1]),TRUE)
    })
    cat("CASE 2 validated.\n")
    
    cat("\nCASE 3: Validating behavior when detecting '+' in file name and an unrecognized task...\n")
    invisible({
      checkEquals(system("Rscript ../../../tools/SSDValidator/SSDValidate.r -q 1 -s foo+_NC2016_Manip_ImgOnly_p-baseline_1/foo+_NC2016_Manip_ImgOnly_p-baseline_1.csv -x NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv"),1)
      errmsg <- system("Rscript ../../../tools/SSDValidator/SSDValidate.r -q 1 -s foo+_NC2016_Manip_ImgOnly_p-baseline_1/foo+_NC2016_Manip_ImgOnly_p-baseline_1.csv -x NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv",intern=TRUE)
      checkEquals(grepl("ERROR: The team name must not include characters",errmsg[1]),TRUE)
      checkEquals(grepl("ERROR: What kind of task is",errmsg[2]),TRUE)
    })
    cat("CASE 3 validated. Validating syntactic content of system output.\n")
    
    cat("\nCASE 4: Validating behavior for incorrect headers, duplicate rows, and different number of rows than in index file...\n")
    invisible({
      checkEquals(system("Rscript ../../../tools/SSDValidator/SSDValidate.r -q 1 -s foo_NC2016_Manipulation_ImgOnly_p-baseline_2/foo_NC2016_Manipulation_ImgOnly_p-baseline_2.csv -x NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv"),1)
      errmsg <- system("Rscript ../../../tools/SSDValidator/SSDValidate.r -q 1 -s foo_NC2016_Manipulation_ImgOnly_p-baseline_2/foo_NC2016_Manipulation_ImgOnly_p-baseline_2.csv -x NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv",intern=TRUE)
      checkEquals(grepl("ERROR: Your header\\(s\\)",errmsg[1]),TRUE)
      checkEquals(grepl("ERROR: Your system output contains duplicate rows",errmsg[2]),TRUE)
      checkEquals(grepl("ERROR: The number of rows in the system output does not match the number of rows in the index file\\.",errmsg[3]),TRUE)
    })
    cat("CASE 4 validated.\n")
    
    cat("\nCASE 5: Validating behavior when mask is not a png...\n")
    invisible({
      checkEquals(system("Rscript ../../../tools/SSDValidator/SSDValidate.r -q 1 -s bar_NC2016_Removal_ImgOnly_p-baseline_1/bar_NC2016_Removal_ImgOnly_p-baseline_1.csv -x NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv"),1)
      errmsg <- system("Rscript ../../../tools/SSDValidator/SSDValidate.r -q 1 -s bar_NC2016_Removal_ImgOnly_p-baseline_1/bar_NC2016_Removal_ImgOnly_p-baseline_1.csv -x NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv",intern=TRUE)
      checkEquals(grepl("is not a png. Make it into a png!",errmsg[1]),TRUE)
    })
    cat("CASE 5 validated.\n")
    
    cat("\nCASE 6: Validating behavior when mask is not single channel and when mask does not have the same dimensions.\n")
    invisible({
      checkEquals(system("Rscript ../../../tools/SSDValidator/SSDValidate.r -q 1 -s baz_NC2016_Manipulation_ImgOnly_p-baseline_1/baz_NC2016_Manipulation_ImgOnly_p-baseline_1.csv -x NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv"),1)
      errmsg <- system("Rscript ../../../tools/SSDValidator/SSDValidate.r -q 1 -s baz_NC2016_Manipulation_ImgOnly_p-baseline_1/baz_NC2016_Manipulation_ImgOnly_p-baseline_1.csv -x NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv",intern=TRUE)
      checkEquals(grepl("Dimensions",errmsg[1]),TRUE)
      checkEquals(grepl("Dimensions",errmsg[2]),TRUE)
      checkEquals(grepl("ERROR: The mask image's length and width do not seem to be the same as the base image's.",errmsg[3]),TRUE)
      checkEquals(grepl("should be single-channel\\.",errmsg[4]),TRUE)
    })
    cat("CASE 6 validated.\n")
    
    cat("\nCASE 7: Validating behavior when system output column number is not equal to 3.\n") 
    invisible({
      checkEquals(system("Rscript ../../../tools/SSDValidator/SSDValidate.r -q 1 -s foo_NC2016_Manipulation_ImgOnly_p-baseline_3/foo_NC2016_Manipulation_ImgOnly_p-baseline_3.csv -x NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv"),1)
      errmsg <- system("Rscript ../../../tools/SSDValidator/SSDValidate.r -q 1 -s foo_NC2016_Manipulation_ImgOnly_p-baseline_3/foo_NC2016_Manipulation_ImgOnly_p-baseline_3.csv -x NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv",intern=TRUE)
      checkEquals(grepl("ERROR: The number of columns of the system output file must be equal to 3. Are you using '|' to separate your columns?",errmsg),TRUE)
    })
    cat("CASE 7 validated.\n")
    
    cat("\nCASE 8: Validating behavior when mask file is not present.\n") 
    invisible({
      checkEquals(system("Rscript ../../../tools/SSDValidator/SSDValidate.r -q 1 -s foo_NC2016_Manipulation_ImgOnly_p-baseline_4/foo_NC2016_Manipulation_ImgOnly_p-baseline_4.csv -x NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv"),1)
      errmsg <- system("Rscript ../../../tools/SSDValidator/SSDValidate.r -q 1 -s foo_NC2016_Manipulation_ImgOnly_p-baseline_4/foo_NC2016_Manipulation_ImgOnly_p-baseline_4.csv -x NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv",intern=TRUE)
      checkEquals(grepl("does not exist! Did you name it wrong?",errmsg[1]),TRUE)
    })
  }
,warning=h)
cat("CASE 8 validated.\n")

cat("\nALL SSD VALIDATION TESTS SUCCESSFULLY PASSED.\n")
