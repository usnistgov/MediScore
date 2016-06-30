suppressMessages(require(RUnit))

cat("BASIC FUNCTIONALITY validation of DSDValidator.r beginning...\n")
invisible(checkEquals(system("Rscript ../../../tools/DSDValidator/DSDValidate.r -q 1 -s lorem_NC2016_Splice_ImgOnly_p-baseline_1/lorem_NC2016_Splice_ImgOnly_p-baseline_1.csv -x NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv"),0))
cat("BASIC FUNCTIONALITY validated.\n")

h <- function(w) {
  if (any(grepl("running command",w))) {
    invokeRestart("muffleWarning")
  }
}

withCallingHandlers(
  {
errmsg<-""
#Same checks as Validate SSD, but applied to different files
cat("\nBeginning experiment ID naming error validations. Expect ERROR printouts for the next couple of cases. This is normal here.\n")
cat("\nCASE 0: Validating behavior when files don't exist.\n") 
invisible({
  checkEquals(system("Rscript ../../../tools/DSDValidator/DSDValidate.r -q 1 -s emptydir_NC2016_Splice_ImgOnly_p-baseline_1/emptydir_NC2016_Splice_ImgOnly_p-baseline_1.csv -x NC2016_Test0516_dfz/indexes/NC2016-splice-index0.csv"),1)
  errmsg <- system("Rscript ../../../tools/DSDValidator/DSDValidate.r -q 1 -s emptydir_NC2016_Splice_ImgOnly_p-baseline_1/emptydir_NC2016_Splice_ImgOnly_p-baseline_1.csv -x NC2016_Test0516_dfz/indexes/NC2016-splice-index0.csv",intern=TRUE)
  checkEquals(grepl("ERROR: I can't find your system output",errmsg[1]),TRUE)
  checkEquals(grepl("ERROR: I can't find your index file",errmsg[2]),TRUE)
  
})
cat("CASE 0 validated.\n")

cat("\nCASE 1: Validating behavior when detecting consecutive underscores ('_') in name...\n")
invisible({
  checkEquals(system("Rscript ../../../tools/DSDValidator/DSDValidate.r -q 1 -s lorem__NC2016_Spl_ImgOnly_p-baseline_1/lorem__NC2016_Spl_ImgOnly_p-baseline_1.csv -x NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv"),1)
  errmsg <- system("Rscript ../../../tools/DSDValidator/DSDValidate.r -q 1 -s lorem__NC2016_Spl_ImgOnly_p-baseline_1/lorem__NC2016_Spl_ImgOnly_p-baseline_1.csv -x NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv",intern=TRUE)
  checkEquals(grepl("ERROR: What kind of task is",errmsg[1]),TRUE)
})
cat("CASE 1 validated.\n")

cat("\nCASE 2: Validating behavior when detecting excessive underscores elsewhere...\n")
invisible({
  checkEquals(system("Rscript ../../../tools/DSDValidator/DSDValidate.r -q 1 -s lor_em_NC2016_Manipulation_ImgOnly_p-baseline_1/lor_em_NC2016_Manipulation_ImgOnly_p-baseline_1.csv -x NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv"),1)
  errmsg <- system("Rscript ../../../tools/DSDValidator/DSDValidate.r -q 1 -s lor_em_NC2016_Manipulation_ImgOnly_p-baseline_1/lor_em_NC2016_Manipulation_ImgOnly_p-baseline_1.csv -x NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv",intern=TRUE)
  checkEquals(grepl("ERROR: What kind of task is",errmsg[1]),TRUE)
})
cat("CASE 2 validated.\n")

cat("\nCASE 3: Validating behavior when detecting '+' in file name and an unrecogized task...\n")
invisible({
  checkEquals(system("Rscript ../../../tools/DSDValidator/DSDValidate.r -q 1 -s lorem+_NC2016_Removal_ImgOnly_p-baseline_1/lorem+_NC2016_Removal_ImgOnly_p-baseline_1.csv -x NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv"),1)
  errmsg <- system("Rscript ../../../tools/DSDValidator/DSDValidate.r -q 1 -s lorem+_NC2016_Removal_ImgOnly_p-baseline_1/lorem+_NC2016_Removal_ImgOnly_p-baseline_1.csv -x NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv",intern=TRUE)
  checkEquals(grepl("ERROR: The team name must not include characters",errmsg[1]),TRUE)
  checkEquals(grepl("ERROR: What kind of task is",errmsg[2]),TRUE)
})
cat("CASE 3 validated. Validating syntactic content of system output.\n")

cat("\nCASE 4: Validating behavior for incorrect headers, duplicate rows, and different number of rows than in index file...\n")
invisible({
  checkEquals(system("Rscript ../../../tools/DSDValidator/DSDValidate.r -q 1 -s lorem_NC2016_Splice_ImgOnly_p-baseline_2/lorem_NC2016_Splice_ImgOnly_p-baseline_2.csv -x NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv"),1)
  errmsg <- system("Rscript ../../../tools/DSDValidator/DSDValidate.r -q 1 -s lorem_NC2016_Splice_ImgOnly_p-baseline_2/lorem_NC2016_Splice_ImgOnly_p-baseline_2.csv -x NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv",intern=TRUE)
  checkEquals(grepl("ERROR: Your header\\(s\\)",errmsg[1]),TRUE)
  checkEquals(grepl("ERROR: Your system output contains duplicate rows",errmsg[2]),TRUE)
  checkEquals(grepl("ERROR: The number of rows in the system output does not match the number of rows in the index file\\.",errmsg[3]),TRUE)
})
cat("CASE 4 validated.\n")

cat("\nCase 5: Validating behavior when the number of columns in the system output is not equal to 5.\n")
invisible({
  checkEquals(system("Rscript ../../../tools/DSDValidator/DSDValidate.r -q 1 -s lorem_NC2016_Splice_ImgOnly_p-baseline_4/lorem_NC2016_Splice_ImgOnly_p-baseline_4.csv -x NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv"),1)
  errmsg <- system("Rscript ../../../tools/DSDValidator/DSDValidate.r -q 1 -s lorem_NC2016_Splice_ImgOnly_p-baseline_4/lorem_NC2016_Splice_ImgOnly_p-baseline_4.csv -x NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv",intern=TRUE)
  checkEquals(grepl("ERROR: The number of columns of the system output file must be equal to 5. Are you using '|' to separate your columns?",errmsg),TRUE)
})
cat("CASE 5 validated.\n")

cat("\nCASE 6: Validating behavior for mask semantic deviations. NC2016-1893.jpg and NC2016_6847-mask.jpg are (marked as) jpg's. NC2016_1993-mask.png is not single-channel. NC2016_4281-mask.png doesn't have the same dimensions...\n")
invisible({
  checkEquals(system("Rscript ../../../tools/DSDValidator/DSDValidate.r -q 1 -s ipsum_NC2016_Splice_ImgOnly_p-baseline_1/ipsum_NC2016_Splice_ImgOnly_p-baseline_1.csv -x NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv"),1)
  errmsg <- system("Rscript ../../../tools/DSDValidator/DSDValidate.r -q 1 -s ipsum_NC2016_Splice_ImgOnly_p-baseline_1/ipsum_NC2016_Splice_ImgOnly_p-baseline_1.csv -x NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv",intern=TRUE)
  checkEquals(grepl("is not a png. Make it into a png!",errmsg[1]),TRUE)
  checkEquals(grepl("Dimensions",errmsg[2]),TRUE)
  checkEquals(grepl("Dimensions",errmsg[3]),TRUE)
  checkEquals(grepl("ERROR: The mask image's length and width do not seem to be the same as the base image's.",errmsg[4]),TRUE)
  checkEquals(grepl("should be single-channel\\.",errmsg[5]),TRUE)
  checkEquals(grepl("is not a png. Make it into a png!",errmsg[6]),TRUE)
})
cat("CASE 6 validated.\n")

cat("\nCASE 7: Validating behavior when mask file is not present...\n") 
invisible({
  checkEquals(system("Rscript ../../../tools/DSDValidator/DSDValidate.r -q 1 -s lorem_NC2016_Splice_ImgOnly_p-baseline_3/lorem_NC2016_Splice_ImgOnly_p-baseline_3.csv -x NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv"),1)
  errmsg <- system("Rscript ../../../tools/DSDValidator/DSDValidate.r -q 1 -s lorem_NC2016_Splice_ImgOnly_p-baseline_3/lorem_NC2016_Splice_ImgOnly_p-baseline_3.csv -x NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv",intern=TRUE)
  checkEquals(grepl("does not exist! Did you name it wrong?",errmsg[1]),TRUE)
  checkEquals(grepl("does not exist! Did you name it wrong?",errmsg[2]),TRUE)
})
}
,warning=h)
cat("CASE 7 validated.\n")

cat("\nALL DSD VALIDATION TESTS SUCCESSFULLY PASSED.\n")
