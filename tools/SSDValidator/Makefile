################# Makefile for SSDValidate.r ##########################

#### Creation Date: June 8, 2016 
MAKE=make

check:
	(cd ../../data/test_suite/validatorTests; Rscript ValidateSSDTest.r)

makecompcheckfiles:
	(Rscript ../../data/test_suite/validatorTests/ValidateSSDTest.r > res_test_ssdval.txt-temp)
	(diff res_test_ssdval.txt-temp ../../data/test_suite/rvalidatorTests/es_test_ssdval.txt > res_test_ssdval.txt-comp && rm res_test_ssdval.txt-temp)
