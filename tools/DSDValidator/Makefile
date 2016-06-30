################# Makefile for DSDValidate.r ##########################

#### Creation Date: June 8, 2016 
MAKE=make

check:
	(cd ../../data/test_suite/validatorTests; Rscript ValidateDSDTest.r)

makecompcheckfiles:
	(Rscript ../../data/test_suite/validatorTests/ValidateDSDTest.r > res_test_dsdval.txt-temp)
	(diff res_test_dsdval.txt-temp ../../data/test_suite/validatorTests/res_test_dsdval.txt > res_test_dsdval.txt-comp && rm res_test_dsdval.txt-temp)
