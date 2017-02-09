################# Makefile for MaskScorer.py ##########################

#### Creation Date: June 9, 2016 
MAKE=make

check:
	(cd ../../lib; python2 -m unittest maskTests)
	(rm ../../lib/testImg*.png)
	(./maskcompcheckfiles_mini.sh)

checkplus:
	(cd ../../lib; python2 -m unittest maskTests)
	(rm ../../lib/testImg*.png)
	(./maskcompcheckfiles_0.sh)
	(./maskcompcheckfiles_1.sh)
	(./maskcompcheckfiles_2.sh)

