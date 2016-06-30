################### Makefile for All Tools ##########################

#### Creation Date: June 9, 2016
MAKE=make

check:
	(cd tools/SSDValidator; make check)
	(cd tools/DSDValidator; make check)
	(cd tools/DetectionScorer; make check)
	(cd tools/MaskScorer; make check)
