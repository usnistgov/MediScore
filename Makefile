################### Makefile for All Tools ##########################

#### Creation Date: June 9, 2016
MAKE=make

check:
	(cd tools/Validator; python2 -m unittest validatorUnitTest)
	(cd tools/MaskScorer; make check)
	(cd tools/DetectionScorer; make check)
	(cd tools/ProvenanceScorer; make check)
