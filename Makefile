################### Makefile for All Tools ##########################

#### Creation Date: June 9, 2016
MAKE=make

check:
	(cd tools/VideoTemporalLocalizationScorer; make check)
	(cd tools/VideoSpatialLocalizationScorer; make check)
#	(cd tools/Validator; make check)
#	(cd tools/DetectionScorer; make check)
#	(cd tools/ProvenanceScorer; make check)
#	(cd tools/LocalizationVisualizer; make check)
#	(cd tools/MaskScorer; make check)
