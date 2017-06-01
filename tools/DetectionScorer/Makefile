################# Makefile for ProvenanceScorer tools ##########################

#### Creation Date: Feb 27, 2017
MAKE=make

check:
	@(echo "** Running UnitTests **\n")
	# Ignoring warnings here as we divide by zero for some tests,
	# which produce warnings
	(python2 -W ignore DetectionScorerUnitTest.py)
	@(echo "** Running detection tests **\n")
	@(./run_detection_tests.sh)
	@(echo "DETECTION SCORER TESTS SUCCESSFULLY PASSED")

makecheckfiles:
	(./make_checkfiles.sh)

render_readmes_to_html:
	(jupyter-nbconvert --execute --to html DetectionScorerReadMe.ipynb)
	(jupyter-nbconvert --execute --to html DMRenderReadMe.ipynb)
