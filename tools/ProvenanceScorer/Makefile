################# Makefile for ProvenanceScorer tools ##########################

#### Creation Date: Feb 27, 2017 
MAKE=make

check:
	@(echo "** Running UnitTests **\n")
	(python2 ../../lib/ProvenanceGraphBuildingUnitTest.py)
	# Ignoring warnings here as we divide by zero for some tests,
	# which produce warnings
	(python2 -W ignore ../../lib/ProvenanceMetricsUnitTest.py)
	@(echo "** Running integration tests **\n")
	@(./run_integration_tests.sh)
	@(echo "PROVENANCE SCORER TESTS SUCCESSFULLY PASSED")

makecheckfiles:
	(./make_checkfiles.sh)

render_readmes_to_html:
	(jupyter-nbconvert --to html ProvenanceFilteringScorerReadMe.ipynb)
	(jupyter-nbconvert --to html ProvenanceGraphBuildingScorerReadMe.ipynb)
