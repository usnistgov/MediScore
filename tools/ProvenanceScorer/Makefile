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
	@(echo "** Running integration tests for validator **\n")
	@(./run_integration_tests_validator.sh)
	@(./finalcheck.sh)

makecheckfiles:
	(./make_checkfiles.sh)

render_readmes_to_html:
	(jupyter-nbconvert --execute --to html ProvenanceFilteringScorerReadMe.ipynb)
	(jupyter-nbconvert --execute --to html ProvenanceGraphBuildingScorerReadMe.ipynb)
	(jupyter-nbconvert --to html ProvenanceValidatorReadMe.ipynb)
