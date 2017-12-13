################# Makefile for Video Temporal Localisation tools ##########################

#### Creation Date: Nov 16, 2017 
MAKE=make

check:
	@(echo "** Running UnitTests **\n")
	(python Unittest_IntervalConfusion.py)
	@(echo "** Running Intergration Test Case **\n")
	@(./run_integration_tests.sh)

makecheckfiles:
	(./make_checkfiles.sh)

render_readmes_to_html:
	(jupyter-nbconvert --execute --to html VideoTemporalLocalizationScoringReadMe.ipynb)