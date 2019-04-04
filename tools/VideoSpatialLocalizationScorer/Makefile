################# Makefile for MaskScorer.py ##########################

#### Creation Date: June 9, 2016 
MAKE=make

#check:
#	(cd ../../lib; python2 -m unittest maskTests)
#	(rm ../../lib/testImg*.png)
#	(./maskcompcheckfiles_mini.sh)

check:
	(./test_init.sh)
#	(cd ../../lib; python2 score_video_masks.py)
#	(cd ../../lib; python2 -m unittest maskTests)
	(./run_integration_tests.sh)

render_readmes_to_html:
	(jupyter-nbconvert --to html MaskScorerReadMe.ipynb)
	(jupyter-nbconvert --to html modules/HTMLReadMe.ipynb)
