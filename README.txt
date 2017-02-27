File: README.txt
Date: January 25, 2017
MediScore Version: 1.0.3

This directory contains MediScore, the NIST Medifor scoring and
evaluation toolkit. MediScore contains the source, documentation, and
example data for the following tools:

  Validator        V2.0 - Single/Double Source Detection Validator
  DetectionScorer  V2.1 - Single/Double Source Detection Evaluation
                          Scorer
  MaskScorer       V2.1 - Single/Double Source Mask Evaluation
                          (Localization) Scorer

This distribution consists of a set of Python2.7 scripts intended to be run
from a command line.  These scripts have been tested under the
following versions of Ubuntu Linux and OS X.

  Mac OS X 10.11.6
  Ubuntu Linux 14.04.4


INSTALLATION
------------

(Lines starting with % are command lines)

1) Install Python 2.7 (tested in Python == 2.7.12).

2) Required packages:
  Prior to running the Scorer, the following packages need to be installed :
  - opencv (tested in version 2.4.13)
  - numpy  (tested in version 1.11.1)
  - pandas (tested in version 0.18.1) - make sure to use the latest version.
  - matplotlib (tested in version 1.5.1)
  - scipy (tested in version 0.18.0)
  - scikit-learn (tested in version 0.17.1)
  - unittest

* Installation example for Linux:
  - Install Anaconda for Python 2.7 version: https://www.continuum.io/downloads
    1) https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86_64.sh
    2) $ bash Anaconda2-4.2.0-Linux-x86_64.sh
  - Install opencv using conda:
    $ conda install -c https://conda.binstar.org/menpo opencv
  - Check the required packages using "conda list"
  - Install missing packages using "conda install"

* SSH with XQuartz on OS X for saving or displaying plots:
  - If you want to run the code using ssh on OSX, you may install XQuartz from the website:
    https://www.xquartz.org/, then
    $ ssh -X servername

3) To test your installation, run MediScore's "make check" in the MediScore directory.
   You should expect to see ERROR messages pop up as we test the behavior of the functions.
   If you see the following messages, your make check completed successfully.
   - ALL DSD VALIDATION TESTS SUCCESSFULLY PASSED
   - ALL SSD VALIDATION TESTS SUCCESSFULLY PASSED
   - MASK SCORER TESTS SUCCESSFULLY PASSED
   - DETECTION SCORER TESTS SUCCESSFULLY PASSED

   Due to the sheer volume of the mask scorer's test cases and the enormous amount of computation
   involved for each test case, the make check passes through only two test cases. For a thorough
   check of the mask scorer's capabilities, go into the tools/MaskScorer directory and type
   'make checkplus'. Expect it to run for 3-4 hours.


USAGE
-----

Usage text for each script can be seen by executing the script with
the option '--help'.  For example:

  $ cd MediScore/tools/DetectionScorer
  $ python2 DetectionScorer.py --help

Both DetectionScorer and MaskScorer scripts have additional
HTML files (DetectionScorerReadMe.html and MaskScorerReadMe.html) with more detailed information on their usage.

To try some command lines with data files, go to the testing
directories in 'MediScore/tools/DetectionScorer', and run the command
lines below.
  $ python DetectionScorer.py -t manipulation --refDir ../../data/test_suite/detectionScorerTests/sample \
  -r NC2016-manipulation-ref.csv -x NC2016-manipulation-index.csv --sysDir ../../data/test_suite/detectionScorerTests/sample \
  -s D_NC2016_Manipulation_ImgOnly_p-me_1/D_NC2016_Manipulation_ImgOnly_p-me_1.csv --outRoot ./testcases/NC16_01 --display

To validate the system output, cd to the tools directory and execute the
following script with the index file after the -x option, your system output
after the -s option, and the validation task after the -vt option. For example:

   $ python2 validator.py -x ../data/test_suite/validatorTests/NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv \
     -s ../../data/test_suite/validatorTests/foo_NC2016_Manipulation_ImgOnly_p-whole_1/foo_NC2016_Manipulation_ImgOnly_p-whole_1.csv \
     -vt SSD
   $ python2 validator.py -x ../data/test_suite/validatorTests/NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv \
     -s ../../data/test_suite/validatorTests/lorem_NC2016_Splice_ImgOnly_p-whole_1/lorem_NC2016_Splice_ImgOnly_p-whole_1.csv \
     -vt DSD

You may also control printout with the -v option. Add -v 1 for more detailed print output and -v 0 to suppress all printout. For example:

   $ python2 validator.py -x ../data/test_suite/validatorTests/NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv \
     -s ../../data/test_suite/validatorTests/foo_NC2016_Manipulation_ImgOnly_p-whole_1/foo_NC2016_Manipulation_ImgOnly_p-whole_1.csv \
     -vt SSD \
     -v 1

will generate more detailed information for the SSD Validator than if -v is not selected.


HISTORY
-------

  Oct. 28, 2016 - MediScore Version 1.0.0:
    - Python release
  Jan. 6, 2017 - MediScore Version 1.0.2:
    - Started to support Selective Manipulation Scoring. This is a roll out of the
      new DetectionScorer.  Note the filter options changed
  Jan. 25, 2017 - MediScore Version 1.0.3:
    - Add Selective Manipulation Scoring to the Mask scoring.
    

CONTACT
-------

Please send bug reports to <medifor-nist@nist.gov>

Please include the command line, files and text output, including the
error message in your email.




TEST CASE BUG REPORT (TODO)
--------------------

If the error occurred wile doing a 'make check', go in the directory
associated with the tool that failed (for example:
'tools/DetectionScorer'), and type 'make makecompcheckfiles'. This
process will create a file corresponding to each test number named
"res_test*.txt-comp". These file are (like their .txt equivalent) text
files that can be compared to the original "res_test*.txt" files in the
data/test_suite directory.

 When a test fails, please send us the "res_test*.txt-comp" file of the
failed test(s) for us to try to understand what happened, as well as
information about your system (OS, architecture, ...) that you think
might help us.  Thank you for helping us improve MEDIFOR system.



AUTHORS
-------
Jonathan G. Fiscus (PI)
Andrew Delgado
Timothee Kheyrkhah
Yooyoung Lee
Daniel F. Zhou


COPYRIGHT
---------

Full details can be found at: http://nist.gov/data/license.cfm

This software was developed at the National Institute of Standards and
Technology by employees of the Federal Government in the course of
their official duties.  Pursuant to Title 17 Section 105 of the United
States Code this software is not subject to copyright protection
within the United States and is in the public domain. This evaluation
framework is an experimental system.  NIST assumes no responsibility
whatsoever for its use by any party, and makes no guarantees,
expressed or implied, about its quality, reliability, or any other
characteristic.

We would appreciate acknowledgement if the software is used.  This
software can be redistributed and/or modified freely provided that any
derivative works bear some notice that they are derived from it, and
any modified versions bear some notice that they have been modified.

THIS SOFTWARE IS PROVIDED "AS IS."  With regard to this software, NIST
MAKES NO EXPRESS OR IMPLIED WARRANTY AS TO ANY MATTER WHATSOEVER,
INCLUDING MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
