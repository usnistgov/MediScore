File: README.txt
Date: June 17, 2016
MediScore Version: 1.0.0

PYTHON

This directory contains MediScore, the NIST Medifor scoring and
evaluation toolkit. MediScore contains the source, documentation, and
example data for the following tools:

  SSDValidate      V1.0 - Single Source Detection Validator
  DSDValidate      V1.0 - Double Source Detection Validator
  DetectionScorer  V1.0 - Single/Double Source Detection Evaaluation
                          Scorer
  MaskScorer       V1.0 - Single/Double Source Mask Evaluation
                          (Localization) Scorer

This distribution consists of a set of R scripts intended to be run
from a command line.  These scripts have been tested under the
following versions of Ubuntu Linux and OS X.

  Mac OS X 10.9.5
  Ubuntu 14.04


INSTALLATION
------------

(Lines starting with % are command lines)

1) Install the latest R version from https://cran.r-project.org/.
   (This software was developed and tested using R version 3.2.4.
    Our recommendation is to use R version greater than 3.2.x)

2) Using R console (% R) ..

  * Install required packages
  % install.packages(c("optparse", "jsonlite", "scales", "ggplot2", "useful", "png", "RUnit", "RMySQL", "data.table"))

  * Note: If prompted you may need to select a 'CRAN mirror' prior to
  installing the aforementioned packages.  This can be done from
  within the R console by running ..

  % chooseCRANmirror(graphics=FALSE)

  * EBImage package installation
  % source('http://bioconductor.org/biocLite.R')
  % biocLite('EBImage') # please update all when prompted
  * Note: For Ubuntu, you may need to install “sudo apt-get install libfftw3-dev” prior to
  	installing the EBImage package above.

3) To test your installation, run MediScore's "make check" in the MediScore directory.
   You should expect to see ERROR messages pop up as we test the behavior of the functions.
   If you see the following messages, your make check completes succesfully.
   - ALL SSD VALIDATION TESTS SUCCESSFULLY PASSED
   - ALL DSD VALIDATION TESTS SUCCESSFULLY PASSED
   - DETECTION SCORER TESTS SUCCESSFULLY PASSED
   - MASK SCORER TESTS SUCCESSFULLY PASSED


USAGE
-----

Usage text for each script can be seen by executing the script with
the option '--help'.  For example:

  % cd MediScore/tools/DetectionScorer
  % Rscript DetectionScorer.r --help

Both DetectionScorer and MaskScorer scripts have additional
'ReadMe.html' files with more detailed information on their usage.

To try some command lines with data files, go to the testing
directories in 'MediScore/tools/DetectionScorer', and run the command
lines below.
  % Rscript DetectionScorer.r -i file -t manipulation -d ../../data/test_suite/detectionScorerTests \
  -r NC2016-manipulation-ref.csv -x NC2016-manipulation-index.csv \
  -s ../../data/test_suite/detectionScorerTests/D_NC2016_Manipulation_ImgOnly_p-me_1/D_NC2016_Manipulation_ImgOnly_p-me_1.csv \
  -o ../../data/test_suite/detectionScorerTests/temp_detreport.csv -p plot.pdf

To validate the system output, cd to the directory of the relevant
validator (e.g. MediScore/tools/SSDValidator) and execute the relevant
script with the index file after the -x option and your system output
after the -s option. For example:

   % Rscript SSDValidate.r -x ../../data/test_suite/validatorTests/NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv \
     -s ../../data/test_suite/validatorTests/foo_NC2016_Manipulation_ImgOnly_p-whole_1/foo_NC2016_Manipulation_ImgOnly_p-whole_1.csv
   % Rscript DSDValidate.r -x ../../data/test_suite/validatorTests/NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv \
     -s ../../data/test_suite/validatorTests/lorem_NC2016_Splice_ImgOnly_p-whole_1/lorem_NC2016_Splice_ImgOnly_p-whole_1.csv

You may also quiet printout with the -q option. Add -q 1 to suppress all but
error printout and -q 0 to suppress all printout. For example:

   % Rscript SSDValidate.r -q 1 -x ../../data/test_suite/validatorTests/NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv \
     -s ../../data/test_suite/validatorTests/foo_NC2016_Manipulation_ImgOnly_p-whole_1/foo_NC2016_Manipulation_ImgOnly_p-whole_1.csv

will allow the SSD Validator to print only error messages when validating the designated output.


HISTORY
-------

  June 17, 2016 - MediScore Version 1.0.0:

    - Initial release


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
Andrew P Delgado
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
