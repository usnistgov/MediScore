File: README.txt
Date: November 20, 2017
MediScore Version: 1.2.0

This directory contains MediScore, the NIST Medifor scoring and
evaluation toolkit. MediScore contains the source, documentation, and
example data for the following tools:

  Validator                       - Single/Double Source Detection Validator
  DetectionScorer                 - Single/Double Source Detection Evaluation
                                    Scorer
  MaskScorer                      - Single/Double Source Mask Evaluation
                                    (Localization) Scorer
  ProvenanceFilteringScorer       - Scorer for Provenance Filtering
  ProvenanceGraphBuildingScorer   - Scorer for Provenance Graph Building
  VideoTemporalLocalisationScoer  - Scorer for Video Temporal Localization


This distribution consists of a set of Python2.7 scripts intended to be run
from a command line.  These scripts have been tested under the
following versions of Ubuntu Linux and OS X.

  Mac OS X 10.12.3
  Ubuntu Linux 14.04.4


INSTALLATION
------------

(Lines starting with $ are command lines)

1) Install Python 2.7 (tested in Python == 2.7.13).

2) Required packages:
  Prior to running the Scorer, the following packages need to be installed :
  - opencv (tested in versions 2.4.13 and 3.4.0)
  - numpy  (tested in versions 1.14.0)
  - pandas (tested in version 0.21.1) - make sure to use the latest version.
  - matplotlib (tested in version 1.5.1)
  - scipy (tested in version 0.18.0)
  - scikit-learn (tested in version 0.17.1)
  - rawpy (tested in version 0.9.0)
  - numpngw (tested in version 0.0.6)
  - Glymur (tested in version 0.8.12)
  - bokeh (tested in version 0.12.13)
  - jsonschema (tested in version 2.5.1)
  Most of these packages can be installed by running basic_install.sh on your machine.

  Optional :
  - pydot (tested in version 1.2.3) -- For graphical output from
    ProvenanceGraphBuildingScorer.py

  ImageMagick is not required, but is highly recommended to accelerate
  the validator.  Download instructions may be found in the following
  link: http://imagemagick.org/script/download.php

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
  $ python2 DetectionScorer.py -t manipulation --refDir ../../data/test_suite/detectionScorerTests/sample \
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
    - Added Selective Manipulation Scoring to the Mask scoring.

  Feb. 28, 2017 - MediScore Version 1.1.0:
    - Added the Provenance scorers.
    - Add Selective Manipulation Scoring to the Mask scoring.

  Mar 10, 2017 - MediScore Version 1.1.1:
    * Validator:
      - Validator now reads in a file stream for the DSD task. Major speedup applied.
      - Neglect mask feature added to validator for speedup.
      - ImageMagick channel reading slightly fixed.
    * MaskScorer:
      - The dilation parameter for selective Mask scoring has been changed from 9 to 11.
      - Donor splice reference mask is expected to be binarized. Mask Scorer now reflect these changes.
      - Both probe and donor reference masks are expected to be binarized. Mask Scorer now reflects these changes.
      - JournalID now changed to JournalName in reference files. Mask Scorer now reflects this change.
      - Manipulation journal tables no longer duplicate rows. Indexing problem for journal table output has been fixed.
      - Bug for query scoring corrected due to a join problem between files.
      - IsOptOut option added to Mask Scorer. Test cases also edited to reflect this change.
      - Mask binarization bug fixed. System output masks with two or less distinct colors will now be tested to see if these colors are black (0) and/or white (255).
    * DetectionScorer:
      - The d-prime metric has been added to DetectionScorer
      - Confidence level option(--ciLevel) for calculating confidence interval has been added (e.g., --ciLevel .95)
      - The number of target and non-target trials has been added to the plot legend and commands
      - The optOut option (--optOut) and its test cases have been added.
      - Based on the reference file name, DetectionScorer checks existence of JournalID/JournalName and loads the files automatically.
      - The lower and upper bound option (--dLevel) for d-prime calculation have been added
      - AUC and number of target/non-target trials have been added to the plot legend
      - The plot title and legends are changed when using the optOut option (e.g, trROC, trDET, trAUC)
    * Provenance:
      - Updated Provenance scoring test files to adhere to the latest version of the Provenance output json schemas (v 1.2)
  Mar 16, 2017 - MediScore Version 1.1.2:
    * Makefile:
      - Reorganized so that detection scorer and provenance scorer are validated before mask scorer, due to taking less time.
    * Validator:
      - Validator now prints number of channels when printing error message about masks not being single-channel.
      - Validator unit test now toggle-able with identify to hack ImageMagick behavioral discrepancies.
    * MaskScorer:
      - Score reporting and averaging bug fixed. Dummy scores were leaking through the csv and HTML.
  Mar 31, 2017 - MediScore Version 1.1.3:
    * Validator:
      - Added checker to see if ImageMagick is installed and in working order. If it is not, it will terminate the
        validator before it can run over the files.
    * MaskScorer:
      - Absolute paths added. Path dependency for the mask scorer is no longer required.
      - Bug to averaging procedure for splice portion of the mask scorer is fixed.
      - Bug regarding indexing and averaging for splice portion of the mask scorer is fixed.
    * Provenance:
      - Provenance validator and formal test cases added. Error messages corresponding to test cases with malformed system
        output should be expected.
  Apr 12, 2017 - MediScore Version 1.1.4:
    * DetectionScorer:
      - Absolute paths added. Path dependency for the detection scorer is no longer required.
    * MaskScorer:
      - Mask scoring sped up. Time taken to run has decreased by approximately 25%.
      - Other generalizations applied. Initial steps taken towards further modularization of mask scoring code.
      - System opt out option introduced. Select pixel values in the mask can be treated as no-score zones.
      - The threshold metric table for the HTML output is replaced by a plot of the MCC value per binarization threshold. If some issue crops up
        during the plotting attempt, the threshold metric table will appear instead.
    * Validator:
      - Reference file option included. Scoring for tasks can now be limited to target reference masks for considerable speedup.
    * Provenance:
      - Provenance validator has added a task option for when checking the EXPID is irrelevant.
  Apr 21, 2017 - MediScore Version 1.1.5
    * MaskScorer:
      - Option to use faster mask scorer for code stability.
    * Validator:
      - Included validation of Nimble Challenge ID.
    * ProvenanceValidator:
      - Included validation of Nimble Challenge ID.
  Apr 24, 2017 - MediScore Version 1.1.6
    * MaskScorer:
      - Added capability to read raw image files
    * ProvenanceValidator:
      - Fixed a minor typo in the error message output
      - Set default NCID to "NC17"
    * Validator:
      - Set default NCID to "NC17"
      - Added option to skip IsOptOut=='Y' rows
  Apr 26, 2017 - MediScore Version 1.1.7
    * DetectionScorer:
      - Changed the join method (left to inner) for merging the reference and index cvs file.
      - Added the plotTitle option.
      - Added the outMeta and outAllmeta options for producing the meta information along with system output.
      - Updated the column names in the test cases.
      - Fixed a bug on detcompcheckfile.sh
      - Changed the csv separation ',' to '|' for all report csv files.
    * MaskScorer:
      - Fixed plotting issue with HTML reports.
      - Added white mask scoring
      - Accounts for case in which the no-score zone covers the entire image
      - CSV outputs are now pipe-separated
      - NaN output for columns that are not scores are substituted with empty string ''
      - Re-distributed code in maskMetrics.py and maskMetrics_old.py to separate the metrics class (maskMetrics.py and maskMetrics_old.py)
        and the metric runner (metricRunner.py)
    * Provenance:
      - Updated Provenance scoring scripts to produce mapping files, optional html reports, and optional graphical mapping for the graph building task.
  Apr 28, 2017 - MediScore Version 1.1.8
    * DetectionScorer:
      - Added test cases for merging behavior between index and reference.
      - Took out OutputProbeMaskFileName and OutputDonorMaskFileName while saving the meta csv file.
    * MaskScorer:
      - Added option to score on a smaller index file for testing purposes.
      - Fixed index parsing bug.
      - Added increased capability to read images of different formats, including raw and bmp images.
      - Fixed bug related to parsing the extension of the images.
    * Validator:
      - IsOptOut column is mandated
      - Substituted range for xrange for speedup
    * ProvenanceValidator:
      - ConfidenceScore and IsOptOut columns are mandated
  May 1, 2017 - MediScore Version 1.1.9
    * DetectionScorer:
      - Added subtitle on the plot
      - Added the noNum option to not print the number of trials on the plot legend.
      - Changed the aspect of plot ratio
    * MaskScorer:
      - Added optout querying and some verbose printout for mask score partitioner.
  May 2, 2017 - MediScore Version 1.1.10
    * DetectionScorer:
      - Added “subtitle’ and ‘subtitle_fontsize” to the plot option json file
      - Added the columns “TRR” and “SYS_RESPONSE” to the report table
      - Changed the number of total data to the number of the merged data for TRR’s denominator

    * Mask Scorer:
      - Added verbose output for image dimension checking, just in case.
      - Added catcher for each iteration of mask scoring loop for runtime stability.
      - Further stabilized mask partitioner.
      - Revised partitioner querying based on query mode.
    * Validator:
      - Revised video header checking.
    * Provenance:
      - Changed column names is ProvenanceFilteringScorer to use 'At' instead of '@'
      - Added option for specifiying a thumbnails directory for ProvenanceGraphBuildingScorer when graphical output is requested with -p
      - Added cycle detection for system output provided to ProvenanceGraphBuildingScorer
      - Misc. optimizations
  May 12, 2017 - MediScore Version 1.1.11
    * Mask Scorer:
      - Revised test cases for queryPartition and queryPartition functionality.
      - Added more verbose messages to metric runner for easier error tracking.
    * Provenance:
      - Added system confidence scores to mapping output for GraphBuilding and Filtering scorers
      - Rearranged column order in HTML report output for GraphBuilding and Filtering scorers
  May 26, 2017 - MediScore Version 1.1.12
    * DetectionScorer:
      - Applied the "noNum" option for both partition and EER
    * Provenance:
      - Updated integration test runner for OS compatability
    * MaskScorer:
      - Temporarily commented out the query and queryPartition options for Mask Scorer to stabilize code.
  June 7, 2017 - MediScore Version 1.1.13
    * DetectionScorer:
      - Repaired the Splice Detection Scorer introduced in 1.1.8.  The uniqueness constraint for scoring uses both ProbeID and DonorID.
    * MaskScorer:
      - Added the tools/MaskScorer/CrossTeamLocalizationReport.py script.
      - Introduced minor speedup for confusion measures.
      - Restructured code to be more modular.
      - Introduced speedup to mask color filtering and NMM and BWL1 metrics.
      - Fixed pixel-based no-score region for grayscale mask case.
      - Shrunk test cases to facilitate more frequent testing.
      - Fixed html count for total pixels to account for pixel no-score.
      - Parallelization option and features added.
      - Fixed plotting bug aggregating all plots onto each subsequent plot.
      - Added atomic printout per mask generated.
      - Minor edit to HTML template.
      - Minor fixes.
    * Provenance:
      - Updated integration test runner for OS compatability
  June 7, 2017 - MediScore Version 1.1.13
    * Mask Scorer:
      - Fixed minor bug in error printout.
  June 19, 2017
    * Mask Scorer:
      - Added confusion measures and no-score pixel counts to extensive report.
      - Fixed minor typo in HTML template.
      - Added outMeta and outAllmeta options for some control of reporting detail.
      - Added TRR and related figures to the per trial data file.
      - Changed --optOut behavior to more closely match that of Detection Scorer --optOut behavior.
      - Updated test cases to reflect above changes.
      - Moved the print buffer class to a separate file for use by other modules that may need parallelization.
      - Added displayScoredOnly to display scored images only.
      - Added -1 to a list of thresholds to binarize everything to white.
    * Validator:
      - Sped up validator with pandas apply implementation.
      - Parallelization options and features added.
  August 1, 2017
    * Mask Scorer:
      - Redirected control of selective regions to mask object to prepare for introduction of JPEG2000 bit-mask references.
      - Added animated png creation function to mask object for JPEG2000 bit-mask references.
      - HTML output now supports animated png's and multiple ways of computing the metrics involved.
      - Added Maximum Threshold metrics and Threshold averaging.
      - Added ROC curve averages.
      - Added standard deviation for metrics.
      - Added reading functionality for empty reference masks.
    * Detection Scorer:
    * Validator:
      - Reverting splice validator to serial dictionary implementation for stability.
      - Added confidence score checking to make sure the confidence score is a real number.
      - Added additional optOut functionality (the code is still backwards compatible)
    * Provenance Validator:
      - Reworded the -task option in the ReadMe for clarity.
  Sep 25, 2017 - MediScore Version 1.1.13
    * DetectionScorer:
      - cleaned up the test cases and added exclusions for “.DS_Store” and PDF files
    * MaskScorer:
      - Output reformatting.
      - File prefix added to --outRoot option as a requirement.
      - Added per-probe pixel no-score
      - Averaging functionality now treats constant metrics (e.g. ActualThreshold) differently.
  Nov 1, 2017 - MediScore Version 1.1.14
    * DetectionScorer:
      - reimplemented the merging process
      - added ProbeStatus with ['Processed', 'NonProcessed', 'OptOutAll', 'OptOutDetection', 'OptOutLocalization'] for Optout options and kept the old version of the IsOptOut column process
      - added test cases:
        1) Testing with the manipulation OptOut case -- IsOptOut
        2) Testing a query with the manipulation OptOut case -- IsOptOut
        3) Testing with the manipulation OptOut case -- ProbeStatus
      - added all the examples (16 test cases) from the DetectionScorer ReadMe document to make check
      - Fixed plot leakage issue for rendering.
    * Mask Scorer
      - Renamed Mask Scorer test cases to conform to new format
      - Added -1 as a threshold to binarize masks to all white where relevant.
      - Expanded test cases to cover optOut criteria for code.
  Nov 20, 2017 - MediScore Version 1.2.0
    * DetectionScorer:
      - Added the Correct Detection Rate at False Alarm Rate into the results
      - Regenerated all the test cases
    * Provenance Scorers:
      - Added support for opt out in ProvenanceProbeStatus
  Feb 7, 2018 - MediScore Version *.*.*
    * DetectionScorer:
      - Changed to ProbeStatus==['Processed', 'NonProcessed', 'OptOutLocalization']
      - Modified test cases for OptOut
      - Updated test cases to use "python2" instead of "python" to remove ambiguity.
    * Mask Scorer
      - Added more thorough tests for ProbeStatus criteria.
      - Changed help message for -xF to be in line with updates to the validator.
      - Fixed bug that treated -1 as blank ProbeOptOutPixelValue. Blank now ignores opting out of pixel.
      - Fixed rounding issues with new pandas version.
      - Now reading cr2 with rawpy instead of OpenCV.
      - Added option to turn off "debug mode" and added a test case.
      - Extended precision control to TRR and standard deviations for the average metrics.
         - Updated test cases accordingly for bitmask tests, specifically for precision control for AUC and EER.
      - Adjusted unit test png saving parameter to be compatible with OpenCV 3+, just in case.
      - Streamlined localization scorer for efficient pandas usage.
         - Simplified saving the reference png.
      - Vectorized metrics computation.
    * Validator
      - Updated ReadMe for readability.
      - Added more thorough tests for ProbeStatus criteria.
      - Simplified and edited error messsages to be more helpful.
      - Fixed bug that was reporting ground truth negatives.
      - Fixed bug that skipped masks when namechecking is turned off.
      - Fixed error message reporting for validator mask checker.
      - Added Probe and DonorOptOutPixelValue checks for validator.
      - Checks for image readability as png after verifying file name.
      - Fixed video validator and adding video cases.
      - Adjusted video validator parameter to be compatible with OpenCV 3+, just in case.
      - Added optOut option to skip opted out rows where relevant.
    * Video Temporal Localization Scorer
      - Initial release
    * Provenance Scorers:
      - Fixed rounding issues with new numpy version
  Feb 28, 2018 - MediScore Version *.*.*
    * DetectionScorer:
      - Added EventRepurpose task and its test case
      - Modified DetectionScoreReadMe files
    * MaskScorer:
      - Additional misc speedups
      - Vectorized maximum metrics computation.
      - Refactored mask no-score function to take in less arguments.
      - Added option to cache no-score zones and others in a directory.
    * Provenance Validator:
      - Fixed a bug that only changed the column name when the optOut option was selected.


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
David Joy
August Pereira

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
