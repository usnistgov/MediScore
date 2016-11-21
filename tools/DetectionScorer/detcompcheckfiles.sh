#!/bin/bash
clean=TRUE

python2 DetectionScorer.py -t manipulation --refDir ../../data/test_suite/detectionScorerTests/sample -r NC2016-manipulation-ref.csv -x NC2016-manipulation-index.csv --sysDir ../../data/test_suite/detectionScorerTests/sample -s D_NC2016_Manipulation_ImgOnly_p-me_1/D_NC2016_Manipulation_ImgOnly_p-me_1.csv --outRoot ./testcases/NC16_01

python2 DetectionScorer.py -t manipulation --refDir ../../data/test_suite/detectionScorerTests/sample -r NC2016-manipulation-ref.csv -x NC2016-manipulation-index.csv --sysDir ../../data/test_suite/detectionScorerTests/sample -s D_NC2016_Manipulation_ImgOnly_p-me_1/D_NC2016_Manipulation_ImgOnly_p-me_1.csv --outRoot ./testcases/NC16_01_ci --ci

python2 DetectionScorer.py -t splice --refDir ../../data/test_suite/detectionScorerTests/sample -r NC2016-splice-ref.csv -x NC2016-splice-index.csv --sysDir ../../data/test_suite/detectionScorerTests/sample -s D_NC2016_Splice_ImgOnly_p-me_1/D_NC2016_Splice_ImgOnly_p-me_1.csv --outRoot ./testcases/NC16_02

diff testcases/NC16_01_all.csv ../../data/test_suite/detectionScorerTests/sample/ref_detreport.csv > comp_detreport.txt
diff testcases/NC16_01_ci_all.csv ../../data/test_suite/detectionScorerTests/sample/ref_detreport_ci.csv > comp_detreport_ci.txt
diff testcases/NC16_02_all.csv ../../data/test_suite/detectionScorerTests/sample/ref_detreport2.csv > comp_detreport2.txt

filter="cat comp_detreport.txt | grep -v CVS"
filter_ci="cat comp_detreport_ci.txt | grep -v CVS"
filter2="cat comp_detreport2.txt | grep -v CVS"
flag=1
flag_ci=1
flag2=1

if ([ ! -e comp_detreport.txt -o ! -e comp_detreport_ci.txt -o ! -e comp_detreport2.txt ]); then
  echo
  echo "    !!!!! DETECTION SCORER TESTS FAILED !!!!!    "
  echo
  exit
fi

if test "`eval $filter`" = "" ; then
        flag=0
	if [ $clean = "TRUE" ] ; then
		rm testcases/NC16_01_roc_all.pdf
		rm -rf plotJsonFiles
	fi
	rm comp_detreport.txt
else
	cat comp_detreport.txt
fi
if test "`eval $filter_ci`" = "" ; then
        flag_ci=0
	if [ $clean = "TRUE" ] ; then
		rm testcases/NC16_01_ci_roc_all.pdf
		rm -rf plotJsonFiles
	fi
	rm comp_detreport_ci.txt
else
	cat comp_detreport_ci.txt
fi
if test "`eval $filter2`" = "" ; then
        flag2=0
	if [ $clean = "TRUE" ] ; then
		rm testcases/NC16_02_roc_all.pdf
		rm -rf plotJsonFiles
	fi
	rm comp_detreport2.txt
else
	cat comp_detreport2.txt
fi

if [ $flag == 0 -a $flag_ci == 0 -a $flag2 == 0 ] ; then
	echo
	echo "DETECTION SCORER TESTS SUCCESSFULLY PASSED."
	echo
else
	rm -rf testcases
	echo
	echo "    !!!!! DETECTION SCORER TESTS FAILED !!!!!"
	echo
fi
