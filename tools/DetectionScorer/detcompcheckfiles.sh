#!/bin/bash
clean=TRUE

Rscript DetectionScorer.r -i file -t removal -d ../../data/test_suite/detectionScorerTests -r NC2016-manipulation-ref.csv -x NC2016-manipulation-index.csv -s ../../data/test_suite/detectionScorerTests/D_NC2016_Manipulation_ImgOnly_p-me_1/D_NC2016_Manipulation_ImgOnly_p-me_1.csv -o ../../data/test_suite/detectionScorerTests/temp_detreport.csv -p plot.pdf > res_test_detscr.txt-temp && rm plot.pdf
diff res_test_detscr.txt-temp ../../data/test_suite/detectionScorerTests/res_test_detctscr.txt > res_test_detscr.txt-comp
diff ../../data/test_suite/detectionScorerTests/temp_detreport.csv ../../data/test_suite/detectionScorerTests/ref_detreport.csv > comp_detreport.csv

filter="diff res_test_detscr.txt-temp ../../data/test_suite/detectionScorerTests/res_test_detctscr.txt | grep -v CVS"
filter2="diff ../../data/test_suite/detectionScorerTests/temp_detreport.csv ../../data/test_suite/detectionScorerTests/ref_detreport.csv | grep -v CVS"
flag1=1
flag2=1
if test "`eval $filter`" = "" ; then
	flag1=0
	if [ $clean = "TRUE" ] ; then
		rm res_test_detscr.txt-temp
	fi
	rm res_test_detscr.txt-comp
else
	flag1=1
	cat res_test_detscr.txt-comp
fi
if test "`eval $filter2`" = "" ; then
	flag2=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/detectionScorerTests/temp_detreport.csv
	fi
	rm comp_detreport.csv
else
	flag2=1
	cat comp_detreport.csv
fi
if ([ $flag1 == 0 -a $flag2 == 0 ]); then
  echo
	echo "DETECTION SCORER TESTS SUCCESSFULLY PASSED."
	echo
else
  echo
	echo "    !!!!! DETECTION SCORER TESTS FAILED !!!!!"
	echo
fi
