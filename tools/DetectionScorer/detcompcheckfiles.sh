#!/bin/bash
clean=TRUE

echo
echo "BEGINNING FUNCTIONALITY TEST OF DETECTION SCORER"

echo
echo "CASE 0: VALIDATING FULL SCORING WITH BASELINEs"
echo

echo "Testing NC2016 Manipulation"
python2 DetectionScorer.py -t manipulation --refDir ../../data/test_suite/detectionScorerTests/reference -r NC2016-manipulation-ref.csv -x NC2016-manipulation-index.csv --sysDir ../../data/test_suite/detectionScorerTests/baseline -s Base_NC2016_Manipulation_ImgOnly_p-dct_02.csv --outRoot ./testcases/NC16_C0_01
echo
echo "Testing NC2016 Splice"
python2 DetectionScorer.py -t splice --refDir ../../data/test_suite/detectionScorerTests/reference -r NC2016-splice-ref.csv -x NC2016-splice-index.csv --sysDir ../../data/test_suite/detectionScorerTests/baseline -s Base_NC2016_Splice_ImgOnly_p-splice_01.csv --outRoot ./testcases/NC16_C0_02
echo
echo "Testing NC2017 Manipulation"
python2 DetectionScorer.py -t manipulation --refDir ../../data/test_suite/detectionScorerTests/reference -r NC2017-manipulation-ref.csv -x NC2017-manipulation-index.csv --sysDir ../../data/test_suite/detectionScorerTests/baseline -s Base_NC2017_Manipulation_ImgOnly_p-copymove_01.csv --outRoot ./testcases/NC17_C0_01

diff testcases/NC16_C0_01_all_report.csv ../../data/test_suite/detectionScorerTests/sample/NC16_C0_01_all_test.csv > comp_NC16_C0_01_all.txt
diff testcases/NC16_C0_02_all_report.csv ../../data/test_suite/detectionScorerTests/sample/NC16_C0_02_all_test.csv > comp_NC16_C0_02_all.txt
diff testcases/NC17_C0_01_all_report.csv ../../data/test_suite/detectionScorerTests/sample/NC17_C0_01_all_test.csv > comp_NC17_C0_01_all.txt

c0_res1="cat comp_NC16_C0_01_all.txt | grep -v CVS"
c0_res2="cat comp_NC16_C0_02_all.txt | grep -v CVS"
c0_res3="cat comp_NC17_C0_01_all.txt | grep -v CVS"
c0_flag1=1
c0_flag2=1
c0_flag3=1

if ([ ! -e comp_NC16_C0_01_all.txt -o ! -e comp_NC16_C0_02_all.txt -o ! -e comp_NC17_C0_01_all.txt ]); then
  echo
  echo "DETECTION SCORER TESTS FAILED FOR CASE 0 !!! "
  echo
  exit
fi

if test "`eval $c0_res1`" = "" ; then
  c0_flag1=0
	rm comp_NC16_C0_01_all.txt
else
	cat comp_NC16_C0_01_all.txt
fi

if test "`eval $c0_res2`" = "" ; then
  c0_flag2=0
	rm comp_NC16_C0_02_all.txt
else
	cat comp_NC16_C0_02_all.txt
fi

if test "`eval $c0_res3`" = "" ; then
  c0_flag3=0
	rm comp_NC17_C0_01_all.txt
else
	cat comp_NC17_C0_01_all.txt
fi

if [ $c0_flag1 == 0 -a $c0_flag2 == 0 -a $c0_flag3 == 0 ] ; then
	echo
	echo "DETECTION SCORER TESTS SUCCESSFULLY PASSED FOR CASE 0."
	echo
  if [ $clean = "TRUE" ] ; then
    rm -rf testcases
		rm -rf plotJsonFiles
	fi
else
	echo
	echo "DETECTION SCORER TESTS FAILED FOR CASE 0 !!!"
	echo
fi


echo
echo "CASE 1: VALIDATING SYSTEM OUTPUT SCORING TESTCASES"
echo
echo "Testing with the manipulation case"
python2 DetectionScorer.py -t manipulation --refDir ../../data/test_suite/detectionScorerTests/sample/reference -r NC2016-manipulation-ref.csv -x NC2016-manipulation-index.csv --sysDir ../../data/test_suite/detectionScorerTests/sample -s D_NC2016_Manipulation_ImgOnly_p-me_1/D_NC2016_Manipulation_ImgOnly_p-me_1.csv --outRoot ./testcases/NC16_C1_01
echo

echo "Testing with the splice case"
python2 DetectionScorer.py -t splice --refDir ../../data/test_suite/detectionScorerTests/sample/reference -r NC2016-splice-ref.csv -x NC2016-splice-index.csv --sysDir ../../data/test_suite/detectionScorerTests/sample -s D_NC2016_Splice_ImgOnly_p-me_1/D_NC2016_Splice_ImgOnly_p-me_1.csv --outRoot ./testcases/NC16_C1_02
echo

echo "Testing with the same scores across all image files"
python2 DetectionScorer.py -t manipulation --refDir ../../data/test_suite/detectionScorerTests/sample/reference -r NC2016-manipulation-ref.csv -x NC2016-manipulation-index.csv --sysDir ../../data/test_suite/detectionScorerTests/sample -s D_NC2016_Manipulation_ImgOnly_p-me_2/D_NC2016_Manipulation_ImgOnly_p-me_2.csv --outRoot ./testcases/NC16_C1_03
echo

echo "Testing with no non-target value"
python2 DetectionScorer.py -t manipulation --refDir ../../data/test_suite/detectionScorerTests/sample/reference -r NC2017-manipulation-ref.csv -x NC2017-manipulation-index.csv --sysDir ../../data/test_suite/detectionScorerTests/sample -s D_NC2017_Manipulation_ImgOnly_c-me_2/D_NC2017_Manipulation_ImgOnly_c-me_2.csv --outRoot ./testcases/NC17_C1_04
echo

echo "Testing with one target and one non-target trial"
python2 DetectionScorer.py -t manipulation --refDir ../../data/test_suite/detectionScorerTests/sample/reference -r NC2017-manipulation-ref2.csv -x NC2017-manipulation-index.csv --sysDir ../../data/test_suite/detectionScorerTests/sample -s D_NC2017_Manipulation_ImgOnly_c-me_2/D_NC2017_Manipulation_ImgOnly_c-me_2.csv --outRoot ./testcases/NC17_C1_05
echo

echo "Testing with the manipulation OptOut case"
python2 DetectionScorer.py -t manipulation --refDir ../../data/test_suite/detectionScorerTests/sample/reference -r NC2016-manipulation-ref.csv -x NC2016-manipulation-index.csv --sysDir ../../data/test_suite/detectionScorerTests/sample -s D_NC2016_Manipulation_ImgOnly_p-me_3/D_NC2016_Manipulation_ImgOnly_p-me_3.csv --outRoot ./testcases/NC16_C1_06 --optOut
echo

diff testcases/NC16_C1_01_all_report.csv ../../data/test_suite/detectionScorerTests/sample/NC16_C1_01_all_test.csv > comp_NC16_C1_01_all.txt
diff testcases/NC16_C1_02_all_report.csv ../../data/test_suite/detectionScorerTests/sample/NC16_C1_02_all_test.csv > comp_NC16_C1_02_all.txt
diff testcases/NC16_C1_03_all_report.csv ../../data/test_suite/detectionScorerTests/sample/NC16_C1_03_all_test.csv > comp_NC16_C1_03_all.txt
diff testcases/NC17_C1_04_all_report.csv ../../data/test_suite/detectionScorerTests/sample/NC17_C1_04_all_test.csv > comp_NC17_C1_04_all.txt
diff testcases/NC17_C1_05_all_report.csv ../../data/test_suite/detectionScorerTests/sample/NC17_C1_05_all_test.csv > comp_NC17_C1_05_all.txt
diff testcases/NC16_C1_06_all_report.csv ../../data/test_suite/detectionScorerTests/sample/NC16_C1_06_all_test.csv > comp_NC16_C1_06_all.txt

c1_res1="cat comp_NC16_C1_01_all.txt | grep -v CVS"
c1_res2="cat comp_NC16_C1_02_all.txt | grep -v CVS"
c1_res3="cat comp_NC16_C1_03_all.txt | grep -v CVS"
c1_res4="cat comp_NC17_C1_04_all.txt | grep -v CVS"
c1_res5="cat comp_NC17_C1_05_all.txt | grep -v CVS"
c1_res6="cat comp_NC16_C1_06_all.txt | grep -v CVS"

c1_flag1=1
c1_flag2=1
c1_flag3=1
c1_flag4=1
c1_flag5=1
c1_flag6=1

if ([ ! -e comp_NC16_C1_01_all.txt -o ! -e comp_NC16_C1_02_all.txt -o ! -e comp_NC16_C1_03_all.txt -o ! -e comp_NC17_C1_04_all.txt -o ! -e comp_NC17_C1_05_all.txt -o ! -e comp_NC16_C1_06_all.txt ]); then
  echo
  echo "DETECTION SCORER TESTS FAILED FOR CASE 1 !!! "
  echo
  exit
fi

if test "`eval $c1_res1`" = "" ; then
  c1_flag1=0
	rm comp_NC16_C1_01_all.txt
else
	cat comp_NC16_C1_01_all.txt
fi

if test "`eval $c1_res2`" = "" ; then
  c1_flag2=0
	rm comp_NC16_C1_02_all.txt
else
	cat comp_NC16_C1_02_all.txt
fi

if test "`eval $c1_res3`" = "" ; then
  c1_flag3=0
	rm comp_NC16_C1_03_all.txt
else
	cat comp_NC16_C1_03_all.txt
fi

if test "`eval $c1_res4`" = "" ; then
  c1_flag4=0
	rm comp_NC17_C1_04_all.txt
else
	cat comp_NC17_C1_04_all.txt
fi

if test "`eval $c1_res5`" = "" ; then
  c1_flag5=0
	rm comp_NC17_C1_05_all.txt
else
	cat comp_NC17_C1_05_all.txt
fi

if test "`eval $c1_res6`" = "" ; then
  c1_flag6=0
	rm comp_NC16_C1_06_all.txt
else
	cat comp_NC16_C1_06_all.txt
fi

if ([ $c1_flag1 == 0 -a $c1_flag2 == 0 -a $c1_flag3 == 0 -a $c1_flag4 == 0 -a $c1_flag5 == 0 -a $c1_flag6 == 0 ]) ; then
	echo
	echo "DETECTION SCORER TESTS SUCCESSFULLY PASSED FOR CASE 1."
	echo
  if [ $clean = "TRUE" ] ; then
    rm -rf testcases
		rm -rf plotJsonFiles
	fi
else
	echo
	echo "DETECTION SCORER TESTS FAILED FOR CASE 1!!!"
	echo
fi


#echo
#echo "CASE 2: VALIDATING QUERY-BASED SCORING TESTCASES"
#echo

echo
echo "CASE 2: VALIDATING FULL INDEX and SUBSET INDEX"
echo

echo "Testing with the manipulation case with full index"
python2 DetectionScorer.py -t manipulation --refDir ../../data/test_suite/detectionScorerTests/sample/reference -r NC2016-manipulation-ref.csv -x NC2016-manipulation-index.csv --sysDir ../../data/test_suite/detectionScorerTests/sample -s D_NC2016_Manipulation_ImgOnly_p-me_1/D_NC2016_Manipulation_ImgOnly_p-me_1.csv --outRoot ./testcases/NC16_C1_01
echo

echo "Testing with the manipulation case with sub index (1 less)"
python2 DetectionScorer.py -t manipulation --refDir ../../data/test_suite/detectionScorerTests/sample/reference -r NC2016-manipulation-ref.csv -x NC2016-manipulation-index_sub.csv --sysDir ../../data/test_suite/detectionScorerTests/sample -s D_NC2016_Manipulation_ImgOnly_p-me_1/D_NC2016_Manipulation_ImgOnly_p-me_1.csv --outRoot ./testcases/NC16_C1_01_01
echo

echo "Testing with the splice case with full index"
python2 DetectionScorer.py -t splice --refDir ../../data/test_suite/detectionScorerTests/sample/reference -r NC2016-splice-ref.csv -x NC2016-splice-index.csv --sysDir ../../data/test_suite/detectionScorerTests/sample -s D_NC2016_Splice_ImgOnly_p-me_1/D_NC2016_Splice_ImgOnly_p-me_1.csv --outRoot ./testcases/NC16_C1_02
echo

echo "Testing with the splice case with sub index (2 less)"
python2 DetectionScorer.py -t splice --refDir ../../data/test_suite/detectionScorerTests/sample/reference -r NC2016-splice-ref.csv -x NC2016-splice-index_sub.csv --sysDir ../../data/test_suite/detectionScorerTests/sample -s D_NC2016_Splice_ImgOnly_p-me_1/D_NC2016_Splice_ImgOnly_p-me_1.csv --outRoot ./testcases/NC16_C1_02_01
echo

diff testcases/NC16_C1_01_01_all_report.csv ../../data/test_suite/detectionScorerTests/sample/NC16_C1_01_01_all_test.csv > comp_NC16_C1_01_01_all.txt
diff testcases/NC16_C1_02_01_all_report.csv ../../data/test_suite/detectionScorerTests/sample/NC16_C1_02_01_all_test.csv > comp_NC16_C1_02_01_all.txt


c1_res7="cat comp_NC16_C1_01_01_all.txt | grep -v CVS"
c1_res8="cat comp_NC16_C1_02_01_all.txt | grep -v CVS"

c1_flag7=1
c1_flag8=1

if ([ ! -e comp_NC16_C1_01_01_all.txt -o ! -e comp_NC16_C1_02_01_all.txt ]); then
  echo
  echo "DETECTION SCORER TESTS FAILED FOR CASE 2 !!! "
  echo
  exit
fi

if test "`eval $c1_res7`" = "" ; then
  c1_flag7=0
	rm comp_NC16_C1_01_01_all.txt
else
	cat comp_NC16_C1_01_01_all.txt
fi

if test "`eval $c1_res8`" = "" ; then
  c1_flag8=0
	rm comp_NC16_C1_02_01_all.txt
else
	cat comp_NC16_C1_02_01_all.txt
fi


if ([ $c1_flag7 == 0 -a $c1_flag8 == 0 ]) ; then
	echo
	echo "DETECTION SCORER TESTS SUCCESSFULLY PASSED FOR CASE 2."
	echo
    echo
	echo "All DETECTION SCORER TESTS SUCCESSFULLY PASSED!!!"
	echo
  if [ $clean = "TRUE" ] ; then
    rm -rf testcases
		rm -rf plotJsonFiles
	fi
else
	echo
	echo "DETECTION SCORER TESTS FAILED FOR CASE 2!!!"
	echo
fi


