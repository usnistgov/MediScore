#!/bin/bash
clean=TRUE

Rscript MaskScorer.r -i file -t removal -d ../../data/test_suite/maskScorerTests -r NC2016-removal-ref.csv -x NC2016-removal-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2016_Removal_ImgOnly_p-me_1/B_NC2016_Removal_ImgOnly_p-me_1.csv --maskout n -o ../../data/test_suite/maskScorerTests/temp_maskreport_1.csv > res_test_maskscr.txt-temp && rm ../../data/test_suite/maskScorerTests/temp_maskreport_1_avg.csv
echo >> res_test_maskscr.txt-temp
Rscript MaskScorer.r -i file -t removal -d ../../data/test_suite/maskScorerTests -r NC2016-removal-ref.csv -x NC2016-removal-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2016_Removal_ImgOnly_c-me2_1/B_NC2016_Removal_ImgOnly_c-me2_1.csv --maskout n -o ../../data/test_suite/maskScorerTests/temp_maskreport_2.csv >> res_test_maskscr.txt-temp && rm ../../data/test_suite/maskScorerTests/temp_maskreport_2_avg.csv
echo >> res_test_maskscr.txt-temp
Rscript MaskScorer.r -i file -t removal -d ../../data/test_suite/maskScorerTests -r NC2016-removal-ref.csv -x NC2016-removal-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2016_Removal_ImgOnly_c-me2_2/B_NC2016_Removal_ImgOnly_c-me2_2.csv --maskout n -o ../../data/test_suite/maskScorerTests/temp_maskreport_3.csv >> res_test_maskscr.txt-temp && rm ../../data/test_suite/maskScorerTests/temp_maskreport_3_avg.csv
diff res_test_maskscr.txt-temp ../../data/test_suite/maskScorerTests/res_test_maskscr.txt > res_test_maskscr.txt-comp
diff ../../data/test_suite/maskScorerTests/temp_maskreport_1.csv ../../data/test_suite/maskScorerTests/ref_maskreport_1.csv >> comp_maskreport_1.txt
diff ../../data/test_suite/maskScorerTests/temp_maskreport_2.csv ../../data/test_suite/maskScorerTests/ref_maskreport_2.csv >> comp_maskreport_2.txt
diff ../../data/test_suite/maskScorerTests/temp_maskreport_3.csv ../../data/test_suite/maskScorerTests/ref_maskreport_3.csv >> comp_maskreport_3.txt

flag=1
flag1=1
flag2=1
flag3=1

filter="diff res_test_maskscr.txt-temp ../../data/test_suite/maskScorerTests/res_test_maskscr.txt | grep -v CVS"
filter1="diff ../../data/test_suite/maskScorerTests/temp_maskreport_1.csv ../../data/test_suite/maskScorerTests/ref_maskreport_1.csv | grep -v CVS"
filter2="diff ../../data/test_suite/maskScorerTests/temp_maskreport_2.csv ../../data/test_suite/maskScorerTests/ref_maskreport_2.csv | grep -v CVS"
filter3="diff ../../data/test_suite/maskScorerTests/temp_maskreport_3.csv ../../data/test_suite/maskScorerTests/ref_maskreport_3.csv | grep -v CVS"
if test "`eval $filter`" = "" ; then
  flag=0
	if [ $clean = "TRUE" ] ; then
		rm res_test_maskscr.txt-temp
	fi
	rm res_test_maskscr.txt-comp
else
  flag=1
	echo cat res_test_maskscr.txt-comp
fi
if test "`eval $filter1`" = "" ; then
  flag1=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/temp_maskreport_1.csv
	fi
	rm comp_maskreport_1.txt
else
  flag1=1
	echo cat comp_maskreport_1.txt
fi
if test "`eval $filter2`" = "" ; then
  flag2=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/temp_maskreport_2.csv
	fi
	rm comp_maskreport_2.txt
else
  flag2=1
	echo cat comp_maskreport_2.txt
fi
if test "`eval $filter3`" = "" ; then
  flag3=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/temp_maskreport_3.csv
	fi
	rm comp_maskreport_3.txt
else
  flag3=1
	echo cat comp_maskreport_3.txt
fi
if ([ $flag == 0 -a $flag1 == 0 -a $flag2 == 0 -a $flag3 == 0 ]); then
  echo
  echo "MASK SCORER TESTS SUCCESSFULLY PASSED"
  echo
else
  echo
  echo "    !!!!! MASK SCORER TESTS FAILED !!!!!    "
  cat comp_maskreport_3.txt
  echo
fi
