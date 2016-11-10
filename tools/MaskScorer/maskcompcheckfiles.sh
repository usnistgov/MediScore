#!/bin/bash
clean=TRUE

#produce the output files
python2 MaskScorer.py -t splice --refDir ../../data/test_suite/maskScorerTests -r NC2016-splice-ref.csv -x NC2016-splice-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2016_Splice_ImgOnly_p-me_1/B_NC2016_Splice_ImgOnly_p-me_1.csv -oR ../../data/test_suite/maskScorerTests/temp_maskreport_1
python2 MaskScorer.py -t manipulation --refDir ../../data/test_suite/maskScorerTests -r NC2016-manipulation-ref.csv -x NC2016-manipulation-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2016_Manipulation_ImgOnly_c-me2_1/B_NC2016_Manipulation_ImgOnly_c-me2_1.csv -oR ../../data/test_suite/maskScorerTests/temp_maskreport_2
python2 MaskScorer.py -t removal --refDir ../../data/test_suite/maskScorerTests -r NC2016-removal-ref.csv -x NC2016-removal-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2016_Removal_ImgOnly_c-me2_2/B_NC2016_Removal_ImgOnly_c-me2_2.csv -oR ../../data/test_suite/maskScorerTests/temp_maskreport_3

#compare them to ground truth files
diff ../../data/test_suite/maskScorerTests/temp_maskreport_1.csv ../../data/test_suite/maskScorerTests/ref_maskreport_1.csv >> comp_maskreport_1.txt
diff ../../data/test_suite/maskScorerTests/temp_maskreport_2.csv ../../data/test_suite/maskScorerTests/ref_maskreport_2.csv >> comp_maskreport_2.txt
diff ../../data/test_suite/maskScorerTests/temp_maskreport_3.csv ../../data/test_suite/maskScorerTests/ref_maskreport_3.csv >> comp_maskreport_3.txt
diff ../../data/test_suite/maskScorerTests/temp_maskreport_1-perimage.csv ../../data/test_suite/maskScorerTests/ref_maskreport_1-perimage.csv >> comp_maskreport_1-perimage.txt
diff ../../data/test_suite/maskScorerTests/temp_maskreport_2-perimage.csv ../../data/test_suite/maskScorerTests/ref_maskreport_2-perimage.csv >> comp_maskreport_2-perimage.txt
diff ../../data/test_suite/maskScorerTests/temp_maskreport_3-perimage.csv ../../data/test_suite/maskScorerTests/ref_maskreport_3-perimage.csv >> comp_maskreport_3-perimage.txt

flag1=1
flag2=1
flag3=1
flag1_2=1
flag2_2=1
flag3_2=1

filter1="cat comp_maskreport_1.txt | grep -v CVS"
filter2="cat comp_maskreport_2.txt | grep -v CVS"
filter3="cat comp_maskreport_3.txt | grep -v CVS"
filter1_2="cat comp_maskreport_1-perimage.txt | grep -v CVS"
filter2_2="cat comp_maskreport_2-perimage.txt | grep -v CVS"
filter3_2="cat comp_maskreport_3-perimage.txt | grep -v CVS"

if test "`eval $filter1`" = "" ; then
  flag1=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/temp_maskreport_1.csv
	fi
	rm comp_maskreport_1.txt
else
	echo cat comp_maskreport_1.txt
	cat comp_maskreport_1.txt
fi
if test "`eval $filter1_2`" = "" ; then
  flag1_2=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/temp_maskreport_1-perimage.csv
	fi
	rm comp_maskreport_1-perimage.txt
else
	echo cat comp_maskreport_1-perimage.txt
	cat comp_maskreport_1-perimage.txt
fi
if test "`eval $filter2`" = "" ; then
  flag2=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/temp_maskreport_2.csv
	fi
	rm comp_maskreport_2.txt
else
	echo cat comp_maskreport_2.txt
	cat comp_maskreport_2.txt
fi
if test "`eval $filter2_2`" = "" ; then
  flag2_2=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/temp_maskreport_2-perimage.csv
	fi
	rm comp_maskreport_2-perimage.txt
else
	echo cat comp_maskreport_2-perimage.txt
	cat comp_maskreport_2-perimage.txt
fi
if test "`eval $filter3`" = "" ; then
  flag3=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/temp_maskreport_3.csv
	fi
	rm comp_maskreport_3.txt
else
	echo cat comp_maskreport_3.txt
	cat comp_maskreport_3.txt
fi
if test "`eval $filter3_2`" = "" ; then
  flag3_2=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/temp_maskreport_3-perimage.csv
	fi
	rm comp_maskreport_3-perimage.txt
else
	echo cat comp_maskreport_3-perimage.txt
	cat comp_maskreport_3-perimage.txt
fi
if ([ $flag1 == 0 -a $flag1_2 == 0 -a $flag2 == 0 -a $flag2_2 == 0 -a $flag3 == 0 -a $flag3_2 == 0 ]); then
  echo
  echo "MASK SCORER TESTS SUCCESSFULLY PASSED"
  echo
else
  echo
  echo "    !!!!! MASK SCORER TESTS FAILED !!!!!    "
  echo
fi
