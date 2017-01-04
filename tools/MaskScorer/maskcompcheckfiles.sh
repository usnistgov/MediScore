#!/bin/bash
clean=TRUE

#produce the output files
python2 MaskScorer.py -t splice --refDir ../../data/test_suite/maskScorerTests -r reference/splice/NC2016-splice-ref.csv -x index/NC2016-splice-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2016_Splice_ImgOnly_p-me_1/B_NC2016_Splice_ImgOnly_p-me_1.csv -oR ../../data/test_suite/maskScorerTests
python2 MaskScorer.py -t manipulation --refDir ../../data/test_suite/maskScorerTests -r reference/manipulation/NC2016-manipulation-ref.csv -x index/NC2016-manipulation-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2016_Manipulation_ImgOnly_c-me2_1/B_NC2016_Manipulation_ImgOnly_c-me2_1.csv -oR ../../data/test_suite/maskScorerTests
#python2 MaskScorer.py -t removal --refDir ../../data/test_suite/maskScorerTests -r reference/removal/NC2016-removal-ref.csv -x index/NC2016-removal-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2016_Removal_ImgOnly_c-me2_2/B_NC2016_Removal_ImgOnly_c-me2_2.csv -oR ../../data/test_suite/maskScorerTests/temp_maskreport_3 --sbin 127

#compare them to ground truth files
diff ../../data/test_suite/maskScorerTests/B_NC2016_Splice_ImgOnly_p-me_1-mask_score.csv ../../data/test_suite/maskScorerTests/ref_maskreport_splice.csv > comp_maskreport_splice.txt
diff ../../data/test_suite/maskScorerTests/B_NC2016_Splice_ImgOnly_p-me_1-mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/ref_maskreport_splice-perimage.csv > comp_maskreport_splice-perimage.txt
diff ../../data/test_suite/maskScorerTests/B_NC2016_Manipulation_ImgOnly_c-me2_1-mask_score.csv ../../data/test_suite/maskScorerTests/ref_maskreport_manip.csv > comp_maskreport_manip.txt
diff ../../data/test_suite/maskScorerTests/B_NC2016_Manipulation_ImgOnly_c-me2_1-mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/ref_maskreport_manip-perimage.csv > comp_maskreport_manip-perimage.txt
diff ../../data/test_suite/maskScorerTests/B_NC2016_Manipulation_ImgOnly_c-me2_1-journalResults.csv ../../data/test_suite/maskScorerTests/ref_maskreport_manip-journalResults.csv > comp_maskreport_manip-journalResults.txt

#TODO: update the testing script accordingly 


diff ../../data/test_suite/maskScorerTests/temp_maskreport_1.csv ../../data/test_suite/maskScorerTests/ref_maskreport_1.csv > comp_maskreport_1.txt
diff ../../data/test_suite/maskScorerTests/temp_maskreport_2.csv ../../data/test_suite/maskScorerTests/ref_maskreport_2.csv > comp_maskreport_2.txt
diff ../../data/test_suite/maskScorerTests/temp_maskreport_3.csv ../../data/test_suite/maskScorerTests/ref_maskreport_3.csv > comp_maskreport_3.txt
diff ../../data/test_suite/maskScorerTests/temp_maskreport_1-perimage.csv ../../data/test_suite/maskScorerTests/ref_maskreport_1-perimage.csv > comp_maskreport_1-perimage.txt
diff ../../data/test_suite/maskScorerTests/temp_maskreport_2-perimage.csv ../../data/test_suite/maskScorerTests/ref_maskreport_2-perimage.csv > comp_maskreport_2-perimage.txt
diff ../../data/test_suite/maskScorerTests/temp_maskreport_3-perimage.csv ../../data/test_suite/maskScorerTests/ref_maskreport_3-perimage.csv > comp_maskreport_3-perimage.txt

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

if ([ ! -e comp_maskreport_1.txt -o ! -e comp_maskreport_2.txt -o ! -e comp_maskreport_3.txt -o ! -e comp_maskreport_1-perimage.txt -o ! -e comp_maskreport_2-perimage.txt -o ! -e comp_maskreport_3-perimage.txt ]); then
  echo
  echo "    !!!!! MASK SCORER TESTS FAILED !!!!!    "
  echo
  exit
fi

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

  rm -rf ../../data/test_suite/maskScorerTests/morelight
  rm -rf ../../data/test_suite/maskScorerTests/gullflower_flower
  rm -rf ../../data/test_suite/maskScorerTests/gullflower2_flower2
else
  echo
  echo "    !!!!! MASK SCORER TESTS FAILED !!!!!    "
  echo
fi
