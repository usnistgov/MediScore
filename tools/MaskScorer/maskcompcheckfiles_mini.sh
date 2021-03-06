#!/bin/bash
clean=TRUE
procs=4

echo "BEGINNING FUNCTIONALITY TEST OF MASK SCORER"

#produce the output files
python2 MaskScorer.py -t splice --refDir ../../data/test_suite/maskScorerTests -r reference/splice/NC2017-splice-ref.csv -x indexes/NC2017-splice-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2017_Splice_ImgOnly_p-me_1/B_NC2017_Splice_ImgOnly_p-me_1.csv -oR ../../data/test_suite/maskScorerTests/splicetest -html --optOut -xF -p $procs --speedup
python2 MaskScorer.py -t manipulation --refDir ../../data/test_suite/maskScorerTests -r reference/manipulation/NC2017-manipulation-ref.csv -x indexes/NC2017-manipulation-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2017_Manipulation_ImgOnly_c-me2_1/B_NC2017_Manipulation_ImgOnly_c-me2_1.csv -oR ../../data/test_suite/maskScorerTests/manipconfmanmade -html -q "ConfidenceScore < 0.5" "ManMade=='no'" --optOut -p $procs --speedup

#compare them to ground truth files
diff ../../data/test_suite/maskScorerTests/splicetest/B_NC2017_Splice_ImgOnly_p-me_1-mask_score.csv ../../data/test_suite/maskScorerTests/ref_maskreport_splice.csv > comp_maskreport_splice.txt
diff ../../data/test_suite/maskScorerTests/splicetest/B_NC2017_Splice_ImgOnly_p-me_1-mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/ref_maskreport_splice-perimage.csv > comp_maskreport_splice-perimage.txt
diff ../../data/test_suite/maskScorerTests/splicetest/B_NC2017_Splice_ImgOnly_p-me_1-journalResults.csv ../../data/test_suite/maskScorerTests/ref_maskreport_splice-journalResults.csv > comp_maskreport_splice-journalResults.txt

diff ../../data/test_suite/maskScorerTests/manipconfmanmade/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_scores_0.csv ../../data/test_suite/maskScorerTests/ref_maskreport_manipconfmanmade_0.csv > comp_maskreport_manipconfmanmade_0.txt
diff ../../data/test_suite/maskScorerTests/manipconfmanmade/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_scores_1.csv ../../data/test_suite/maskScorerTests/ref_maskreport_manipconfmanmade_1.csv > comp_maskreport_manipconfmanmade_1.txt
#diff ../../data/test_suite/maskScorerTests/manipconfmanmade/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_scores_optout_0.csv ../../data/test_suite/maskScorerTests/ref_maskreport_manipconfmanmade_optout_0.csv > comp_maskreport_manipconfmanmade_optout_0.txt
#diff ../../data/test_suite/maskScorerTests/manipconfmanmade/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_scores_optout_1.csv ../../data/test_suite/maskScorerTests/ref_maskreport_manipconfmanmade_optout_1.csv > comp_maskreport_manipconfmanmade_optout_1.txt
diff ../../data/test_suite/maskScorerTests/manipconfmanmade/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/ref_maskreport_manipconfmanmade-perimage.csv > comp_maskreport_manipconfmanmade-perimage.txt
diff ../../data/test_suite/maskScorerTests/manipconfmanmade/B_NC2017_Manipulation_ImgOnly_c-me2_1-journalResults.csv ../../data/test_suite/maskScorerTests/ref_maskreport_manipconfmanmade-journalResults.csv > comp_maskreport_manipconfmanmade-journalResults.txt

flag_s=1
flag_spi=1
flag_sjr=1

flag_manipconfmanmade_0=0
flag_manipconfmanmade_1=0
flag_manipconfmanmade_optout_0=0
flag_manipconfmanmade_optout_1=0
flag_manipconfmanmadepi=0
flag_manipconfmanmadejr=0

filter_s="cat comp_maskreport_splice.txt | grep -v CVS"
filter_spi="cat comp_maskreport_splice-perimage.txt | grep -v CVS"
filter_sjr="cat comp_maskreport_splice-journalResults.txt | grep -v CVS"

filter_manipconfmanmade_0="cat comp_maskreport_manipconfmanmade_0.txt | grep -v CVS"
filter_manipconfmanmade_1="cat comp_maskreport_manipconfmanmade_1.txt | grep -v CVS"
filter_manipconfmanmade_optout_0="cat comp_maskreport_manipconfmanmade_optout_0.txt | grep -v CVS"
filter_manipconfmanmade_optout_1="cat comp_maskreport_manipconfmanmade_optout_1.txt | grep -v CVS"
filter_manipconfmanmadepi="cat comp_maskreport_manipconfmanmade-perimage.txt | grep -v CVS"
filter_manipconfmanmadejr="cat comp_maskreport_manipconfmanmade-journalResults.txt | grep -v CVS"

# -o ! -f comp_maskreport_manipconfmanmade_optout_0.txt -o ! -f comp_maskreport_manipconfmanmade_optout_1.txt \
if ([ ! -f comp_maskreport_splice-perimage.txt -o ! -f comp_maskreport_splice.txt -o ! -f comp_maskreport_splice-journalResults.txt \
 -o ! -f comp_maskreport_manipconfmanmade_0.txt -o ! -f comp_maskreport_manipconfmanmade_1.txt \
 -o ! -f comp_maskreport_manipconfmanmade-perimage.txt -o ! -f comp_maskreport_manipconfmanmade-journalResults.txt \
]); then
  echo
  echo "    !!!!! MASK SCORER TEST FAILED !!!!!    "
  echo "     MISSING FILES ABSENT     "
  echo
  exit
fi

if test "`eval $filter_s`" = "" ; then
  flag_s=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/splicetest/B_NC2017_Splice_ImgOnly_p-me_1-mask_score.csv
	fi
	rm comp_maskreport_splice.txt
else
	echo comp_maskreport_splice.txt
	cat comp_maskreport_splice.txt
fi

if test "`eval $filter_spi`" = "" ; then
  flag_spi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/splicetest/B_NC2017_Splice_ImgOnly_p-me_1-mask_scores_perimage.csv
	fi
	rm comp_maskreport_splice-perimage.txt
else
	echo comp_maskreport_splice-perimage.txt
	cat comp_maskreport_splice-perimage.txt
fi

if test "`eval $filter_sjr`" = "" ; then
  flag_sjr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/splicetest/B_NC2017_Splice_ImgOnly_p-me_1-journalResults.csv
	fi
	rm comp_maskreport_splice-journalResults.txt
else
	echo comp_maskreport_splice-journalResults.txt
	cat comp_maskreport_splice-journalResults.txt
fi

if test "`eval $filter_manipconfmanmade_0`" = "" ; then
  flag_manipconfmanmade_0=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/manipconfmanmade/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_scores_0.csv
	fi
	rm comp_maskreport_manipconfmanmade_0.txt
else
	echo comp_maskreport_manipconfmanmade_0.txt
	cat comp_maskreport_manipconfmanmade_0.txt
fi

if test "`eval $filter_manipconfmanmade_1`" = "" ; then
  flag_manipconfmanmade_1=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/manipconfmanmade/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_scores_1.csv
	fi
	rm comp_maskreport_manipconfmanmade_1.txt
else
	echo comp_maskreport_manipconfmanmade_1.txt
	cat comp_maskreport_manipconfmanmade_1.txt
fi

#if test "`eval $filter_manipconfmanmade_optout_0`" = "" ; then
#  flag_manipconfmanmade_optout_0=0
#	if [ $clean = "TRUE" ] ; then
#		rm ../../data/test_suite/maskScorerTests/manipconfmanmade/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_scores_optout_0.csv
#	fi
#	rm comp_maskreport_manipconfmanmade_optout_0.txt
#else
#	echo comp_maskreport_manipconfmanmade_optout_0.txt
#	cat comp_maskreport_manipconfmanmade_optout_0.txt
#fi
#
#if test "`eval $filter_manipconfmanmade_optout_1`" = "" ; then
#  flag_manipconfmanmade_optout_1=0
#	if [ $clean = "TRUE" ] ; then
#		rm ../../data/test_suite/maskScorerTests/manipconfmanmade/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_scores_optout_1.csv
#	fi
#	rm comp_maskreport_manipconfmanmade_optout_1.txt
#else
#	echo comp_maskreport_manipconfmanmade_optout_1.txt
#	cat comp_maskreport_manipconfmanmade_optout_1.txt
#fi

if test "`eval $filter_manipconfmanmadepi`" = "" ; then
  flag_manipconfmanmadepi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/manipconfmanmade/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_scores_perimage.csv
	fi
	rm comp_maskreport_manipconfmanmade-perimage.txt
else
	echo comp_maskreport_manipconfmanmade-perimage.txt
	cat comp_maskreport_manipconfmanmade-perimage.txt
fi

if test "`eval $filter_manipconfmanmadejr`" = "" ; then
  flag_manipconfmanmadejr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/manipconfmanmade/B_NC2017_Manipulation_ImgOnly_c-me2_1-journalResults.csv
	fi
	rm comp_maskreport_manipconfmanmade-journalResults.txt
else
	echo comp_maskreport_manipconfmanmade-journalResults.txt
	cat comp_maskreport_manipconfmanmade-journalResults.txt
fi

flag_total=$(($flag_s + $flag_spi + $flag_sjr + $flag_manipconfmanmade_0 + $flag_manipconfmanmade_1 + $flag_manipconfmanmade_output_0 + $flag_manipconfmanmade_output_1 + $flag_manipconfmanmadepi + $flag_manipconfmanmadejr))

if ([ $flag_total -eq 0 ]); then
  echo
  echo "MASK SCORER SUCCESSFULLY PASSED"
  echo
	if [ $clean = "TRUE" ] ; then
		rm -rf ../../data/test_suite/maskScorerTests/splicetest
                rm -rf ../../data/test_suite/maskScorerTests/manipconfmanmade
	fi
else
  echo
  echo "    !!!!! MASK SCORER TEST FAILED !!!!!    "
  echo
  exit
fi
