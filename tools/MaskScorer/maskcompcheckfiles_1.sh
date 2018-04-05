#!/bin/bash
procs=2
source test_init.sh

echo
echo "CASE 1: VALIDATING SCORING OF TARGET REGIONS"
echo

mypython=python2

#$mypython MaskScorer.py -t manipulation --refDir ../../data/test_suite/maskScorerTests -r reference/manipulation/NC2017-manipulation-ref.csv -x indexes/NC2017-manipulation-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1.csv -oR ../../data/test_suite/maskScorerTests/target_all -html
$mypython MaskScorer.py -t manipulation --refDir ../../data/test_suite/maskScorerTests -r reference/manipulation/NC2017-manipulation-ref.csv -x indexes/NC2017-manipulation-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1.csv -oR ../../data/test_suite/maskScorerTests/target_all/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1 --nspx 200 -p $procs -html --speedup --precision 12
$mypython MaskScorer.py -t manipulation --refDir ../../data/test_suite/maskScorerTests -r reference/manipulation/NC2017-manipulation-ref.csv -x indexes/NC2017-manipulation-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1.csv -oR ../../data/test_suite/maskScorerTests/target_purpose/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1 -qm "Purpose==['clone']" "Purpose==['add']" "Purpose==['removal']" "Purpose==['clone','add']" "Purpose==['heal']" "Purpose==['remove']" -p $procs -html --speedup --precision 12 --cache_dir /tmp/selective_loc --cache_flush
$mypython MaskScorer.py -t manipulation --refDir ../../data/test_suite/maskScorerTests -r reference/manipulation/NC2017-manipulation-ref.csv -x indexes/NC2017-manipulation-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1.csv -oR ../../data/test_suite/maskScorerTests/target_display/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1 -qm "Purpose==['heal']" --displayScoredOnly -p $procs --speedup --precision 12 --cache_dir /tmp/selective_loc/q4 -k gaussian

diff ../../data/test_suite/maskScorerTests/target_all/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_score.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_all.csv > comp_maskreport_all.txt
diff ../../data/test_suite/maskScorerTests/target_all/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_all-perimage.csv > comp_maskreport_all-perimage.txt
diff ../../data/test_suite/maskScorerTests/target_all/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_all-journalResults.csv > comp_maskreport_all-journalResults.txt

diff ../../data/test_suite/maskScorerTests/target_purpose/index_0/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_score.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_clone.csv > comp_maskreport_clone.txt
diff ../../data/test_suite/maskScorerTests/target_purpose/index_0/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_clone-perimage.csv > comp_maskreport_clone-perimage.txt
diff ../../data/test_suite/maskScorerTests/target_purpose/index_0/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_clone-journalResults.csv > comp_maskreport_clone-journalResults.txt

diff ../../data/test_suite/maskScorerTests/target_purpose/index_1/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_score.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_add.csv > comp_maskreport_add.txt
diff ../../data/test_suite/maskScorerTests/target_purpose/index_1/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_add-perimage.csv > comp_maskreport_add-perimage.txt
diff ../../data/test_suite/maskScorerTests/target_purpose/index_1/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_add-journalResults.csv > comp_maskreport_add-journalResults.txt

#there should be no files in the removal folder

diff ../../data/test_suite/maskScorerTests/target_purpose/index_3/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_score.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_clone_add.csv > comp_maskreport_clone_add.txt
diff ../../data/test_suite/maskScorerTests/target_purpose/index_3/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_clone_add-perimage.csv > comp_maskreport_clone_add-perimage.txt
diff ../../data/test_suite/maskScorerTests/target_purpose/index_3/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_clone_add-journalResults.csv > comp_maskreport_clone_add-journalResults.txt

#heal should only have journalResults and perimage to validate that nothing was scored or averaged
diff ../../data/test_suite/maskScorerTests/target_purpose/index_4/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_heal-perimage.csv > comp_maskreport_heal-perimage.txt
diff ../../data/test_suite/maskScorerTests/target_purpose/index_4/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_heal-journalResults.csv > comp_maskreport_heal-journalResults.txt

diff ../../data/test_suite/maskScorerTests/target_purpose/index_5/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_score.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_remove.csv > comp_maskreport_remove.txt
diff ../../data/test_suite/maskScorerTests/target_purpose/index_5/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_remove-perimage.csv > comp_maskreport_remove-perimage.txt
diff ../../data/test_suite/maskScorerTests/target_purpose/index_5/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_remove-journalResults.csv > comp_maskreport_remove-journalResults.txt

diff ../../data/test_suite/maskScorerTests/target_display/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_display-perimage.csv > comp_maskreport_display-perimage.txt
diff ../../data/test_suite/maskScorerTests/target_display/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_display-journalResults.csv > comp_maskreport_display-journalResults.txt

#flags
flag_all=1
flag_allpi=1
flag_alljr=1
flag_clone=1
flag_clonepi=1
flag_clonejr=1
flag_add=1
flag_addpi=1
flag_addjr=1
flag_clone_add=1
flag_clone_addpi=1
flag_clone_addjr=1
flag_healpi=1
flag_healjr=1
flag_remove=1
flag_removepi=1
flag_removejr=1
flag_displaypi=1
flag_displayjr=1

#filters to evaluate
filter_all="cat comp_maskreport_all.txt | grep -v CVS"
filter_allpi="cat comp_maskreport_all-perimage.txt | grep -v CVS"
filter_alljr="cat comp_maskreport_all-journalResults.txt | grep -v CVS"
filter_clone="cat comp_maskreport_clone.txt | grep -v CVS"
filter_clonepi="cat comp_maskreport_clone-perimage.txt | grep -v CVS"
filter_clonejr="cat comp_maskreport_clone-journalResults.txt | grep -v CVS"
filter_add="cat comp_maskreport_add.txt | grep -v CVS"
filter_addpi="cat comp_maskreport_add-perimage.txt | grep -v CVS"
filter_addjr="cat comp_maskreport_add-journalResults.txt | grep -v CVS"
filter_clone_add="cat comp_maskreport_clone_add.txt | grep -v CVS"
filter_clone_addpi="cat comp_maskreport_clone_add-perimage.txt | grep -v CVS"
filter_clone_addjr="cat comp_maskreport_clone_add-journalResults.txt | grep -v CVS"
filter_healpi="cat comp_maskreport_heal-perimage.txt | grep -v CVS"
filter_healjr="cat comp_maskreport_heal-journalResults.txt | grep -v CVS"
filter_remove="cat comp_maskreport_remove.txt | grep -v CVS"
filter_removepi="cat comp_maskreport_remove-perimage.txt | grep -v CVS"
filter_removejr="cat comp_maskreport_remove-journalResults.txt | grep -v CVS"
filter_displaypi="cat comp_maskreport_display-perimage.txt | grep -v CVS"
filter_displayjr="cat comp_maskreport_display-journalResults.txt | grep -v CVS"

if ([ -f comp_maskreport_removal.txt -o -f comp_maskreport_removal-perimage.txt -o -f comp_maskreport_removal-journalResults.txt \
 -o -f comp_maskreport_heal.txt \
 -o -f comp_maskreport_display.txt \
]); then
  echo
  echo "    !!!!! MASK SCORER TEST FAILED AT CASE 1 !!!!!    "
  echo "     EXTRA FILES PRESENT     "
  echo
  exit
fi
 
if ([ ! -f comp_maskreport_all.txt -o ! -f comp_maskreport_all-perimage.txt -o ! -f comp_maskreport_all-journalResults.txt \
 -o ! -f comp_maskreport_clone.txt -o ! -f comp_maskreport_clone-perimage.txt -o ! -f comp_maskreport_clone-journalResults.txt \
 -o ! -f comp_maskreport_add.txt -o ! -f comp_maskreport_add-perimage.txt -o ! -f comp_maskreport_add-journalResults.txt \
 -o ! -f comp_maskreport_clone_add.txt -o ! -f comp_maskreport_clone_add-perimage.txt -o ! -f comp_maskreport_clone_add-journalResults.txt \
 -o ! -f comp_maskreport_remove.txt -o ! -f comp_maskreport_remove-perimage.txt -o ! -f comp_maskreport_remove-journalResults.txt \
 -o ! -f comp_maskreport_heal-journalResults.txt -o ! -f comp_maskreport_heal-perimage.txt \
 -o ! -f comp_maskreport_display-journalResults.txt -o ! -f comp_maskreport_display-perimage.txt \
]); then
  echo
  echo "    !!!!! MASK SCORER TEST FAILED AT CASE 1 !!!!!    "
  echo "     MISSING FILES ABSENT     "
  echo
  exit
fi

if test "`eval $filter_all`" = "" ; then
  flag_all=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_all/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_score.csv
	fi
	rm comp_maskreport_all.txt
else
	echo comp_maskreport_all.txt
	cat comp_maskreport_all.txt
fi

if test "`eval $filter_allpi`" = "" ; then
  flag_allpi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_all/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_all-perimage.txt
else
	echo comp_maskreport_all-perimage.txt
	cat comp_maskreport_all-perimage.txt
fi

if test "`eval $filter_alljr`" = "" ; then
  flag_alljr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_all/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv
	fi
	rm comp_maskreport_all-journalResults.txt
else
	echo comp_maskreport_all-journalResults.txt
	cat comp_maskreport_all-journalResults.txt
fi

if test "`eval $filter_clone`" = "" ; then
  flag_clone=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_0/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_score.csv
	fi
	rm comp_maskreport_clone.txt
else
	echo comp_maskreport_clone.txt
	cat comp_maskreport_clone.txt
fi

if test "`eval $filter_clonepi`" = "" ; then
  flag_clonepi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_0/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_clone-perimage.txt
else
	echo comp_maskreport_clone-perimage.txt
	cat comp_maskreport_clone-perimage.txt
fi

if test "`eval $filter_clonejr`" = "" ; then
  flag_clonejr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_0/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv
	fi
	rm comp_maskreport_clone-journalResults.txt
else
	echo comp_maskreport_clone-journalResults.txt
	cat comp_maskreport_clone-journalResults.txt
fi

if test "`eval $filter_add`" = "" ; then
  flag_add=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_1/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_score.csv
	fi
	rm comp_maskreport_add.txt
else
	echo comp_maskreport_add.txt
	cat comp_maskreport_add.txt
fi

if test "`eval $filter_addpi`" = "" ; then
  flag_addpi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_1/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_add-perimage.txt
else
	echo comp_maskreport_add-perimage.txt
	cat comp_maskreport_add-perimage.txt
fi

if test "`eval $filter_addjr`" = "" ; then
  flag_addjr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_1/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv
	fi
	rm comp_maskreport_add-journalResults.txt
else
	echo comp_maskreport_add-journalResults.txt
	cat comp_maskreport_add-journalResults.txt
fi

if test "`eval $filter_clone_add`" = "" ; then
  flag_clone_add=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_3/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_score.csv
	fi
	rm comp_maskreport_clone_add.txt
else
	echo comp_maskreport_clone_add.txt
	cat comp_maskreport_clone_add.txt
fi

if test "`eval $filter_clone_addpi`" = "" ; then
  flag_clone_addpi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_3/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_clone_add-perimage.txt
else
	echo comp_maskreport_clone_add-perimage.txt
	cat comp_maskreport_clone_add-perimage.txt
fi

if test "`eval $filter_clone_addjr`" = "" ; then
  flag_clone_addjr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_3/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv
	fi
	rm comp_maskreport_clone_add-journalResults.txt
else
	echo comp_maskreport_clone_add-journalResults.txt
	cat comp_maskreport_clone_add-journalResults.txt
fi

if test "`eval $filter_healpi`" = "" ; then
  flag_healpi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_4/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_heal-perimage.txt
else
	echo comp_maskreport_heal-perimage.txt
	cat comp_maskreport_heal-perimage.txt
fi

if test "`eval $filter_healjr`" = "" ; then
  flag_healjr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_4/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv
	fi
	rm comp_maskreport_heal-journalResults.txt
else
	echo comp_maskreport_heal-journalResults.txt
	cat comp_maskreport_heal-journalResults.txt
fi

if test "`eval $filter_remove`" = "" ; then
  flag_remove=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_5/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_score.csv
	fi
	rm comp_maskreport_remove.txt
else
	echo comp_maskreport_remove.txt
	cat comp_maskreport_remove.txt
fi

if test "`eval $filter_removepi`" = "" ; then
  flag_removepi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_5/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_remove-perimage.txt
else
	echo comp_maskreport_remove-perimage.txt
	cat comp_maskreport_remove-perimage.txt
fi

if test "`eval $filter_removejr`" = "" ; then
  flag_removejr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_5/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv
	fi
	rm comp_maskreport_remove-journalResults.txt
else
	echo comp_maskreport_remove-journalResults.txt
	cat comp_maskreport_remove-journalResults.txt
fi

if test "`eval $filter_displaypi`" = "" ; then
  flag_displaypi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_display/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_display-perimage.txt
else
	echo comp_maskreport_display-perimage.txt
	cat comp_maskreport_display-perimage.txt
fi

if test "`eval $filter_displayjr`" = "" ; then
  flag_displayjr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_display/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv
	fi
	rm comp_maskreport_display-journalResults.txt
else
	echo comp_maskreport_display-journalResults.txt
	cat comp_maskreport_display-journalResults.txt
fi

#optOut case
$mypython MaskScorer.py -t manipulation --refDir ../../data/test_suite/maskScorerTests -r reference/manipulation/NC2017-manipulation-ref.csv -x indexes/NC2017-manipulation-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1.csv -oR ../../data/test_suite/maskScorerTests/target_optOut/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1 -qm "Purpose==['clone','add']" -p $procs -html --speedup --optOut --precision 12 --cache_dir /tmp/selective_loc/q3

diff ../../data/test_suite/maskScorerTests/target_optOut/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_score.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_target_optOut.csv > comp_maskreport_target_optOut.txt
diff ../../data/test_suite/maskScorerTests/target_optOut/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_target_optOut-perimage.csv > comp_maskreport_target_optOut-perimage.txt
diff ../../data/test_suite/maskScorerTests/target_optOut/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_target_optOut-journalResults.csv > comp_maskreport_target_optOut-journalResults.txt

flag_target_optOut=1
flag_target_optOutpi=1
flag_target_optOutjr=1

filter_target_optOut="cat comp_maskreport_target_optOut.txt | grep -v CVS"
filter_target_optOutpi="cat comp_maskreport_target_optOut-perimage.txt | grep -v CVS"
filter_target_optOutjr="cat comp_maskreport_target_optOut-journalResults.txt | grep -v CVS"

if ([ ! -f comp_maskreport_target_optOut.txt -o ! -f comp_maskreport_target_optOut-perimage.txt -o ! -f comp_maskreport_target_optOut-journalResults.txt \
]); then
  echo
  echo "    !!!!! MASK SCORER TEST FAILED AT CASE 1 !!!!!    "
  echo "     MISSING FILES ABSENT     "
  echo
  exit
fi

if test "`eval $filter_target_optOut`" = "" ; then
  flag_target_optOut=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_optOut/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_score.csv
	fi
	rm comp_maskreport_target_optOut.txt
else
	echo comp_maskreport_target_optOut.txt
	cat comp_maskreport_target_optOut.txt
fi

if test "`eval $filter_target_optOutpi`" = "" ; then
  flag_target_optOutpi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_optOut/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_target_optOut-perimage.txt
else
	echo comp_maskreport_target_optOut-perimage.txt
	cat comp_maskreport_target_optOut-perimage.txt
fi

if test "`eval $filter_target_optOutjr`" = "" ; then
  flag_target_optOutjr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_optOut/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv
	fi
	rm comp_maskreport_target_optOut-journalResults.txt
else
	echo comp_maskreport_target_optOut-journalResults.txt
	cat comp_maskreport_target_optOut-journalResults.txt
fi

target_optOut_total=$(($flag_target_optOut + $flag_target_optOutpi + $flag_target_optOutjr))

$mypython MaskScorer.py -t splice --refDir ../../data/test_suite/maskScorerTests -r reference/splice/NC2017-splice-ref.csv -x indexes/NC2017-splice-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2017_Unittest_Splice_ImgOnly_p-me_1/B_NC2017_Unittest_Splice_ImgOnly_p-me_1.csv -oR ../../data/test_suite/maskScorerTests/splice_query/B_NC2017_Unittest_Splice_ImgOnly_p-me_1 -qm "Operation==['PasteSplice']" -html --precision 12

diff ../../data/test_suite/maskScorerTests/splice_query/B_NC2017_Unittest_Splice_ImgOnly_p-me_1_mask_score.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_splice_query.csv > comp_maskreport_splice_query.txt
diff ../../data/test_suite/maskScorerTests/splice_query/B_NC2017_Unittest_Splice_ImgOnly_p-me_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_splice_query-perimage.csv > comp_maskreport_splice_query-perimage.txt
diff ../../data/test_suite/maskScorerTests/splice_query/B_NC2017_Unittest_Splice_ImgOnly_p-me_1_journalResults.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_splice_query-journalResults.csv > comp_maskreport_splice_query-journalResults.txt

flag_splice_query=1
flag_splice_querypi=1
flag_splice_queryjr=1

filter_splice_query="cat comp_maskreport_splice_query.txt | grep -v CVS"
filter_splice_querypi="cat comp_maskreport_splice_query-perimage.txt | grep -v CVS"
filter_splice_queryjr="cat comp_maskreport_splice_query-journalResults.txt | grep -v CVS"

if ([ ! -f comp_maskreport_splice_query.txt -o ! -f comp_maskreport_splice_query-perimage.txt -o ! -f comp_maskreport_splice_query-journalResults.txt \
]); then
  echo
  echo "    !!!!! MASK SCORER TEST FAILED AT CASE 1 !!!!!    "
  echo "     MISSING FILES ABSENT     "
  echo
  exit
fi

if test "`eval $filter_splice_query`" = "" ; then
  flag_splice_query=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/splice_query/B_NC2017_Unittest_Splice_ImgOnly_p-me_1_mask_score.csv
	fi
	rm comp_maskreport_splice_query.txt
else
	echo comp_maskreport_splice_query.txt
	cat comp_maskreport_splice_query.txt
fi

if test "`eval $filter_splice_querypi`" = "" ; then
  flag_splice_querypi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/splice_query/B_NC2017_Unittest_Splice_ImgOnly_p-me_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_splice_query-perimage.txt
else
	echo comp_maskreport_splice_query-perimage.txt
	cat comp_maskreport_splice_query-perimage.txt
fi

if test "`eval $filter_splice_queryjr`" = "" ; then
  flag_splice_queryjr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/splice_query/B_NC2017_Unittest_Splice_ImgOnly_p-me_1_journalResults.csv
	fi
	rm comp_maskreport_splice_query-journalResults.txt
else
	echo comp_maskreport_splice_query-journalResults.txt
	cat comp_maskreport_splice_query-journalResults.txt
fi

splice_query_total=$((flag_splice_query + flag_splice_querypi + flag_splice_queryjr))

flag_total=$(($flag_all + $flag_allpi + $flag_alljr\
 + $flag_clone + $flag_clonepi + $flag_clonejr\
 + $flag_add + $flag_addpi + $flag_addjr\
 + $flag_clone_add + $flag_clone_addpi + $flag_clone_addjr\
 + $flag_healpi + $flag_healjr\
 + $flag_remove + $flag_removepi + $flag_removejr\
 + $flag_displaypi + $flag_displayjr\
 + $target_optOut_total))
if ([ $flag_total == 0 ]); then
  echo
  echo "CASE 1 SUCCESSFULLY PASSED"
  echo
	if [ $clean = "TRUE" ] ; then
		rm -rf ../../data/test_suite/maskScorerTests/target_all
		rm -rf ../../data/test_suite/maskScorerTests/target_optOut
		rm -rf ../../data/test_suite/maskScorerTests/target_purpose
		rm -rf ../../data/test_suite/maskScorerTests/target_display
		rm -rf ../../data/test_suite/maskScorerTests/splice_query
#		rm -rf ../../data/test_suite/maskScorerTests/target_clone
#		rm -rf ../../data/test_suite/maskScorerTests/target_add
#		rm -rf ../../data/test_suite/maskScorerTests/target_removal
#		rm -rf ../../data/test_suite/maskScorerTests/target_clone_add
#		rm -rf ../../data/test_suite/maskScorerTests/target_heal
#		rm -rf ../../data/test_suite/maskScorerTests/target_remove
	fi
else
  echo
  echo "    !!!!! MASK SCORER TEST FAILED AT CASE 1 !!!!!    "
  echo
  exit
fi

