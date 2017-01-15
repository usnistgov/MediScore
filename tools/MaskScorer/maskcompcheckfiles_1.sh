#!/bin/bash
clean=FALSE

echo
echo "CASE 1: VALIDATING SCORING OF TARGET REGIONS"
echo

python2 MaskScorer.py -t manipulation --refDir ../../data/test_suite/maskScorerTests -r reference/manipulation/NC2017-manipulation-ref.csv -x indexes/NC2017-manipulation-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2017_Manipulation_ImgOnly_c-me2_1/B_NC2017_Manipulation_ImgOnly_c-me2_1.csv -oR ../../data/test_suite/maskScorerTests/target_all -html
python2 MaskScorer.py -t manipulation --refDir ../../data/test_suite/maskScorerTests -r reference/manipulation/NC2017-manipulation-ref.csv -x indexes/NC2017-manipulation-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2017_Manipulation_ImgOnly_c-me2_1/B_NC2017_Manipulation_ImgOnly_c-me2_1.csv -oR ../../data/test_suite/maskScorerTests/target_purpose -html -qm "Purpose==['clone']" "Purpose==['add']" "Purpose==['removal']" "Purpose==['clone','add']" "Purpose==['heal']" "Purpose==['remove']"


diff ../../data/test_suite/maskScorerTests/target_all/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_score.csv ../../data/test_suite/maskScorerTests/ref_maskreport_all.csv > comp_maskreport_all.txt
diff ../../data/test_suite/maskScorerTests/target_all/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/ref_maskreport_all-perimage.csv > comp_maskreport_all-perimage.txt
diff ../../data/test_suite/maskScorerTests/target_all/B_NC2017_Manipulation_ImgOnly_c-me2_1-journalResults.csv ../../data/test_suite/maskScorerTests/ref_maskreport_all-journalResults.csv > comp_maskreport_all-journalResults.txt

diff ../../data/test_suite/maskScorerTests/target_purpose/index_0/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_score.csv ../../data/test_suite/maskScorerTests/ref_maskreport_clone.csv > comp_maskreport_clone.txt
diff ../../data/test_suite/maskScorerTests/target_purpose/index_0/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/ref_maskreport_clone-perimage.csv > comp_maskreport_clone-perimage.txt
diff ../../data/test_suite/maskScorerTests/target_purpose/index_0/B_NC2017_Manipulation_ImgOnly_c-me2_1-journalResults.csv ../../data/test_suite/maskScorerTests/ref_maskreport_clone-journalResults.csv > comp_maskreport_clone-journalResults.txt

diff ../../data/test_suite/maskScorerTests/target_purpose/index_1/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_score.csv ../../data/test_suite/maskScorerTests/ref_maskreport_add.csv > comp_maskreport_add.txt
diff ../../data/test_suite/maskScorerTests/target_purpose/index_1/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/ref_maskreport_add-perimage.csv > comp_maskreport_add-perimage.txt
diff ../../data/test_suite/maskScorerTests/target_purpose/index_1/B_NC2017_Manipulation_ImgOnly_c-me2_1-journalResults.csv ../../data/test_suite/maskScorerTests/ref_maskreport_add-journalResults.csv > comp_maskreport_add-journalResults.txt

#there should be no files in the removal folder

diff ../../data/test_suite/maskScorerTests/target_purpose/index_3/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_score.csv ../../data/test_suite/maskScorerTests/ref_maskreport_clone_add.csv > comp_maskreport_clone_add.txt
diff ../../data/test_suite/maskScorerTests/target_purpose/index_3/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/ref_maskreport_clone_add-perimage.csv > comp_maskreport_clone_add-perimage.txt
diff ../../data/test_suite/maskScorerTests/target_purpose/index_3/B_NC2017_Manipulation_ImgOnly_c-me2_1-journalResults.csv ../../data/test_suite/maskScorerTests/ref_maskreport_clone_add-journalResults.csv > comp_maskreport_clone_add-journalResults.txt

#heal should only have journalResults and perimage to validate that nothing was scored or averaged
diff ../../data/test_suite/maskScorerTests/target_purpose/index_4/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/ref_maskreport_heal-perimage.csv > comp_maskreport_heal-perimage.txt
diff ../../data/test_suite/maskScorerTests/target_purpose/index_4/B_NC2017_Manipulation_ImgOnly_c-me2_1-journalResults.csv ../../data/test_suite/maskScorerTests/ref_maskreport_heal-journalResults.csv > comp_maskreport_heal-journalResults.txt

diff ../../data/test_suite/maskScorerTests/target_purpose/index_5/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_score.csv ../../data/test_suite/maskScorerTests/ref_maskreport_remove.csv > comp_maskreport_remove.txt
diff ../../data/test_suite/maskScorerTests/target_purpose/index_5/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/ref_maskreport_remove-perimage.csv > comp_maskreport_remove-perimage.txt
diff ../../data/test_suite/maskScorerTests/target_purpose/index_5/B_NC2017_Manipulation_ImgOnly_c-me2_1-journalResults.csv ../../data/test_suite/maskScorerTests/ref_maskreport_remove-journalResults.csv > comp_maskreport_remove-journalResults.txt

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

#filters to evaluate
filter_all="cat comp_maskreport_all.txt | grep -v -CVS"
filter_allpi="cat comp_maskreport_all-perimage.txt | grep -v -CVS"
filter_alljr="cat comp_maskreport_all-journalResults.txt | grep -v -CVS"
filter_clone="cat comp_maskreport_clone.txt | grep -v -CVS"
filter_clonepi="cat comp_maskreport_clone-perimage.txt | grep -v -CVS"
filter_clonejr="cat comp_maskreport_clone-journalResults.txt | grep -v -CVS"
filter_add="cat comp_maskreport_add.txt | grep -v -CVS"
filter_addpi="cat comp_maskreport_add-perimage.txt | grep -v -CVS"
filter_addjr="cat comp_maskreport_add-journalResults.txt | grep -v -CVS"
filter_clone_add="cat comp_maskreport_clone_add.txt | grep -v -CVS"
filter_clone_addpi="cat comp_maskreport_clone_add-perimage.txt | grep -v -CVS"
filter_clone_addjr="cat comp_maskreport_clone_add-journalResults.txt | grep -v -CVS"
filter_healpi="cat comp_maskreport_heal-perimage.txt | grep -v -CVS"
filter_healjr="cat comp_maskreport_heal-journalResults.txt | grep -v -CVS"
filter_remove="cat comp_maskreport_remove.txt | grep -v -CVS"
filter_removepi="cat comp_maskreport_remove-perimage.txt | grep -v -CVS"
filter_removejr="cat comp_maskreport_remove-journalResults.txt | grep -v -CVS"

if ([ -f comp_maskreport_removal.txt -o -f comp_maskreport_removal-perimage.txt -o -f comp_maskreport_removal-journalResults.txt \
 -o -f comp_maskreport_heal.txt \
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
		rm ../../data/test_suite/maskScorerTests/target_all/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_score.csv
	fi
	rm comp_maskreport_all.txt
else
	echo comp_maskreport_all.txt
	cat comp_maskreport_all.txt
fi

if test "`eval $filter_allpi`" = "" ; then
  flag_allpi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_all/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_scores_perimage.csv
	fi
	rm comp_maskreport_all-perimage.txt
else
	echo comp_maskreport_all-perimage.txt
	cat comp_maskreport_all-perimage.txt
fi

if test "`eval $filter_alljr`" = "" ; then
  flag_alljr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_all/B_NC2017_Manipulation_ImgOnly_c-me2_1-journalResults.csv
	fi
	rm comp_maskreport_all-journalResults.txt
else
	echo comp_maskreport_all-journalResults.txt
	cat comp_maskreport_all-journalResults.txt
fi

if test "`eval $filter_clone`" = "" ; then
  flag_clone=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_0/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_score.csv
	fi
	rm comp_maskreport_clone.txt
else
	echo comp_maskreport_clone.txt
	cat comp_maskreport_clone.txt
fi

if test "`eval $filter_clonepi`" = "" ; then
  flag_clonepi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_0/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_scores_perimage.csv
	fi
	rm comp_maskreport_clone-perimage.txt
else
	echo comp_maskreport_clone-perimage.txt
	cat comp_maskreport_clone-perimage.txt
fi

if test "`eval $filter_clonejr`" = "" ; then
  flag_clonejr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_0/B_NC2017_Manipulation_ImgOnly_c-me2_1-journalResults.csv
	fi
	rm comp_maskreport_clone-journalResults.txt
else
	echo comp_maskreport_clone-journalResults.txt
	cat comp_maskreport_clone-journalResults.txt
fi

if test "`eval $filter_add`" = "" ; then
  flag_add=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_1/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_score.csv
	fi
	rm comp_maskreport_add.txt
else
	echo comp_maskreport_add.txt
	cat comp_maskreport_add.txt
fi

if test "`eval $filter_addpi`" = "" ; then
  flag_addpi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_1/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_scores_perimage.csv
	fi
	rm comp_maskreport_add-perimage.txt
else
	echo comp_maskreport_add-perimage.txt
	cat comp_maskreport_add-perimage.txt
fi

if test "`eval $filter_addjr`" = "" ; then
  flag_addjr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_1/B_NC2017_Manipulation_ImgOnly_c-me2_1-journalResults.csv
	fi
	rm comp_maskreport_add-journalResults.txt
else
	echo comp_maskreport_add-journalResults.txt
	cat comp_maskreport_add-journalResults.txt
fi

if test "`eval $filter_clone_add`" = "" ; then
  flag_clone_add=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_3/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_score.csv
	fi
	rm comp_maskreport_clone_add.txt
else
	echo comp_maskreport_clone_add.txt
	cat comp_maskreport_clone_add.txt
fi

if test "`eval $filter_clone_addpi`" = "" ; then
  flag_clone_addpi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_3/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_scores_perimage.csv
	fi
	rm comp_maskreport_clone_add-perimage.txt
else
	echo comp_maskreport_clone_add-perimage.txt
	cat comp_maskreport_clone_add-perimage.txt
fi

if test "`eval $filter_clone_addjr`" = "" ; then
  flag_clone_addjr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_3/B_NC2017_Manipulation_ImgOnly_c-me2_1-journalResults.csv
	fi
	rm comp_maskreport_clone_add-journalResults.txt
else
	echo comp_maskreport_clone_add-journalResults.txt
	cat comp_maskreport_clone_add-journalResults.txt
fi

if test "`eval $filter_healpi`" = "" ; then
  flag_healpi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_4/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_scores_perimage.csv
	fi
	rm comp_maskreport_heal-perimage.txt
else
	echo comp_maskreport_heal-perimage.txt
	cat comp_maskreport_heal-perimage.txt
fi

if test "`eval $filter_healjr`" = "" ; then
  flag_healjr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_4/B_NC2017_Manipulation_ImgOnly_c-me2_1-journalResults.csv
	fi
	rm comp_maskreport_heal-journalResults.txt
else
	echo comp_maskreport_heal-journalResults.txt
	cat comp_maskreport_heal-journalResults.txt
fi

if test "`eval $filter_remove`" = "" ; then
  flag_remove=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_5/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_score.csv
	fi
	rm comp_maskreport_remove.txt
else
	echo comp_maskreport_remove.txt
	cat comp_maskreport_remove.txt
fi

if test "`eval $filter_removepi`" = "" ; then
  flag_removepi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_5/B_NC2017_Manipulation_ImgOnly_c-me2_1-mask_scores_perimage.csv
	fi
	rm comp_maskreport_remove-perimage.txt
else
	echo comp_maskreport_remove-perimage.txt
	cat comp_maskreport_remove-perimage.txt
fi

if test "`eval $filter_removejr`" = "" ; then
  flag_removejr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/target_purpose/index_5/B_NC2017_Manipulation_ImgOnly_c-me2_1-journalResults.csv
	fi
	rm comp_maskreport_remove-journalResults.txt
else
	echo comp_maskreport_remove-journalResults.txt
	cat comp_maskreport_remove-journalResults.txt
fi

if ([ $flag_all == 0 -a $flag_allpi == 0 -a $flag_alljr == 0 \
-a $flag_clone == 0 -a $flag_clonepi == 0 -a $flag_clonejr == 0 \
-a $flag_add == 0 -a $flag_addpi == 0 -a $flag_addjr == 0 \
-a $flag_clone_add == 0 -a $flag_clone_addpi == 0 -a $flag_clone_addjr == 0 \
-a $flag_healpi == 0 -a $flag_healjr == 0 \
-a $flag_remove == 0 -a $flag_removepi == 0 -a $flag_removejr == 0 \
]); then
  echo
  echo "CASE 1 SUCCESSFULLY PASSED"
  echo
	if [ $clean = "TRUE" ] ; then
		rm -rf ../../data/test_suite/maskScorerTests/target_all
		rm -rf ../../data/test_suite/maskScorerTests/target_clone
		rm -rf ../../data/test_suite/maskScorerTests/target_add
		rm -rf ../../data/test_suite/maskScorerTests/target_removal
		rm -rf ../../data/test_suite/maskScorerTests/target_clone_add
		rm -rf ../../data/test_suite/maskScorerTests/target_heal
		rm -rf ../../data/test_suite/maskScorerTests/target_remove
	fi
else
  echo
  echo "    !!!!! MASK SCORER TEST FAILED AT CASE 1 !!!!!    "
  echo
  exit
fi

