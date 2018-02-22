#!/bin/bash
clean=TRUE
procs=4

echo "BEGINNING FUNCTIONALITY TEST OF MASK SCORER"
echo "PROCEEDING TO THOROUGHLY CHECK ALL CASES"
echo
echo "CASE 0: VALIDATING FULL SCORING"
echo

mypython=python2

#produce the output files
$mypython MaskScorer.py -t manipulation --refDir ../../data/test_suite/maskScorerTests -r reference/manipulation/NC2016-manipulation-ref.csv -x indexes/NC2016-manipulation-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2016_Unittest_Manipulation_ImgOnly_c-me2_1/B_NC2016_Unittest_Manipulation_ImgOnly_c-me2_1.csv -oR ../../data/test_suite/maskScorerTests/maniptest/B_NC2016_Unittest_Manipulation_ImgOnly_c-me2_1 -p $procs --speedup --precision 12
$mypython MaskScorer.py -t splice --refDir ../../data/test_suite/maskScorerTests -r reference/splice/NC2017-splice-ref.csv -x indexes/NC2017-splice-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2017_Unittest_Splice_ImgOnly_p-me_1/B_NC2017_Unittest_Splice_ImgOnly_p-me_1.csv -oR ../../data/test_suite/maskScorerTests/splicetest/B_NC2017_Unittest_Splice_ImgOnly_p-me_1 -html -p $procs --speedup --precision 12
$mypython MaskScorer.py -t manipulation --refDir ../../data/test_suite/maskScorerTests -r reference/manipulation/NC2017-manipulation-ref.csv -x indexes/NC2017-manipulation-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1.csv -oR ../../data/test_suite/maskScorerTests/threstest/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1 -html --sbin 128 -p $procs --speedup --precision 12
#$mypython MaskScorer.py -t removal --refDir ../../data/test_suite/maskScorerTests -r reference/removal/NC2016-removal-ref.csv -x indexes/NC2016-removal-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2016_Removal_ImgOnly_c-me2_2/B_NC2016_Removal_ImgOnly_c-me2_2.csv -oR ../../data/test_suite/maskScorerTests/temp_maskreport_3 --sbin 127
$mypython MaskScorer.py -t splice --refDir ../../data/test_suite/maskScorerTests -r reference/splice/NC2017-splice-ref.csv -x indexes/NC2017-splice-index.csv --sysDir ../../data/test_suite/maskScorerTests -s C_NC2017_Unittest_Splice_ImgOnly_p-me_1/C_NC2017_Unittest_Splice_ImgOnly_p-me_1.csv -oR ../../data/test_suite/maskScorerTests/splicebin/C_NC2017_Unittest_Splice_ImgOnly_p-me_1 -xF -p 6 -html --outMeta --outAllmeta --sbin 200 --speedup --precision 12

#compare them to ground truth files
diff ../../data/test_suite/maskScorerTests/maniptest/B_NC2016_Unittest_Manipulation_ImgOnly_c-me2_1_mask_score.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_manip.csv > comp_maskreport_manip.txt
diff ../../data/test_suite/maskScorerTests/maniptest/B_NC2016_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_manip-perimage.csv > comp_maskreport_manip-perimage.txt
diff ../../data/test_suite/maskScorerTests/maniptest/B_NC2016_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_manip-journalResults.csv > comp_maskreport_manip-journalResults.txt
diff ../../data/test_suite/maskScorerTests/splicetest/B_NC2017_Unittest_Splice_ImgOnly_p-me_1_mask_score.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_splice.csv > comp_maskreport_splice.txt
diff ../../data/test_suite/maskScorerTests/splicetest/B_NC2017_Unittest_Splice_ImgOnly_p-me_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_splice-perimage.csv > comp_maskreport_splice-perimage.txt
diff ../../data/test_suite/maskScorerTests/splicetest/B_NC2017_Unittest_Splice_ImgOnly_p-me_1_journalResults.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_splice-journalResults.csv > comp_maskreport_splice-journalResults.txt
diff ../../data/test_suite/maskScorerTests/threstest/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_score.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_thres.csv > comp_maskreport_thres.txt
diff ../../data/test_suite/maskScorerTests/threstest/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_thres-perimage.csv > comp_maskreport_thres-perimage.txt
diff ../../data/test_suite/maskScorerTests/threstest/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_thres-journalResults.csv > comp_maskreport_thres-journalResults.txt

diff ../../data/test_suite/maskScorerTests/splicebin/C_NC2017_Unittest_Splice_ImgOnly_p-me_1_mask_score.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_splicebin.csv > comp_maskreport_splicebin.txt
diff ../../data/test_suite/maskScorerTests/splicebin/C_NC2017_Unittest_Splice_ImgOnly_p-me_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_splicebin-perimage.csv > comp_maskreport_splicebin-perimage.txt
diff ../../data/test_suite/maskScorerTests/splicebin/C_NC2017_Unittest_Splice_ImgOnly_p-me_1_journalResults.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_splicebin-journalResults.csv > comp_maskreport_splicebin-journalResults.txt

flag_s=1
flag_spi=1
flag_sjr=1
flag_m=1
flag_mpi=1
flag_mjr=1
flag_t=1
flag_tpi=1
flag_tjr=1

flag_sb=1
flag_sbpi=1
flag_sbjr=1

filter_s="cat comp_maskreport_splice.txt | grep -v CVS"
filter_spi="cat comp_maskreport_splice-perimage.txt | grep -v CVS"
filter_sjr="cat comp_maskreport_splice-journalResults.txt | grep -v CVS"
filter_m="cat comp_maskreport_manip.txt | grep -v CVS"
filter_mpi="cat comp_maskreport_manip-perimage.txt | grep -v CVS"
filter_mjr="cat comp_maskreport_manip-journalResults.txt | grep -v CVS"
filter_t="cat comp_maskreport_thres.txt | grep -v CVS"
filter_tpi="cat comp_maskreport_thres-perimage.txt | grep -v CVS"
filter_tjr="cat comp_maskreport_thres-journalResults.txt | grep -v CVS"

filter_sb="cat comp_maskreport_splicebin.txt | grep -v CVS"
filter_sbpi="cat comp_maskreport_splicebin-perimage.txt | grep -v CVS"
filter_sbjr="cat comp_maskreport_splicebin-journalResults.txt | grep -v CVS"

if ([ ! -f comp_maskreport_splice.txt -o ! -f comp_maskreport_manip.txt -o ! -f comp_maskreport_splice-journalResults.txt \
 -o ! -f comp_maskreport_splice-perimage.txt -o ! -f comp_maskreport_manip-perimage.txt -o ! -f comp_maskreport_manip-journalResults.txt \
 -o ! -f comp_maskreport_thres.txt -o ! -f comp_maskreport_thres-perimage.txt -o ! -f comp_maskreport_thres-journalResults.txt \
 -o ! -f comp_maskreport_splicebin.txt -o ! -f comp_maskreport_manip.txt -o ! -f comp_maskreport_splicebin-journalResults.txt \
]); then
  echo
  echo "    !!!!! MASK SCORER TEST FAILED AT CASE 0 !!!!!    "
  echo "     MISSING FILES ABSENT     "
  echo
  exit
fi

if test "`eval $filter_s`" = "" ; then
  flag_s=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/splicetest/B_NC2017_Unittest_Splice_ImgOnly_p-me_1_mask_score.csv
	fi
	rm comp_maskreport_splice.txt
else
	echo comp_maskreport_splice.txt
	cat comp_maskreport_splice.txt
fi

if test "`eval $filter_spi`" = "" ; then
  flag_spi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/splicetest/B_NC2017_Unittest_Splice_ImgOnly_p-me_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_splice-perimage.txt
else
	echo comp_maskreport_splice-perimage.txt
	cat comp_maskreport_splice-perimage.txt
fi

if test "`eval $filter_sjr`" = "" ; then
  flag_sjr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/splicetest/B_NC2017_Unittest_Splice_ImgOnly_p-me_1_journalResults.csv
	fi
	rm comp_maskreport_splice-journalResults.txt
else
	echo comp_maskreport_splice-journalResults.txt
	cat comp_maskreport_splice-journalResults.txt
fi

if test "`eval $filter_m`" = "" ; then
  flag_m=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/maniptest/B_NC2016_Unittest_Manipulation_ImgOnly_c-me2_1_mask_score.csv
	fi
	rm comp_maskreport_manip.txt
else
	echo comp_maskreport_manip.txt
	cat comp_maskreport_manip.txt
fi

if test "`eval $filter_mpi`" = "" ; then
  flag_mpi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/maniptest/B_NC2016_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_manip-perimage.txt
else
	echo comp_maskreport_manip-perimage.txt
	cat comp_maskreport_manip-perimage.txt
fi

if test "`eval $filter_mjr`" = "" ; then
  flag_mjr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/maniptest/B_NC2016_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv
	fi
	rm comp_maskreport_manip-journalResults.txt
else
	echo comp_maskreport_manip-journalResults.txt
	cat comp_maskreport_manip-journalResults.txt
fi

if test "`eval $filter_t`" = "" ; then
  flag_t=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/threstest/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_score.csv
	fi
	rm comp_maskreport_thres.txt
else
	echo comp_maskreport_thres.txt
	cat comp_maskreport_thres.txt
fi

if test "`eval $filter_tpi`" = "" ; then
  flag_tpi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/threstest/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_thres-perimage.txt
else
	echo comp_maskreport_thres-perimage.txt
	cat comp_maskreport_thres-perimage.txt
fi

if test "`eval $filter_tjr`" = "" ; then
  flag_tjr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/threstest/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv
	fi
	rm comp_maskreport_thres-journalResults.txt
else
	echo comp_maskreport_thres-journalResults.txt
	cat comp_maskreport_thres-journalResults.txt
fi

if test "`eval $filter_sb`" = "" ; then
  flag_sb=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/splicebin/C_NC2017_Unittest_Splice_ImgOnly_p-me_1_mask_score.csv
	fi
	rm comp_maskreport_splicebin.txt
else
	echo comp_maskreport_splicebin.txt
	cat comp_maskreport_splicebin.txt
fi

if test "`eval $filter_sbpi`" = "" ; then
  flag_sbpi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/splicebin/C_NC2017_Unittest_Splice_ImgOnly_p-me_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_splicebin-perimage.txt
else
	echo comp_maskreport_splicebin-perimage.txt
	cat comp_maskreport_splicebin-perimage.txt
fi

if test "`eval $filter_sbjr`" = "" ; then
  flag_sbjr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/splicebin/C_NC2017_Unittest_Splice_ImgOnly_p-me_1_journalResults.csv
	fi
	rm comp_maskreport_splicebin-journalResults.txt
else
	echo comp_maskreport_splicebin-journalResults.txt
	cat comp_maskreport_splicebin-journalResults.txt
fi

#manipulation and splice optOut case
$mypython MaskScorer.py -t manipulation --refDir ../../data/test_suite/maskScorerTests -r reference/manipulation/NC2017-manipulation-ref.csv -x indexes/NC2017-manipulation-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1.csv -oR ../../data/test_suite/maskScorerTests/manip_optOut/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1 -html -p $procs --speedup --optOut --precision 12

diff ../../data/test_suite/maskScorerTests/manip_optOut/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_score.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_manip_optOut.csv > comp_maskreport_manip_optOut.txt
diff ../../data/test_suite/maskScorerTests/manip_optOut/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_manip_optOut-perimage.csv > comp_maskreport_manip_optOut-perimage.txt
diff ../../data/test_suite/maskScorerTests/manip_optOut/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_manip_optOut-journalResults.csv > comp_maskreport_manip_optOut-journalResults.txt

flag_moo=1
flag_moopi=1
flag_moojr=1

filter_moo="cat comp_maskreport_manip_optOut.txt | grep -v CVS"
filter_moopi="cat comp_maskreport_manip_optOut-perimage.txt | grep -v CVS"
filter_moojr="cat comp_maskreport_manip_optOut-journalResults.txt | grep -v CVS"

if ([ ! -f comp_maskreport_manip_optOut.txt -o ! -f comp_maskreport_manip_optOut-perimage.txt -o ! -f comp_maskreport_manip_optOut-journalResults.txt ]); then
  echo
  echo "    !!!!! MASK SCORER TEST FAILED AT CASE 0 !!!!!    "
  echo "     MISSING FILES ABSENT     "
  echo
  exit
fi

if test "`eval $filter_moo`" = "" ; then
  flag_moo=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/manip_optOut/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_score.csv
	fi
	rm comp_maskreport_manip_optOut.txt
else
	echo comp_maskreport_manip_optOut.txt
	cat comp_maskreport_manip_optOut.txt
fi

if test "`eval $filter_moopi`" = "" ; then
  flag_moopi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/manip_optOut/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_manip_optOut-perimage.txt
else
	echo comp_maskreport_manip_optOut-perimage.txt
	cat comp_maskreport_manip_optOut-perimage.txt
fi

if test "`eval $filter_moojr`" = "" ; then
  flag_moojr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/manip_optOut/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv
	fi
	rm comp_maskreport_manip_optOut-journalResults.txt
else
	echo comp_maskreport_manip_optOut-journalResults.txt
	cat comp_maskreport_manip_optOut-journalResults.txt
fi

manip_optOut_total=$(($flag_moo + $flag_moopi + $flag_moojr))


$mypython MaskScorer.py -t splice --refDir ../../data/test_suite/maskScorerTests -r reference/splice/NC2017-splice-ref.csv -x indexes/NC2017-splice-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2017_Unittest_Splice_ImgOnly_p-me_1/B_NC2017_Unittest_Splice_ImgOnly_p-me_1.csv -oR ../../data/test_suite/maskScorerTests/splice_optOut/B_NC2017_Unittest_Splice_ImgOnly_p-me_1 -html -p $procs --speedup --optOut --precision 12

diff ../../data/test_suite/maskScorerTests/splice_optOut/B_NC2017_Unittest_Splice_ImgOnly_p-me_1_mask_score.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_splice_optOut.csv > comp_maskreport_splice_optOut.txt
diff ../../data/test_suite/maskScorerTests/splice_optOut/B_NC2017_Unittest_Splice_ImgOnly_p-me_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_splice_optOut-perimage.csv > comp_maskreport_splice_optOut-perimage.txt
diff ../../data/test_suite/maskScorerTests/splice_optOut/B_NC2017_Unittest_Splice_ImgOnly_p-me_1_journalResults.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_splice_optOut-journalResults.csv > comp_maskreport_splice_optOut-journalResults.txt

flag_soo=1
flag_soopi=1
flag_soojr=1

filter_soo="cat comp_maskreport_splice_optOut.txt | grep -v CVS"
filter_soopi="cat comp_maskreport_splice_optOut-perimage.txt | grep -v CVS"
filter_soojr="cat comp_maskreport_splice_optOut-journalResults.txt | grep -v CVS"

if ([ ! -f comp_maskreport_splice_optOut.txt -o ! -f comp_maskreport_splice_optOut-perimage.txt -o ! -f comp_maskreport_splice_optOut-journalResults.txt ]); then
  echo
  echo "    !!!!! MASK SCORER TEST FAILED AT CASE 0 !!!!!    "
  echo "     MISSING FILES ABSENT     "
  echo
  exit
fi

if test "`eval $filter_soo`" = "" ; then
  flag_soo=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/splice_optOut/B_NC2017_Unittest_Splice_ImgOnly_p-me_1_mask_score.csv
	fi
	rm comp_maskreport_splice_optOut.txt
else
	echo comp_maskreport_splice_optOut.txt
	cat comp_maskreport_splice_optOut.txt
fi

if test "`eval $filter_soopi`" = "" ; then
  flag_soopi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/splice_optOut/B_NC2017_Unittest_Splice_ImgOnly_p-me_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_splice_optOut-perimage.txt
else
	echo comp_maskreport_splice_optOut-perimage.txt
	cat comp_maskreport_splice_optOut-perimage.txt
fi

if test "`eval $filter_soojr`" = "" ; then
  flag_soojr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/splice_optOut/B_NC2017_Unittest_Splice_ImgOnly_p-me_1_journalResults.csv
	fi
	rm comp_maskreport_splice_optOut-journalResults.txt
else
	echo comp_maskreport_splice_optOut-journalResults.txt
	cat comp_maskreport_splice_optOut-journalResults.txt
fi

splice_optOut_total=$(($flag_soo + $flag_soopi + $flag_soojr))

#faulty test case.
errflag=1
$mypython MaskScorer.py -t manipulation --refDir ../../data/test_suite/maskScorerTests/ -r reference/manipulation/NC2017-manipulation-ref.csv -x indexes/NC2017-manipulation-index.csv -s ../../data/test_suite/maskScorerTests/Error_NC2017_Unittest_Manipulation_ImgOnly_c-me_1/Error_NC2017_Unittest_Manipulation_ImgOnly_c-me_1.csv -oR ../../data/test_suite/maskScorerTests/errtest/Error_NC2017_Unittest_Manipulation_ImgOnly_c-me_1 --speedup $optOutClause -v 1 --debug_off > errlog.txt
if `grep -q ERROR.*unreadable errlog.txt` && `grep -q Ending errlog.txt` ; then
    errflag=0
fi

flag_total=$(($flag_s + $flag_spi + $flag_sjr + $flag_m + $flag_mpi + $flag_mjr + $flag_t + $flag_tpi + $flag_tjr + $flag_sb + $flag_sbpi + $flag_sbjr\
 + $manip_optOut_total + $splice_optOut_total\
 + $errflag))

if ([ $flag_total -eq 0 ]); then
  echo
  echo "CASE 0 SUCCESSFULLY PASSED"
  echo
	if [ $clean = "TRUE" ] ; then
		rm -rf ../../data/test_suite/maskScorerTests/maniptest
		rm -rf ../../data/test_suite/maskScorerTests/splicetest
		rm -rf ../../data/test_suite/maskScorerTests/threstest
		rm -rf ../../data/test_suite/maskScorerTests/splicebin
		rm -rf ../../data/test_suite/maskScorerTests/manip_optOut
		rm -rf ../../data/test_suite/maskScorerTests/splice_optOut
                rm -rf ../../data/test_suite/maskScorerTests/errtest
	fi
else
  echo
  echo "    !!!!! MASK SCORER TEST FAILED AT CASE 0 !!!!!    "
  echo
  exit
fi

