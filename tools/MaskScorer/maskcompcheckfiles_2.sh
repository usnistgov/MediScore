#!/bin/bash
clean=TRUE

echo
echo "CASE 2: VALIDATING FACTOR-BASED SCORING"
echo

mypython=python2

$mypython MaskScorer.py -t manipulation --refDir ../../data/test_suite/maskScorerTests -r reference/manipulation/NC2017-manipulation-ref.csv -x indexes/NC2017-manipulation-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1.csv -oR ../../data/test_suite/maskScorerTests/manipconf/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1 -q "0.5 < ConfidenceScore" --speedup --precision 12 --cache_dir /tmp/manip_queries --cache_flush
$mypython MaskScorer.py -t manipulation --refDir ../../data/test_suite/maskScorerTests -r reference/manipulation/NC2017-manipulation-ref.csv -x indexes/NC2017-manipulation-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1.csv -oR ../../data/test_suite/maskScorerTests/manipconfmanmade/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1 -html -q "ConfidenceScore < 0.5" "ManMade=='no'" --optOut --speedup --precision 12 --cache_dir /tmp/manip_queries
$mypython MaskScorer.py -t manipulation --refDir ../../data/test_suite/maskScorerTests -r reference/manipulation/NC2017-manipulation-ref.csv -x indexes/NC2017-manipulation-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1.csv -oR ../../data/test_suite/maskScorerTests/manipfooconf/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1 -q "ConfidenceScore==0.5" --speedup --precision 12 --cache_dir /tmp/manip_queries
$mypython MaskScorer.py -t manipulation --refDir ../../data/test_suite/maskScorerTests -r reference/manipulation/NC2017-manipulation-ref.csv -x indexes/NC2017-manipulation-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1.csv -oR ../../data/test_suite/maskScorerTests/manipconfpart/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1 -qp "0.5 < ConfidenceScore" --speedup --precision 12 --cache_dir /tmp/manip_queries
$mypython MaskScorer.py -t manipulation --refDir ../../data/test_suite/maskScorerTests -r reference/manipulation/NC2017-manipulation-ref.csv -x indexes/NC2017-manipulation-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1.csv -oR ../../data/test_suite/maskScorerTests/manipconfclonepart/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1 -qp "ConfidenceScore > 0.3 & Clone==['yes','no']" --speedup --precision 12 --cache_dir /tmp/manip_queries
$mypython MaskScorer.py -t manipulation --refDir ../../data/test_suite/maskScorerTests -r reference/manipulation/NC2017-manipulation-ref.csv -x indexes/NC2017-manipulation-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1.csv -oR ../../data/test_suite/maskScorerTests/manipfooconfpart/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1 -qp "ConfidenceScore==['foo']" --speedup --precision 12 --cache_dir /tmp/manip_queries

$mypython MaskScorer.py -t splice --refDir ../../data/test_suite/maskScorerTests -r reference/splice/NC2016-splice-ref.csv -x indexes/NC2016-splice-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2016_Unittest_Splice_ImgOnly_p-me_1/B_NC2016_Unittest_Splice_ImgOnly_p-me_1.csv -oR ../../data/test_suite/maskScorerTests/spliceconf/B_NC2016_Unittest_Splice_ImgOnly_p-me_1 -q "0.5 < ConfidenceScore" --speedup --precision 12 --cache_dir /tmp/splice_queries --cache_flush
$mypython MaskScorer.py -t splice --refDir ../../data/test_suite/maskScorerTests -r reference/splice/NC2016-splice-ref.csv -x indexes/NC2016-splice-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2016_Unittest_Splice_ImgOnly_p-me_1/B_NC2016_Unittest_Splice_ImgOnly_p-me_1.csv -oR ../../data/test_suite/maskScorerTests/spliceconfcoll/B_NC2016_Unittest_Splice_ImgOnly_p-me_1 -q "0.3 <= ConfidenceScore" "Collection==['Nimble-WEB']" --speedup --precision 12 --cache_dir /tmp/splice_queries
$mypython MaskScorer.py -t splice --refDir ../../data/test_suite/maskScorerTests -r reference/splice/NC2016-splice-ref.csv -x indexes/NC2016-splice-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2016_Unittest_Splice_ImgOnly_p-me_1/B_NC2016_Unittest_Splice_ImgOnly_p-me_1.csv -oR ../../data/test_suite/maskScorerTests/splicefooconf/B_NC2016_Unittest_Splice_ImgOnly_p-me_1 -q "ConfidenceScore==['foo']" --speedup --displayScoredOnly --precision 12 --cache_dir /tmp/splice_queries
$mypython MaskScorer.py -t splice --refDir ../../data/test_suite/maskScorerTests -r reference/splice/NC2016-splice-ref.csv -x indexes/NC2016-splice-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2016_Unittest_Splice_ImgOnly_p-me_1/B_NC2016_Unittest_Splice_ImgOnly_p-me_1.csv -oR ../../data/test_suite/maskScorerTests/spliceconfpart/B_NC2016_Unittest_Splice_ImgOnly_p-me_1 -qp "ConfidenceScore > 0.5" --speedup --precision 12 --cache_dir /tmp/splice_queries
$mypython MaskScorer.py -t splice --refDir ../../data/test_suite/maskScorerTests -r reference/splice/NC2016-splice-ref.csv -x indexes/NC2016-splice-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2016_Unittest_Splice_ImgOnly_p-me_1/B_NC2016_Unittest_Splice_ImgOnly_p-me_1.csv -oR ../../data/test_suite/maskScorerTests/spliceconfcollpart/B_NC2016_Unittest_Splice_ImgOnly_p-me_1 -qp "ConfidenceScore > 0.3 & Collection==['Nimble-WEB','Nimble-SCI']" --speedup --precision 12 --cache_dir /tmp/splice_queries
$mypython MaskScorer.py -t splice --refDir ../../data/test_suite/maskScorerTests -r reference/splice/NC2016-splice-ref.csv -x indexes/NC2016-splice-index.csv -s ../../data/test_suite/maskScorerTests/B_NC2016_Unittest_Splice_ImgOnly_p-me_1/B_NC2016_Unittest_Splice_ImgOnly_p-me_1.csv -oR ../../data/test_suite/maskScorerTests/splicefooconfpart/B_NC2016_Unittest_Splice_ImgOnly_p-me_1 -qp "ConfidenceScore==['foo']" --speedup --precision 12 --cache_dir /tmp/splice_queries

#manip confs
diff ../../data/test_suite/maskScorerTests/manipconf/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_0.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_manipconf_0.csv > comp_maskreport_manipconf_0.txt
diff ../../data/test_suite/maskScorerTests/manipconf/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_manipconf-perimage.csv > comp_maskreport_manipconf-perimage.txt
diff ../../data/test_suite/maskScorerTests/manipconf/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_manipconf-journalResults.csv > comp_maskreport_manipconf-journalResults.txt

diff ../../data/test_suite/maskScorerTests/manipconfmanmade/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_0.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_manipconfmanmade_0.csv > comp_maskreport_manipconfmanmade_0.txt
diff ../../data/test_suite/maskScorerTests/manipconfmanmade/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_1.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_manipconfmanmade_1.csv > comp_maskreport_manipconfmanmade_1.txt
#diff ../../data/test_suite/maskScorerTests/manipconfmanmade/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_optout_0.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_manipconfmanmade_optout_0.csv > comp_maskreport_manipconfmanmade_optout_0.txt
#diff ../../data/test_suite/maskScorerTests/manipconfmanmade/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_optout_1.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_manipconfmanmade_optout_1.csv > comp_maskreport_manipconfmanmade_optout_1.txt
diff ../../data/test_suite/maskScorerTests/manipconfmanmade/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_manipconfmanmade-perimage.csv > comp_maskreport_manipconfmanmade-perimage.txt
diff ../../data/test_suite/maskScorerTests/manipconfmanmade/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_manipconfmanmade-journalResults.csv > comp_maskreport_manipconfmanmade-journalResults.txt

diff ../../data/test_suite/maskScorerTests/manipfooconf/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_manipfooconf-perimage.csv > comp_maskreport_manipfooconf-perimage.txt
diff ../../data/test_suite/maskScorerTests/manipfooconf/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_manipfooconf-journalResults.csv > comp_maskreport_manipfooconf-journalResults.txt

diff ../../data/test_suite/maskScorerTests/manipconfpart/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_score.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_manipconfpart.csv > comp_maskreport_manipconfpart.txt
diff ../../data/test_suite/maskScorerTests/manipconfpart/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_manipconfpart-perimage.csv > comp_maskreport_manipconfpart-perimage.txt
diff ../../data/test_suite/maskScorerTests/manipconfpart/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_manipconfpart-journalResults.csv > comp_maskreport_manipconfpart-journalResults.txt

diff ../../data/test_suite/maskScorerTests/manipconfclonepart/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_score.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_manipconfclonepart.csv > comp_maskreport_manipconfclonepart.txt
diff ../../data/test_suite/maskScorerTests/manipconfclonepart/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_manipconfclonepart-perimage.csv > comp_maskreport_manipconfclonepart-perimage.txt
diff ../../data/test_suite/maskScorerTests/manipconfclonepart/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_manipconfclonepart-journalResults.csv > comp_maskreport_manipconfclonepart-journalResults.txt

diff ../../data/test_suite/maskScorerTests/manipfooconfpart/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_manipfooconfpart-perimage.csv > comp_maskreport_manipfooconfpart-perimage.txt
diff ../../data/test_suite/maskScorerTests/manipfooconfpart/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_manipfooconfpart-journalResults.csv > comp_maskreport_manipfooconfpart-journalResults.txt


#splice confs
diff ../../data/test_suite/maskScorerTests/spliceconf/B_NC2016_Unittest_Splice_ImgOnly_p-me_1_mask_scores_0.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_spliceconf_0.csv > comp_maskreport_spliceconf_0.txt
diff ../../data/test_suite/maskScorerTests/spliceconf/B_NC2016_Unittest_Splice_ImgOnly_p-me_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_spliceconf-perimage.csv > comp_maskreport_spliceconf-perimage.txt

diff ../../data/test_suite/maskScorerTests/spliceconfcoll/B_NC2016_Unittest_Splice_ImgOnly_p-me_1_mask_scores_0.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_spliceconfcoll_0.csv > comp_maskreport_spliceconfcoll_0.txt
diff ../../data/test_suite/maskScorerTests/spliceconfcoll/B_NC2016_Unittest_Splice_ImgOnly_p-me_1_mask_scores_1.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_spliceconfcoll_1.csv > comp_maskreport_spliceconfcoll_1.txt
diff ../../data/test_suite/maskScorerTests/spliceconfcoll/B_NC2016_Unittest_Splice_ImgOnly_p-me_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_spliceconfcoll-perimage.csv > comp_maskreport_spliceconfcoll-perimage.txt

diff ../../data/test_suite/maskScorerTests/splicefooconf/B_NC2016_Unittest_Splice_ImgOnly_p-me_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_splicefooconf-perimage.csv > comp_maskreport_splicefooconf-perimage.txt

diff ../../data/test_suite/maskScorerTests/spliceconfpart/B_NC2016_Unittest_Splice_ImgOnly_p-me_1_mask_score.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_spliceconfpart.csv > comp_maskreport_spliceconfpart.txt
diff ../../data/test_suite/maskScorerTests/spliceconfpart/B_NC2016_Unittest_Splice_ImgOnly_p-me_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_spliceconfpart-perimage.csv > comp_maskreport_spliceconfpart-perimage.txt

diff ../../data/test_suite/maskScorerTests/spliceconfcollpart/B_NC2016_Unittest_Splice_ImgOnly_p-me_1_mask_score.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_spliceconfcollpart.csv > comp_maskreport_spliceconfcollpart.txt
diff ../../data/test_suite/maskScorerTests/spliceconfcollpart/B_NC2016_Unittest_Splice_ImgOnly_p-me_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_spliceconfcollpart-perimage.csv > comp_maskreport_spliceconfcollpart-perimage.txt

diff ../../data/test_suite/maskScorerTests/splicefooconfpart/B_NC2016_Unittest_Splice_ImgOnly_p-me_1_mask_scores_perimage.csv ../../data/test_suite/maskScorerTests/compcheckfiles/ref_maskreport_splicefooconfpart-perimage.csv > comp_maskreport_splicefooconfpart-perimage.txt


#flags
flag_manipconf_0=1
flag_manipconfpi=1
flag_manipconfjr=1
flag_manipconfmanmade_0=1
flag_manipconfmanmade_1=1
flag_manipconfmanmadepi=1
flag_manipconfmanmadejr=1
flag_manipfooconfpi=1
flag_manipfooconfjr=1
flag_manipconfpart=1
flag_manipconfpartpi=1
flag_manipconfpartjr=1
flag_manipconfclonepart=1
flag_manipconfclonepartpi=1
flag_manipconfclonepartjr=1
flag_manipfooconfpartpi=1
flag_manipfooconfpartjr=1

flag_spliceconf_0=1
flag_spliceconfpi=1
flag_spliceconfcoll_0=1
flag_spliceconfcoll_1=1
flag_spliceconfcollpi=1
flag_splicefooconfpi=1
flag_spliceconfpart=1
flag_spliceconfpartpi=1
flag_spliceconfcollpart=1
flag_spliceconfcollpartpi=1
flag_splicefooconfpartpi=1

#filters to evaluate
filter_manipconf_0="cat comp_maskreport_manipconf_0.txt | grep -v CVS"
filter_manipconfpi="cat comp_maskreport_manipconf-perimage.txt | grep -v CVS"
filter_manipconfjr="cat comp_maskreport_manipconf-journalResults.txt | grep -v CVS"
filter_manipconfmanmade_0="cat comp_maskreport_manipconfmanmade_0.txt | grep -v CVS"
filter_manipconfmanmade_1="cat comp_maskreport_manipconfmanmade_1.txt | grep -v CVS"
#filter_manipconfmanmade_optout_0="cat comp_maskreport_manipconfmanmade_optout_0.txt | grep -v CVS"
#filter_manipconfmanmade_optout_1="cat comp_maskreport_manipconfmanmade_optout_1.txt | grep -v CVS"
filter_manipconfmanmadepi="cat comp_maskreport_manipconfmanmade-perimage.txt | grep -v CVS"
filter_manipconfmanmadejr="cat comp_maskreport_manipconfmanmade-journalResults.txt | grep -v CVS"
filter_manipfooconfpi="cat comp_maskreport_manipfooconf-perimage.txt | grep -v CVS"
filter_manipfooconfjr="cat comp_maskreport_manipfooconf-journalResults.txt | grep -v CVS"
filter_manipconfpart="cat comp_maskreport_manipconfpart.txt | grep -v CVS"
filter_manipconfpartpi="cat comp_maskreport_manipconfpart-perimage.txt | grep -v CVS"
filter_manipconfpartjr="cat comp_maskreport_manipconfpart-journalResults.txt | grep -v CVS"
filter_manipconfclonepart="cat comp_maskreport_manipconfclonepart.txt | grep -v CVS"
filter_manipconfclonepartpi="cat comp_maskreport_manipconfclonepart-perimage.txt | grep -v CVS"
filter_manipconfclonepartjr="cat comp_maskreport_manipconfclonepart-journalResults.txt | grep -v CVS"
filter_manipfooconfpartpi="cat comp_maskreport_manipfooconfpart-perimage.txt | grep -v CVS"
filter_manipfooconfpartjr="cat comp_maskreport_manipfooconfpart-journalResults.txt | grep -v CVS"

filter_spliceconf_0="cat comp_maskreport_spliceconf_0.txt | grep -v CVS"
filter_spliceconfpi="cat comp_maskreport_spliceconf-perimage.txt | grep -v CVS"
filter_spliceconfcoll_0="cat comp_maskreport_spliceconfcoll_0.txt | grep -v CVS"
filter_spliceconfcoll_1="cat comp_maskreport_spliceconfcoll_1.txt | grep -v CVS"
filter_spliceconfcollpi="cat comp_maskreport_spliceconfcoll-perimage.txt | grep -v CVS"
filter_splicefooconfpi="cat comp_maskreport_splicefooconf-perimage.txt | grep -v CVS"
filter_spliceconfpart="cat comp_maskreport_spliceconfpart.txt | grep -v CVS"
filter_spliceconfpartpi="cat comp_maskreport_spliceconfpart-perimage.txt | grep -v CVS"
filter_spliceconfcollpart="cat comp_maskreport_spliceconfcollpart.txt | grep -v CVS"
filter_spliceconfcollpartpi="cat comp_maskreport_spliceconfcollpart-perimage.txt | grep -v CVS"
filter_splicefooconfpartpi="cat comp_maskreport_splicefooconfpart-perimage.txt | grep -v CVS"

if ([ -f comp_maskreport_manipfooconf_0.txt -o -f comp_maskreport_manipfooconfpart.txt -o -f comp_maskreport_splicefooconf_0.txt \
 -o -f comp_maskreport_splicefooconfpart.txt \
]); then
  echo
  echo "    !!!!! MASK SCORER TEST FAILED AT CASE 2 !!!!!    "
  echo "     EXTRA FILES PRESENT     "
  echo
  exit
fi
 
if ([ ! -f comp_maskreport_manipconf_0.txt -o ! -f comp_maskreport_manipconf-perimage.txt -o ! -f comp_maskreport_manipconf-journalResults.txt \
 -o ! -f comp_maskreport_manipconfmanmade_0.txt -o ! -f comp_maskreport_manipconfmanmade_1.txt\
 -o ! -f comp_maskreport_manipconfmanmade-perimage.txt -o ! -f comp_maskreport_manipconfmanmade-journalResults.txt \
 -o ! -f comp_maskreport_manipfooconf-perimage.txt -o ! -f comp_maskreport_manipfooconf-journalResults.txt \
 -o ! -f comp_maskreport_manipconfpart.txt -o ! -f comp_maskreport_manipconfpart-perimage.txt -o ! -f comp_maskreport_manipconfpart-journalResults.txt \
 -o ! -f comp_maskreport_manipconfclonepart.txt -o ! -f comp_maskreport_manipconfclonepart-perimage.txt -o ! -f comp_maskreport_manipconfclonepart-journalResults.txt \
 -o ! -f comp_maskreport_manipfooconfpart-journalResults.txt -o ! -f comp_maskreport_manipfooconfpart-perimage.txt \
 -o ! -f comp_maskreport_spliceconf_0.txt -o ! -f comp_maskreport_spliceconf-perimage.txt \
 -o ! -f comp_maskreport_spliceconfcoll_0.txt -o ! -f comp_maskreport_spliceconfcoll_1.txt -o ! -f comp_maskreport_spliceconfcoll-perimage.txt \
 -o ! -f comp_maskreport_splicefooconf-perimage.txt \
 -o ! -f comp_maskreport_spliceconfpart.txt -o ! -f comp_maskreport_spliceconfpart-perimage.txt \
 -o ! -f comp_maskreport_spliceconfcollpart.txt -o ! -f comp_maskreport_spliceconfcollpart-perimage.txt \
 -o ! -f comp_maskreport_splicefooconfpart-perimage.txt \
]); then
  echo
  echo "    !!!!! MASK SCORER TEST FAILED AT CASE 2 !!!!!    "
  echo "     MISSING FILES ABSENT     "
  echo
  exit
fi

if test "`eval $filter_manipconf_0`" = "" ; then
  flag_manipconf_0=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/manipconf/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_0.csv
	fi
	rm comp_maskreport_manipconf_0.txt
else
	echo comp_maskreport_manipconf_0.txt
	cat comp_maskreport_manipconf_0.txt
fi

if test "`eval $filter_manipconfpi`" = "" ; then
  flag_manipconfpi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/manipconf/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_manipconf-perimage.txt
else
	echo comp_maskreport_manipconf-perimage.txt
	cat comp_maskreport_manipconf-perimage.txt
fi

if test "`eval $filter_manipconfjr`" = "" ; then
  flag_manipconfjr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/manipconf/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv
	fi
	rm comp_maskreport_manipconf-journalResults.txt
else
	echo comp_maskreport_manipconf-journalResults.txt
	cat comp_maskreport_manipconf-journalResults.txt
fi

if test "`eval $filter_manipconfmanmade_0`" = "" ; then
  flag_manipconfmanmade_0=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/manipconfmanmade/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_0.csv
	fi
	rm comp_maskreport_manipconfmanmade_0.txt
else
	echo comp_maskreport_manipconfmanmade_0.txt
	cat comp_maskreport_manipconfmanmade_0.txt
fi

if test "`eval $filter_manipconfmanmade_1`" = "" ; then
  flag_manipconfmanmade_1=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/manipconfmanmade/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_1.csv
	fi
	rm comp_maskreport_manipconfmanmade_1.txt
else
	echo comp_maskreport_manipconfmanmade_1.txt
	cat comp_maskreport_manipconfmanmade_1.txt
fi

#if test "`eval $filter_manipconfmanmade_optout_0`" = "" ; then
#  flag_manipconfmanmade_optout_0=0
#	if [ $clean = "TRUE" ] ; then
#		rm ../../data/test_suite/maskScorerTests/manipconfmanmade/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_optout_0.csv
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
#		rm ../../data/test_suite/maskScorerTests/manipconfmanmade/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_optout_1.csv
#	fi
#	rm comp_maskreport_manipconfmanmade_optout_1.txt
#else
#	echo comp_maskreport_manipconfmanmade_optout_1.txt
#	cat comp_maskreport_manipconfmanmade_optout_1.txt
#fi

if test "`eval $filter_manipconfmanmadepi`" = "" ; then
  flag_manipconfmanmadepi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/manipconfmanmade/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_manipconfmanmade-perimage.txt
else
	echo comp_maskreport_manipconfmanmade-perimage.txt
	cat comp_maskreport_manipconfmanmade-perimage.txt
fi

if test "`eval $filter_manipconfmanmadejr`" = "" ; then
  flag_manipconfmanmadejr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/manipconfmanmade/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv
	fi
	rm comp_maskreport_manipconfmanmade-journalResults.txt
else
	echo comp_maskreport_manipconfmanmade-journalResults.txt
	cat comp_maskreport_manipconfmanmade-journalResults.txt
fi

if test "`eval $filter_manipfooconfpi`" = "" ; then
  flag_manipfooconfpi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/manipfooconf/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_manipfooconf-perimage.txt
else
	echo comp_maskreport_manipfooconf-perimage.txt
	cat comp_maskreport_manipfooconf-perimage.txt
fi

if test "`eval $filter_manipfooconfjr`" = "" ; then
  flag_manipfooconfjr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/manipfooconf/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv
	fi
	rm comp_maskreport_manipfooconf-journalResults.txt
else
	echo comp_maskreport_manipfooconf-journalResults.txt
	cat comp_maskreport_manipfooconf-journalResults.txt
fi

if test "`eval $filter_manipconfpart`" = "" ; then
  flag_manipconfpart=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/manipconfpart/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_score.csv
	fi
	rm comp_maskreport_manipconfpart.txt
else
	echo comp_maskreport_manipconfpart.txt
	cat comp_maskreport_manipconfpart.txt
fi

if test "`eval $filter_manipconfpartpi`" = "" ; then
  flag_manipconfpartpi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/manipconfpart/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_manipconfpart-perimage.txt
else
	echo comp_maskreport_manipconfpart-perimage.txt
	cat comp_maskreport_manipconfpart-perimage.txt
fi

if test "`eval $filter_manipconfpartjr`" = "" ; then
  flag_manipconfpartjr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/manipconfpart/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv
	fi
	rm comp_maskreport_manipconfpart-journalResults.txt
else
	echo comp_maskreport_manipconfpart-journalResults.txt
	cat comp_maskreport_manipconfpart-journalResults.txt
fi

if test "`eval $filter_manipconfclonepart`" = "" ; then
  flag_manipconfclonepart=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/manipconfclonepart/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_score.csv
	fi
	rm comp_maskreport_manipconfclonepart.txt
else
	echo comp_maskreport_manipconfclonepart.txt
	cat comp_maskreport_manipconfclonepart.txt
fi

if test "`eval $filter_manipconfclonepartpi`" = "" ; then
  flag_manipconfclonepartpi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/manipconfclonepart/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_manipconfclonepart-perimage.txt
else
	echo comp_maskreport_manipconfclonepart-perimage.txt
	cat comp_maskreport_manipconfclonepart-perimage.txt
fi

if test "`eval $filter_manipconfclonepartjr`" = "" ; then
  flag_manipconfclonepartjr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/manipconfclonepart/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv
	fi
	rm comp_maskreport_manipconfclonepart-journalResults.txt
else
	echo comp_maskreport_manipconfclonepart-journalResults.txt
	cat comp_maskreport_manipconfclonepart-journalResults.txt
fi

if test "`eval $filter_manipfooconfpartpi`" = "" ; then
  flag_manipfooconfpartpi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/manipfooconfpart/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_manipfooconfpart-perimage.txt
else
	echo comp_maskreport_manipfooconfpart-perimage.txt
	cat comp_maskreport_manipfooconfpart-perimage.txt
fi

if test "`eval $filter_manipfooconfpartjr`" = "" ; then
  flag_manipfooconfpartjr=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/manipfooconfpart/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1_journalResults.csv
	fi
	rm comp_maskreport_manipfooconfpart-journalResults.txt
else
	echo comp_maskreport_manipfooconfpart-journalResults.txt
	cat comp_maskreport_manipfooconfpart-journalResults.txt
fi

if test "`eval $filter_spliceconf_0`" = "" ; then
  flag_spliceconf_0=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/spliceconf/B_NC2016_Unittest_Splice_ImgOnly_p-me_1_mask_scores_0.csv
	fi
	rm comp_maskreport_spliceconf_0.txt
else
	echo comp_maskreport_spliceconf_0.txt
	cat comp_maskreport_spliceconf_0.txt
fi

if test "`eval $filter_spliceconfpi`" = "" ; then
  flag_spliceconfpi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/spliceconf/B_NC2016_Unittest_Splice_ImgOnly_p-me_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_spliceconf-perimage.txt
else
	echo comp_maskreport_spliceconf-perimage.txt
	cat comp_maskreport_spliceconf-perimage.txt
fi

if test "`eval $filter_spliceconfcoll_0`" = "" ; then
  flag_spliceconfcoll_0=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/spliceconfcoll/B_NC2016_Unittest_Splice_ImgOnly_p-me_1_mask_scores_0.csv
	fi
	rm comp_maskreport_spliceconfcoll_0.txt
else
	echo comp_maskreport_spliceconfcoll_0.txt
	cat comp_maskreport_spliceconfcoll_0.txt
fi

if test "`eval $filter_spliceconfcoll_1`" = "" ; then
  flag_spliceconfcoll_1=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/spliceconfcoll/B_NC2016_Unittest_Splice_ImgOnly_p-me_1_mask_scores_1.csv
	fi
	rm comp_maskreport_spliceconfcoll_1.txt
else
	echo comp_maskreport_spliceconfcoll_1.txt
	cat comp_maskreport_spliceconfcoll_1.txt
fi

if test "`eval $filter_spliceconfcollpi`" = "" ; then
  flag_spliceconfcollpi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/spliceconfcoll/B_NC2016_Unittest_Splice_ImgOnly_p-me_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_spliceconfcoll-perimage.txt
else
	echo comp_maskreport_spliceconfcoll-perimage.txt
	cat comp_maskreport_spliceconfcoll-perimage.txt
fi

if test "`eval $filter_splicefooconfpi`" = "" ; then
  flag_splicefooconfpi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/splicefooconf/B_NC2016_Unittest_Splice_ImgOnly_p-me_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_splicefooconf-perimage.txt
else
	echo comp_maskreport_splicefooconf-perimage.txt
	cat comp_maskreport_splicefooconf-perimage.txt
fi

if test "`eval $filter_spliceconfpart`" = "" ; then
  flag_spliceconfpart=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/spliceconfpart/B_NC2016_Unittest_Splice_ImgOnly_p-me_1_mask_score.csv
	fi
	rm comp_maskreport_spliceconfpart.txt
else
	echo comp_maskreport_spliceconfpart.txt
	cat comp_maskreport_spliceconfpart.txt
fi

if test "`eval $filter_spliceconfpartpi`" = "" ; then
  flag_spliceconfpartpi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/spliceconfpart/B_NC2016_Unittest_Splice_ImgOnly_p-me_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_spliceconfpart-perimage.txt
else
	echo comp_maskreport_spliceconfpart-perimage.txt
	cat comp_maskreport_spliceconfpart-perimage.txt
fi

if test "`eval $filter_spliceconfcollpart`" = "" ; then
  flag_spliceconfcollpart=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/spliceconfcollpart/B_NC2016_Unittest_Splice_ImgOnly_p-me_1_mask_score.csv
	fi
	rm comp_maskreport_spliceconfcollpart.txt
else
	echo comp_maskreport_spliceconfcollpart.txt
	cat comp_maskreport_spliceconfcollpart.txt
fi

if test "`eval $filter_spliceconfcollpartpi`" = "" ; then
  flag_spliceconfcollpartpi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/spliceconfcollpart/B_NC2016_Unittest_Splice_ImgOnly_p-me_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_spliceconfcollpart-perimage.txt
else
	echo comp_maskreport_spliceconfcollpart-perimage.txt
	cat comp_maskreport_spliceconfcollpart-perimage.txt
fi

if test "`eval $filter_splicefooconfpartpi`" = "" ; then
  flag_splicefooconfpartpi=0
	if [ $clean = "TRUE" ] ; then
		rm ../../data/test_suite/maskScorerTests/splicefooconfpart/B_NC2016_Unittest_Splice_ImgOnly_p-me_1_mask_scores_perimage.csv
	fi
	rm comp_maskreport_splicefooconfpart-perimage.txt
else
	echo comp_maskreport_splicefooconfpart-perimage.txt
	cat comp_maskreport_splicefooconfpart-perimage.txt
fi

flag_total=$(($flag_manipconf_0 + $flag_manipconfpi + $flag_manipconfjr\
 + $flag_manipconfmanmade_0 + $flag_manipconfmanmade_1\
 + $flag_manipconfmanmadepi + $flag_manipconfmanmadejr\
 + $flag_manipfooconfpi + $flag_manipfooconfjr\
 + $flag_manipconfpart + $flag_manipconfpartpi + $flag_manipconfpartjr\
 + $flag_manipconfclonepart + $flag_manipconfclonepartpi + $flag_manipconfclonepartjr\
 + $flag_manipfooconfpartpi + $flag_manipfooconfpartjr\
 + $flag_spliceconf_0 + $flag_spliceconfpi\
 + $flag_spliceconfcoll_0 + $flag_spliceconfcoll_1 + $flag_spliceconfcollpi\
 + $flag_splicefooconfpi\
 + $flag_spliceconfpart + $flag_spliceconfpartpi\
 + $flag_spliceconfcollpart + $flag_spliceconfcollpartpi\
 + $flag_splicefooconfpartpi))
if ([ $flag_total -eq 0 ]); then
  echo
  echo "CASE 2 SUCCESSFULLY PASSED"
  echo
	if [ $clean = "TRUE" ] ; then
		rm -rf ../../data/test_suite/maskScorerTests/manipconf
		rm -rf ../../data/test_suite/maskScorerTests/manipconfmanmade
		rm -rf ../../data/test_suite/maskScorerTests/manipfooconf
		rm -rf ../../data/test_suite/maskScorerTests/manipconfpart
		rm -rf ../../data/test_suite/maskScorerTests/manipconfclonepart
		rm -rf ../../data/test_suite/maskScorerTests/manipfooconfpart

		rm -rf ../../data/test_suite/maskScorerTests/spliceconf
		rm -rf ../../data/test_suite/maskScorerTests/spliceconfcoll
		rm -rf ../../data/test_suite/maskScorerTests/splicefooconf
		rm -rf ../../data/test_suite/maskScorerTests/spliceconfpart
		rm -rf ../../data/test_suite/maskScorerTests/spliceconfcollpart
		rm -rf ../../data/test_suite/maskScorerTests/splicefooconfpart
	fi
else
  echo
  echo "    !!!!! MASK SCORER TEST FAILED AT CASE 2 !!!!!    "
  echo
  exit
fi

