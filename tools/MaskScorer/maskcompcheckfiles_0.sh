#!/bin/bash
procs=4

echo "PROCEEDING TO THOROUGHLY CHECK ALL CASES"
echo
echo "CASE 0: VALIDATING FULL SCORING"
echo

source test_init.sh
TESTDIR=../../data/test_suite/maskScorerTests

#produce the output files
$mypython MaskScorer.py -t manipulation --refDir $TESTDIR -r reference/manipulation/NC2016-manipulation-ref.csv -x indexes/NC2016-manipulation-index.csv -s $TESTDIR/B_NC2016_Unittest_Manipulation_ImgOnly_c-me2_1/B_NC2016_Unittest_Manipulation_ImgOnly_c-me2_1.csv -oR $TESTDIR/maniptest/B_NC2016_Unittest_Manipulation_ImgOnly_c-me2_1 -p $procs --speedup --precision 12
$mypython MaskScorer.py -t splice --refDir $TESTDIR -r reference/splice/NC2017-splice-ref.csv -x indexes/NC2017-splice-index.csv -s $TESTDIR/B_NC2017_Unittest_Splice_ImgOnly_p-me_1/B_NC2017_Unittest_Splice_ImgOnly_p-me_1.csv -oR $TESTDIR/splicetest/B_NC2017_Unittest_Splice_ImgOnly_p-me_1 -html -p $procs --speedup --precision 12
$mypython MaskScorer.py -t manipulation --refDir $TESTDIR -r reference/manipulation/NC2017-manipulation-ref.csv -x indexes/NC2017-manipulation-index.csv -s $TESTDIR/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1.csv -oR $TESTDIR/threstest/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1 -html --sbin 128 -p $procs --speedup --precision 12
$mypython MaskScorer.py -t splice --refDir $TESTDIR -r reference/splice/NC2017-splice-ref.csv -x indexes/NC2017-splice-index.csv --sysDir $TESTDIR -s C_NC2017_Unittest_Splice_ImgOnly_p-me_1/C_NC2017_Unittest_Splice_ImgOnly_p-me_1.csv -oR $TESTDIR/splicebin/C_NC2017_Unittest_Splice_ImgOnly_p-me_1 -xF -p 6 -html --outMeta --outAllmeta --sbin 200 --speedup --precision 12 --jpeg2000 #shouldn't make a difference whether this option is here or not

#manipulation and splice optOut case
$mypython MaskScorer.py -t manipulation --refDir $TESTDIR -r reference/manipulation/NC2017-manipulation-ref.csv -x indexes/NC2017-manipulation-index.csv -s $TESTDIR/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1.csv -oR $TESTDIR/manip_optOut/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1 -html -p $procs --speedup --optOut --precision 12
$mypython MaskScorer.py -t splice --refDir $TESTDIR -r reference/splice/NC2017-splice-ref.csv -x indexes/NC2017-splice-index.csv -s $TESTDIR/B_NC2017_Unittest_Splice_ImgOnly_p-me_1/B_NC2017_Unittest_Splice_ImgOnly_p-me_1.csv -oR $TESTDIR/splice_optOut/B_NC2017_Unittest_Splice_ImgOnly_p-me_1 -html -p $procs --speedup --optOut --precision 12

#add examples for FailedValidation
$mypython MaskScorer.py -t manipulation --refDir $TESTDIR -r reference/manipulation/NC2017-manipulation-ref.csv -x indexes/NC2017-manipulation-index.csv -s $TESTDIR/p-failvalid_1_manip/p-failvalid_1_manip.csv -oR $TESTDIR/manipfailvalid/p-failvalid_1_manip -html -p $procs --speedup --precision 12
$mypython MaskScorer.py -t splice --refDir $TESTDIR -r reference/splice/NC2017-splice-ref.csv -x indexes/NC2017-splice-index.csv -s $TESTDIR/p-failvalid_2_splice/p-failvalid_2_splice.csv -oR $TESTDIR/splicefailvalid/p-failvalid_2_splice -html -p $procs --speedup --precision 12


flagsum=0

#run a for loop to check everything
for test_fields in\
    $TESTDIR/compcheckfiles/ref_maskreport_manip:$TESTDIR/maniptest/B_NC2016_Unittest_Manipulation_ImgOnly_c-me2_1:manip\
    $TESTDIR/compcheckfiles/ref_maskreport_splice:$TESTDIR/splicetest/B_NC2017_Unittest_Splice_ImgOnly_p-me_1:splice\
    $TESTDIR/compcheckfiles/ref_maskreport_thres:$TESTDIR/threstest/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1:thres\
    $TESTDIR/compcheckfiles/ref_maskreport_splicebin:$TESTDIR/splicebin/C_NC2017_Unittest_Splice_ImgOnly_p-me_1:splicebin\
    $TESTDIR/compcheckfiles/ref_maskreport_manipfailvalid:$TESTDIR/manipfailvalid/p-failvalid_1_manip:manipfailvalid\
    $TESTDIR/compcheckfiles/ref_maskreport_splicefailvalid:$TESTDIR/splicefailvalid/p-failvalid_2_splice:splicefailvalid\
    $TESTDIR/compcheckfiles/ref_maskreport_manip_optOut:$TESTDIR/manip_optOut/B_NC2017_Unittest_Manipulation_ImgOnly_c-me2_1:manip_optOut\
    $TESTDIR/compcheckfiles/ref_maskreport_splice_optOut:$TESTDIR/splice_optOut/B_NC2017_Unittest_Splice_ImgOnly_p-me_1:splice_optOut; do
    ref_pfx=`echo $test_fields | awk -F: '{print $1}'`
    sys_pfx=`echo $test_fields | awk -F: '{print $2}'`
    comp_sfx=`echo $test_fields | awk -F: '{print $3}'`
    for f_sfxs in _mask_score:\
                 _mask_scores_perimage:-perimage\
                 _journalResults:-journalResults; do
        sys_sfx=`echo $f_sfxs | awk -F: '{print $1}'`
        ref_sfx=`echo $f_sfxs | awk -F: '{print $2}'`
    
        flag=`check_file ${ref_pfx}${ref_sfx}.csv ${sys_pfx}${sys_sfx}.csv comp_maskreport_${comp_sfx}${ref_sfx}.txt`
        flagsum=$((flagsum+flag))
    done
done

#faulty test case.
errflag=1
$mypython MaskScorer.py -t manipulation --refDir $TESTDIR/ -r reference/manipulation/NC2017-manipulation-ref.csv -x indexes/NC2017-manipulation-index.csv -s $TESTDIR/Error_NC2017_Unittest_Manipulation_ImgOnly_c-me_1/Error_NC2017_Unittest_Manipulation_ImgOnly_c-me_1.csv -oR $TESTDIR/errtest/Error_NC2017_Unittest_Manipulation_ImgOnly_c-me_1 --speedup $optOutClause -v 1 --debug_off > errlog.txt
if `grep -q ERROR.*unreadable errlog.txt` && `grep -q Ending errlog.txt` ; then
    errflag=0
fi
flagsum=$((flagsum+errflag))

if ([ $flagsum -eq 0 ]); then
    echo
    echo "CASE 0 SUCCESSFULLY PASSED"
    echo
	if [ $clean = "TRUE" ] ; then
		rm -rf $TESTDIR/maniptest
		rm -rf $TESTDIR/splicetest
		rm -rf $TESTDIR/threstest
		rm -rf $TESTDIR/splicebin
		rm -rf $TESTDIR/manip_optOut
		rm -rf $TESTDIR/splice_optOut
                rm -rf $TESTDIR/errtest
	fi
else
    echo
    echo "    !!!!! MASK SCORER TEST FAILED AT CASE 0 !!!!!    "
    echo
    exit 1
fi

