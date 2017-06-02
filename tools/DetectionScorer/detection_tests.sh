#!/bin/bash

export testsuite_directory=../../data/test_suite/detectionScorerTests

check_status() {
    status=$?
    if [ $status -ne 0 ]; then
	echo "*** FAILED ***"
	exit $status
    fi
}

run_test() {
    test=$1
    checkfile_outdir=$2
    checkfile_outdir_basename=`basename $checkfile_outdir`
    compcheck_outdir=${3-compcheckfiles}
    compcheckfile_outdir="$compcheck_outdir/$checkfile_outdir_basename"

    echo "** Running detection test case: '$test' **"
    $test "$compcheckfile_outdir" 1> $compcheckfile_outdir.com.log 2>&1
    check_status $compcheckfile_outdir.com.log

    # Replace paths in logfile
    if [ -f "${compcheckfile_outdir}/log.txt" ]; then
	     sed -e "s:${compcheckfile_outdir}/:${checkfile_outdir}/:g" -i "" "${compcheckfile_outdir}/log.txt"
    fi
    #exclude pdf files
    diff -x "*.pdf" -r "$checkfile_outdir" "$compcheckfile_outdir" 1> $compcheckfile_outdir.diff.log 2>&1
    check_status $compcheckfile_outdir.diff.log

    echo "*** OK ***"
}

# baseline test_c1_1
test_c1_1() {
    echo "  * Testing NC2016 Manipulation with a baseline DCT  * "
    python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/reference/" \
				       -x "NC2016-manipulation-index.csv" \
				       -r "NC2016-manipulation-ref.csv" \
                       --sysDir "$testsuite_directory/baseline/" \
				       -s "Base_NC2016_Manipulation_ImgOnly_p-dct_02.csv"
}

test_c1_2() {
    echo "  * Testing NC2016 Splice with a baseline  * "
    python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t splice \
                       --refDir "$testsuite_directory/reference/" \
				       -x "NC2016-splice-index.csv" \
				       -r "NC2016-splice-ref.csv" \
                       --sysDir "$testsuite_directory/baseline/" \
				       -s "Base_NC2016_Splice_ImgOnly_p-splice_01.csv"
}

test_c1_3() {
    echo "  * Testing NC2017 Manipulation with a baseline COPYMOVE *  "
    python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/reference/" \
				       -x "NC2017-manipulation-index.csv" \
				       -r "NC2017-manipulation-ref.csv" \
                       --sysDir "$testsuite_directory/baseline/" \
				       -s "Base_NC2017_Manipulation_ImgOnly_p-copymove_01.csv"
}


test_c2_1() {
    echo "  * Testing system output test case for manipulation *  "
    python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2016-manipulation-index.csv" \
        				       -r "NC2016-manipulation-ref.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2016_Manipulation_ImgOnly_p-me_1/D_NC2016_Manipulation_ImgOnly_p-me_1.csv"
}

test_c2_2() {
    echo "  * Testing system output test case for splice *  "
    python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t splice \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2016-splice-index.csv" \
        				       -r "NC2016-splice-ref.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2016_Splice_ImgOnly_p-me_1/D_NC2016_Splice_ImgOnly_p-me_1.csv"
}

test_c2_3() {
    echo "  * Testing with the same scores across all image files *  "
    python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2016-manipulation-index.csv" \
        				       -r "NC2016-manipulation-ref.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2016_Manipulation_ImgOnly_p-me_2/D_NC2016_Manipulation_ImgOnly_p-me_2.csv"
}

test_c2_4() {
    echo "  * Testing with no non-target value *  "
    python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2017-manipulation-index.csv" \
        				       -r "NC2017-manipulation-ref.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2017_Manipulation_ImgOnly_c-me_2/D_NC2017_Manipulation_ImgOnly_c-me_2.csv"
}

test_c2_5() {
    echo "  * Testing with one target and one non-target trial*  "
    python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2017-manipulation-index.csv" \
        				       -r "NC2017-manipulation-ref2.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2017_Manipulation_ImgOnly_c-me_2/D_NC2017_Manipulation_ImgOnly_c-me_2.csv"
}

test_c2_6() {
    echo "  * Testing with the manipulation OptOut case *  "
    python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2016-manipulation-index.csv" \
        				       -r "NC2016-manipulation-ref.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2016_Manipulation_ImgOnly_p-me_3/D_NC2016_Manipulation_ImgOnly_p-me_3.csv"
}

test_c3_1() {
    echo "  * Testing with the manipulation case with full index *  "
    python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2016-manipulation-index.csv" \
        				       -r "NC2016-manipulation-ref.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2016_Manipulation_ImgOnly_p-me_1/D_NC2016_Manipulation_ImgOnly_p-me_1.csv"
}

test_c3_2() {
    echo "  * Testing with the manipulation case with sub index (1 less) *  "
    python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2016-manipulation-index_sub.csv" \
        				       -r "NC2016-manipulation-ref.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2016_Manipulation_ImgOnly_p-me_1/D_NC2016_Manipulation_ImgOnly_p-me_1.csv"
}

test_c3_3() {
    echo "  * Testing with the splice case with full index *  "
    python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t splice \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2016-splice-index.csv" \
        				       -r "NC2016-splice-ref.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2016_Splice_ImgOnly_p-me_1/D_NC2016_Splice_ImgOnly_p-me_1.csv"
}

test_c3_4() {
    echo "  * Testing with the splice case with sub index (2 less) *  "
    python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t splice \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2016-splice-index_sub.csv" \
        				       -r "NC2016-splice-ref.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2016_Splice_ImgOnly_p-me_1/D_NC2016_Splice_ImgOnly_p-me_1.csv"
}

test_c4_1() {
    echo "  * Testing all data without queries - Manipulation *  "
    python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2017-manipulation-index-jt.csv" \
        				       -r "NC2017-manipulation-ref-jt.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2017_Manipulation_ImgOnly_c-me_3/D_NC2017_Manipulation_ImgOnly_c-me_3.csv"
}

test_c4_2() {
    echo "  * Testing a query existed in all ProbeFileIDs - Manipulation *  "
    python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2017-manipulation-index-jt.csv" \
        				       -r "NC2017-manipulation-ref-jt.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2017_Manipulation_ImgOnly_c-me_3/D_NC2017_Manipulation_ImgOnly_c-me_3.csv" \
               -qm "Operation==['PasteSplice']"
}

test_c4_3() {
    echo "  * Testing a query existed in only part of ProbeFileIDs - Manipulation *  "
    python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2017-manipulation-index-jt.csv" \
        				       -r "NC2017-manipulation-ref-jt.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2017_Manipulation_ImgOnly_c-me_3/D_NC2017_Manipulation_ImgOnly_c-me_3.csv" \
               -qm "Operation==['FillContentAwareFill']"
}
