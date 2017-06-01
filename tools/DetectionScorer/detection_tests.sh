#!/bin/bash

export testsuite_directory=../../data/test_suite/detectionScorerTests1

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
    $test "$compcheckfile_outdir"
    check_status

    # Replace paths in logfile
    if [ -f "${compcheckfile_outdir}/log.txt" ]; then
	sed -e "s:${compcheckfile_outdir}/:${checkfile_outdir}/:g" -i "" "${compcheckfile_outdir}/log.txt"
    fi
    diff -x "*.pdf" -r "$checkfile_outdir" "$compcheckfile_outdir" #exclude pdf files
    check_status

    echo "*** OK ***"
}

# baseline test_c1_1
test_c1_1() {
    echo "  **  Testing NC2016 Manipulation with a baseline DCT  **  "
    python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/reference/" \
				       -x "NC2016-manipulation-index.csv" \
				       -r "NC2016-manipulation-ref.csv" \
                       --sysDir "$testsuite_directory/baseline/" \
				       -s "Base_NC2016_Manipulation_ImgOnly_p-dct_02.csv"
}

test_c1_2() {
    echo "  **  Testing NC2016 Splice with a baseline  **  "
    python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t splice \
                       --refDir "$testsuite_directory/reference/" \
				       -x "NC2016-splice-index.csv" \
				       -r "NC2016-splice-ref.csv" \
                       --sysDir "$testsuite_directory/baseline/" \
				       -s "Base_NC2016_Splice_ImgOnly_p-splice_01.csv"
}

test_c1_3() {
    echo "  **  Testing NC2017 Manipulation with a baseline COPYMOVE  **  "
    python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/reference/" \
				       -x "NC2017-manipulation-index.csv" \
				       -r "NC2017-manipulation-ref.csv" \
                       --sysDir "$testsuite_directory/baseline/" \
				       -s "Base_NC2017_Manipulation_ImgOnly_p-copymove_01.csv"
}
