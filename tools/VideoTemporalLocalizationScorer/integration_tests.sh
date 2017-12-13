#!/bin/bash

export testsuite_directory=../../data/test_suite/videoTemporalLocalizationScorerTests

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

    echo "** Running integration test '$test' **"
    $test "$compcheckfile_outdir"
    check_status

    # Replace paths in logfile
    log_fn="${compcheckfile_outdir}/log.txt"
    if [ -f "$log_fn" ]; then
	sed -e "s:${compcheckfile_outdir}/:${checkfile_outdir}/:g" "$log_fn" >"${log_fn}.new"
	mv "${log_fn}.new" "$log_fn"
    fi
    diff -r "$checkfile_outdir" "$compcheckfile_outdir"
    check_status
    
    echo "*** OK ***"
}

# Graph Building test 1_0
test_1_0() {
    python ./VideoTemporalLocalizationScoring.py -o "$1" \
                        -r "$testsuite_directory/test_case_1_videotemploc-ref.csv" \
                        -i "$testsuite_directory/test_case_1_videotemploc-index.csv" \
                        -j "$testsuite_directory/test_case_1_videotemploc-ref-journalmask.csv" \
                        -p "$testsuite_directory/test_case_1_videotemploc-ref-probejournaljoin.csv" \
                        -s "$testsuite_directory/test_case_1_system_output.csv" \
                        -c 5 \
                        --query "*" "Operation == 'PasteFrames'" \
                        -l
}

