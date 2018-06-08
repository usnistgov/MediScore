#!/bin/bash

export testsuite_directory=../../data/test_suite/maskScorerTests

check_status() {
    status=$?
    passfail=$1
    if [ $status -eq 0 ]; then
        if [ $passfail -eq 0 ]; then
            echo "*** OK ***"
        else
	    echo "*** FAILED ***"
	    exit 1
        fi
    else
        if [ $passfail -ne 0 ]; then
            echo "*** OK ***"
        else
	    echo "*** FAILED ***"
	    exit 1
        fi
    fi
}

run_test() {
    test=$1
    passfail=$2

    echo "** Running integration test '$test' **"
    $test
    check_status $passfail
}

diff_csv(){
    test_csv=$1
    ref_csv=$2
    test_pfx=$3

    comp_file_name=$test_pfx.diff.txt
    diff $test_csv $ref_csv > $comp_file_name

    filter_comp="cat $comp_file_name | grep -v CVS"
    if test "`eval $filter_comp`" = ""; then
        echo "Test $test_pfx passed."
        rm $comp_file_name
    else
        cat $comp_file_name
        echo "     !!!!! TEST $test_pfx FAILED !!!!!     "
    fi
}

base_manip_test() {
    echo 'basic manip test'
    #test csv
    run_test "python2 get_dims_from_probes.py\
 -t manipulation\
 --refDir $testsuite_directory\
 -r reference/manipulation-image/MFC18-manipulation-image-ref.csv\
 -x indexes/MFC18-manipulation-image-subindex.csv\
 -oR sample_manip_viz_db\
 --filter" 0
    diff_csv sample_manip_viz_db_refdims.csv $testsuite_directory/compcheckfiles/visualizer_check_files/ref_manip_base.csv manip_viz

    #test HTML
    run_test "python2 get_html_from_probefiles.py\
 -t manipulation\
 --refDir $testsuite_directory\
 -rx sample_manip_viz_db_refdims.csv\
 -oR sample_manip/sample_manip_viz_db\
 -rr --filter -ow" 0
}

base_splice_test() {
    echo 'basic splice test'
    #test csv
    run_test "python2 get_dims_from_probes.py\
 -t splice\
 --refDir $testsuite_directory\
 -r reference/splice/NC2017-splice-ref.csv\
 -x indexes/NC2017-splice-index.csv\
 -oR sample_splice_viz_db\
 --filter" 0
    diff_csv sample_splice_viz_db_refdims.csv $testsuite_directory/compcheckfiles/visualizer_check_files/ref_splice_base.csv splice_viz

    #test HTML
    run_test "python2 get_html_from_probefiles.py\
 -t splice\
 --refDir $testsuite_directory\
 -rx sample_splice_viz_db_refdims.csv\
 -oR sample_splice/sample_splice_viz_db\
 -rr --filter -ow" 0
}

