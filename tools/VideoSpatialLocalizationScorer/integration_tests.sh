#!/bin/bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export DIR

source test_init.sh

check_string() {
    string=$1
    log_file=$2
    mode=$3

    to_contain=""
    if [ "$mode" = "fail" ]; then
        to_contain="not "
    fi

    echo "Checking log "$log_file" for string '$s'. Log file should ${to_contain}contain this string."

    strflag=0
    if [ "`grep $string $log_file`" = "" ]; then
        if [ "$mode" = "pass" ]; then
            echo "*** FAILED ***"
            strflag=1
        fi
    else
        if [ "$mode" = "fail" ]; then
            echo "*** FAILED ***"
            strflag=1
        fi
    fi
    return $strflag
}

run_test() {
    test=$1
    status=$2

    flag=0

    echo "** Running integration test '$1' **"
    $test
    test_stat=$?
    if [[ $test_stat -ne $status ]]; then
        echo "*** FAILED ***"
    else
        echo "*** Status OK ***"
    fi
}

TESTDIR=../../data/test_suite/videoSpatialLocalizationScorerTests
procs=4

sfx_list_1=(
    _mask_score:\
    _mask_scores_perimage:-perimage\
    _journalResults:-journalResults
)

sfx_list_2=(
    _mask_score_0:_0\
    _mask_score_1:_1\
    _mask_scores_perimage:-perimage\
    _journalResults:-journalResults
)

test_wrapper(){
    test=$1
    ref_pfx=$2
    sys_pfx=$3
    comp_sfx=$4
    n_avg=$5
    
    run_test $test 0

    if [[ $n_avg == 1 ]]; then
        sfx_list=("${sfx_list_1[@]}")
    elif [[ $n_avg == 2 ]]; then
        sfx_list=("${sfx_list_2[@]}")
    else
        echo "Error: Test with $n_avg does not exist in this test suite."
        return 1
    fi

    flagsum=0
    for f_sfxs in ${sfx_list[@]}; do
        sys_sfx=`echo $f_sfxs | awk -F: '{print $1}'`
        ref_sfx=`echo $f_sfxs | awk -F: '{print $2}'`
        ref_file_name=${ref_pfx}${ref_sfx}.csv
        sys_file_name=${sys_pfx}${sys_sfx}.csv
        comp_file_name=comp_maskreport_${comp_sfx}${ref_sfx}.txt

        if [ ! -f $ref_file_name ]; then
            echo $ref_file_name does not exist. Test case may proceed.
            continue
        fi

        flag=`check_file $ref_file_name $sys_file_name $comp_file_name`
        if [[ $flag == 1 ]]; then
            echo "Error: Test $test failed."
        fi
    done
}

basic_test(){
    rm -rf test1
    command=(python2 VideoSpatialLocalizationScorer.py -t manipulation\
                                          --refDir $TESTDIR\
                                          --sysDir $TESTDIR/p-vsltest_1\
                                          -r reference/manipulation-video/MFC18_Dev2-manipulation-video-ref.csv\
                                          -s p-vsltest_1.csv\
                                          -x indexes/MFC18_Dev2-manipulation-video-index.csv\
                                          -oR test1/test1\
                                          --outMeta\
                                          --truncate)
    echo "${command[@]}"
    "${command[@]}"
}

basic_test_wrapper(){
    python2 $TESTDIR/gen_masks_for_ds.py -ds $TESTDIR
    python2 $TESTDIR/gen_spatial_mask.py -s $TESTDIR/p-vsltest_1/p-vsltest_1.csv -x $TESTDIR/indexes/MFC18_Dev2-manipulation-video-index.csv
    run_test basic_test 0
    rm $TESTDIR/p-vsltest_1/mask/*
    rm $TESTDIR/reference/manipulation-video/mask/*
#    ref_pfx=$TESTDIR/compcheckfiles/ref_maskreport_manip
#    sys_pfx=$TESTDIR/maniptest/B_NC2016_Unittest_Manipulation_ImgOnly_c-me2_1
#    comp_sfx=manip
#    test_wrapper maniptest "$ref_pfx" "$sys_pfx" "$comp_sfx" 1
}

basic_collar_test(){
    rm -rf test1
    command=(python2 VideoSpatialLocalizationScorer.py -t manipulation\
                                          --refDir $TESTDIR\
                                          --sysDir $TESTDIR/p-vsltest_1\
                                          -r reference/manipulation-video/MFC18_Dev2-manipulation-video-ref.csv\
                                          -s p-vsltest_1.csv\
                                          -x indexes/MFC18_Dev2-manipulation-video-index.csv\
                                          -oR test1c/test1c\
                                          --outMeta\
                                          --collars 3\
                                          --video_opt_out\
                                          --truncate)
    echo "${command[@]}"
    "${command[@]}"
}

basic_collar_test_wrapper(){
    python2 $TESTDIR/gen_masks_for_ds.py -ds $TESTDIR
    python2 $TESTDIR/gen_spatial_mask.py -s $TESTDIR/p-vsltest_1/p-vsltest_1.csv -x $TESTDIR/indexes/MFC18_Dev2-manipulation-video-index.csv
    run_test basic_collar_test 0
    rm $TESTDIR/p-vsltest_1/mask/*
    rm $TESTDIR/reference/manipulation-video/mask/*
#    ref_pfx=$TESTDIR/compcheckfiles/ref_maskreport_manip
#    sys_pfx=$TESTDIR/maniptest/B_NC2016_Unittest_Manipulation_ImgOnly_c-me2_1
#    comp_sfx=manip
#    test_wrapper maniptest "$ref_pfx" "$sys_pfx" "$comp_sfx" 1
}

