#!/bin/bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export DIR
clean=yes
export clean

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
        return 1
    else
        echo "*** Status OK ***"
        return 0
    fi
}

TESTDIR=$(realpath ../../data/test_suite/videoSpatialLocalizationScorerTests)
procs=4

sfx_list_1=(
    _mask_score:_mask_score\
    _pervideo:_pervideo\
    _journalResults:_journalResults
)

sfx_list_2=(
    _mask_score_0:_0\
    _mask_score_1:_1\
    _pervideo:_pervideo\
    _journalResults:_journalResults
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

gen_masks(){
    chmod 777 $TESTDIR/reference/manipulation-video
    mkdir -p $TESTDIR/reference/manipulation-video/mask
    python2 $TESTDIR/gen_masks_for_ds.py -ds $TESTDIR
    chmod 777 $TESTDIR/p-vsltest_1
    mkdir -p $TESTDIR/p-vsltest_1/mask
    python2 $TESTDIR/gen_spatial_mask.py -s $TESTDIR/p-vsltest_1/p-vsltest_1.csv -x $TESTDIR/indexes/MFC18_Dev2-manipulation-video-index.csv
}

rm_masks(){
    rm $TESTDIR/p-vsltest_1/mask/*
    rm $TESTDIR/p-vsltest_dims/mask/*
    rm $TESTDIR/reference/manipulation-video/mask/*
}

basic_test(){
    command=(python2 $DIR/VideoSpatialLocalizationScorer.py -t manipulation\
                                          --refDir $TESTDIR\
                                          --sysDir $TESTDIR/p-vsltest_1\
                                          -r reference/manipulation-video/MFC18_Dev2-manipulation-video-ref.csv\
                                          -s p-vsltest_1.csv\
                                          -x indexes/MFC18_Dev2-manipulation-video-index.csv\
                                          -oR test1/test1\
                                          --precision 12\
                                          --truncate)
#                                          --outMeta\
    echo "${command[@]}"
    "${command[@]}"
}

basic_test_wrapper(){
    ref_pfx=$TESTDIR/compcheckfiles/test1
    sys_pfx=$DIR/test1/test1
    comp_sfx=full_test
    test_wrapper basic_test "$ref_pfx" "$sys_pfx" "$comp_sfx" 1
    exitstatus=$?
    if [[ "$clean" == "yes" ]]; then
        rm ${comp_sfx}*
        rm -rf $DIR/test1
    fi
    return $exitstatus
}

basic_test_noed(){
    command=(python2 $DIR/VideoSpatialLocalizationScorer.py -t manipulation\
                                          --refDir $TESTDIR\
                                          --sysDir $TESTDIR/p-vsltest_1\
                                          -r reference/manipulation-video/MFC18_Dev2-manipulation-video-ref.csv\
                                          -s p-vsltest_1.csv\
                                          -x indexes/MFC18_Dev2-manipulation-video-index.csv\
                                          -oR $DIR/test1noed/test1noed\
                                          --eks 0\
                                          --dks 0\
                                          --precision 12\
                                          --truncate)
#                                          --outMeta\
    echo "${command[@]}"
    "${command[@]}"
}

basic_test_noed_wrapper(){
    ref_pfx=$TESTDIR/compcheckfiles/test1noed
    sys_pfx=$DIR/test1noed/test1noed
    comp_sfx=no_erode_dilate
    test_wrapper basic_test_noed "$ref_pfx" "$sys_pfx" "$comp_sfx" 1
    exitstatus=$?
    if [[ "$clean" == "yes" ]]; then
        rm ${comp_sfx}*
        rm -rf $DIR/test1noed
    fi
    return $exitstatus
}

basic_collar_test(){
    command=(python2 $DIR/VideoSpatialLocalizationScorer.py -t manipulation\
                                          --refDir $TESTDIR\
                                          --sysDir $TESTDIR/p-vsltest_1\
                                          -r reference/manipulation-video/MFC18_Dev2-manipulation-video-ref.csv\
                                          -s p-vsltest_1.csv\
                                          -x indexes/MFC18_Dev2-manipulation-video-index.csv\
                                          -oR test1c/test1c\
                                          --collars 3\
                                          --precision 12\
                                          --truncate)
#                                          --outMeta\
    echo "${command[@]}"
    "${command[@]}"
}

basic_collar_test_wrapper(){
    ref_pfx=$TESTDIR/compcheckfiles/test1c
    sys_pfx=$DIR/test1c/test1c
    comp_sfx=test_collar
    test_wrapper basic_collar_test "$ref_pfx" "$sys_pfx" "$comp_sfx" 1
    exitstatus=$?
    if [[ "$clean" == "yes" ]]; then
        rm ${comp_sfx}*
        rm -rf $DIR/test1c
    fi
    return $exitstatus
}

basic_video_oo_test(){
    command=(python2 $DIR/VideoSpatialLocalizationScorer.py -t manipulation\
                                          --refDir $TESTDIR\
                                          --sysDir $TESTDIR/p-vsltest_1\
                                          -r reference/manipulation-video/MFC18_Dev2-manipulation-video-ref.csv\
                                          -s p-vsltest_1.csv\
                                          -x indexes/MFC18_Dev2-manipulation-video-index.csv\
                                          -oR test1oo/test1oo\
                                          --eks 0\
                                          --dks 0\
                                          --video_opt_out\
                                          --precision 12\
                                          --truncate)
#                                          --outMeta\
    echo "${command[@]}"
    "${command[@]}"
}

basic_video_oo_test_wrapper(){
    ref_pfx=$TESTDIR/compcheckfiles/test1oo
    sys_pfx=$DIR/test1oo/test1oo
    comp_sfx=test_optout
    test_wrapper basic_video_oo_test "$ref_pfx" "$sys_pfx" "$comp_sfx" 1
    exitstatus=$?
    if [[ "$clean" == "yes" ]]; then
        rm ${comp_sfx}*
        rm -rf $DIR/test1oo
    fi
    return $exitstatus
}

selective_test_1(){
    command=(python2 $DIR/VideoSpatialLocalizationScorer.py -t manipulation\
                                          --refDir $TESTDIR\
                                          --sysDir $TESTDIR/p-vsltest_1\
                                          -r reference/manipulation-video/MFC18_Dev2-manipulation-video-ref.csv\
                                          -s p-vsltest_1.csv\
                                          -x indexes/MFC18_Dev2-manipulation-video-index.csv\
                                          -oR $DIR/sel1/sel1\
                                          --eks 0\
                                          --dks 0\
                                          --ntdks 0\
                                          -qm "Operation == 'PasteSampled'"\
                                          --precision 12\
                                          --truncate)
#                                          --outMeta\
    echo "${command[@]}"
    "${command[@]}"
}

selective_test_10(){
    command=(python2 $DIR/VideoSpatialLocalizationScorer.py -t manipulation\
                                          --refDir $TESTDIR\
                                          --sysDir $TESTDIR/p-vsltest_1\
                                          -r reference/manipulation-video/MFC18_Dev2-manipulation-video-ref.csv\
                                          -s p-vsltest_1.csv\
                                          -x indexes/MFC18_Dev2-manipulation-video-index.csv\
                                          -oR $DIR/sel10/sel10\
                                          --eks 0\
                                          --dks 0\
                                          --ntdks 0\
                                          -qm "Operation == 'PasteImageSpliceToFrames'"\
                                          --precision 12\
                                          --truncate)
#                                          --outMeta\
    echo "${command[@]}"
    "${command[@]}"
}

selective_tests_wrapper(){
    ref_pfx=$TESTDIR/compcheckfiles/sel1
    sys_pfx=$DIR/sel1/sel1
    comp_sfx=selective_1
#    run_test selective_test_1 0
    test_wrapper selective_test_1 "$ref_pfx" "$sys_pfx" "$comp_sfx" 1
    if [[ "$clean" == "yes" ]]; then
        rm ${comp_sfx}*
        rm -rf $DIR/sel1
    fi

    ref_pfx=$TESTDIR/compcheckfiles/sel10
    sys_pfx=$DIR/sel10/sel10
    comp_sfx=selective_10
#    run_test selective_test_10 0
    test_wrapper selective_test_10 "$ref_pfx" "$sys_pfx" "$comp_sfx" 1
    exitstatus=$?
    if [[ "$clean" == "yes" ]]; then
        rm ${comp_sfx}*
        rm -rf $DIR/sel10
    fi
    return $exitstatus
}

err_test(){
    #gen err mask for different system, then run basic test with those masks
    chmod 777 $TESTDIR/p-vsltest_dims
    mkdir -p $TISTDIR/p-vsltest_dims/mask
    python2 $TESTDIR/gen_spatial_mask.py -s $TESTDIR/p-vsltest_dims/p-vsltest_dims.csv -x $TESTDIR/indexes/MFC18_Dev2-manipulation-video-index.csv --shift_frames 8
    command=(python2 $DIR/VideoSpatialLocalizationScorer.py -t manipulation\
                                          --refDir $TESTDIR\
                                          --sysDir $TESTDIR/p-vsltest_dims\
                                          -r reference/manipulation-video/MFC18_Dev2-manipulation-video-ref.csv\
                                          -s p-vsltest_dims.csv\
                                          -x indexes/MFC18_Dev2-manipulation-video-index.csv\
                                          -oR test_err/test_err\
                                          --precision 12\
                                          --truncate)
#                                          --outMeta\
    echo "${command[@]}"
    "${command[@]}"
}

err_test_wrapper(){
    ref_pfx=$TESTDIR/compcheckfiles/test1
    sys_pfx=$DIR/test_err/test_err
    run_test err_test 1
    exitstatus=$?
    if [[ "$clean" == "yes" ]]; then
        rm -rf $DIR/test_err
    fi
    rm $TESTDIR/p-vsltest_dims/mask/*
    return $exitstatus
}

