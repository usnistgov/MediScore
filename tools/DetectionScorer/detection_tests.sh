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
    $test "$compcheckfile_outdir" 1> $compcheckfile_outdir.comp.log 2>&1
    check_status $compcheckfile_outdir.comp.log

    # Replace paths in logfile
    if [ -f "${compcheckfile_outdir}/log.txt" ]; then
	    sed -e "s:${compcheckfile_outdir}/:${checkfile_outdir}/:g" -i "" "${compcheckfile_outdir}/log.txt"
    fi
    #exclude pdf and .DS_Store files
    #diff -x "*.pdf" -r "$checkfile_outdir" "$compcheckfile_outdir" 1> $compcheckfile_outdir.diff.log 2>&1
    #check_status $compcheckfile_outdir.diff.log

    diff --exclude="*.pdf" --exclude="*DS_Store" -r "$checkfile_outdir" "$compcheckfile_outdir" 1> $compcheckfile_outdir.diff.log 2>&1
    check_status $compcheckfile_outdir.diff.log

    echo "*** OK ***"
}

echo_and_run() { echo "$@" ; "$@" ; }

# baseline test_c1_1
test_c1_1() {
    echo "  * Testing NC2016 Manipulation with a baseline DCT  * "
    echo_and_run python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/reference/" \
				       -x "NC2016-manipulation-index.csv" \
				       -r "NC2016-manipulation-ref.csv" \
                       --sysDir "$testsuite_directory/baseline/" \
				       -s "Base_NC2016_Manipulation_ImgOnly_p-dct_02.csv"
}

test_c1_2() {
    echo "  * Testing NC2016 Splice with a baseline  * "
    echo_and_run python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t splice \
                       --refDir "$testsuite_directory/reference/" \
				       -x "NC2016-splice-index.csv" \
				       -r "NC2016-splice-ref.csv" \
                       --sysDir "$testsuite_directory/baseline/" \
				       -s "Base_NC2016_Splice_ImgOnly_p-splice_01.csv"
}

test_c1_3() {
    echo "  * Testing NC2017 Manipulation with a baseline COPYMOVE *  "
    echo_and_run python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/reference/" \
				       -x "NC2017-manipulation-index.csv" \
				       -r "NC2017-manipulation-ref.csv" \
                       --sysDir "$testsuite_directory/baseline/" \
				       -s "Base_NC2017_Manipulation_ImgOnly_p-copymove_01.csv"
}


test_c2_1() {
    echo "  * Testing system output test case for manipulation *  "
    echo_and_run python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2016-manipulation-index.csv" \
        				       -r "NC2016-manipulation-ref.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2016_Manipulation_ImgOnly_p-me_1/D_NC2016_Manipulation_ImgOnly_p-me_1.csv"
}

test_c2_2() {
    echo "  * Testing system output test case for splice *  "
    echo_and_run python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t splice \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2016-splice-index.csv" \
        				       -r "NC2016-splice-ref.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2016_Splice_ImgOnly_p-me_1/D_NC2016_Splice_ImgOnly_p-me_1.csv"
}

test_c2_3() {
    echo "  * Testing with the same scores across all image files *  "
    echo_and_run python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2016-manipulation-index.csv" \
        				       -r "NC2016-manipulation-ref.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2016_Manipulation_ImgOnly_p-me_2/D_NC2016_Manipulation_ImgOnly_p-me_2.csv"
}

test_c2_4() {
    echo "  * Testing with no non-target value *  "
    echo_and_run python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2017-manipulation-index.csv" \
        				       -r "NC2017-manipulation-ref.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2017_Manipulation_ImgOnly_c-me_2/D_NC2017_Manipulation_ImgOnly_c-me_2.csv"
}

test_c2_5() {
    echo "  * Testing with one target and one non-target trial*  "
    echo_and_run python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2017-manipulation-index.csv" \
        				       -r "NC2017-manipulation-ref2.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2017_Manipulation_ImgOnly_c-me_2/D_NC2017_Manipulation_ImgOnly_c-me_2.csv"
}

test_c2_6() {
    echo "  * Testing with the manipulation OptOut case -- IsOptOut*  "
    echo_and_run python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2016-manipulation-index.csv" \
        				       -r "NC2016-manipulation-ref.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2016_Manipulation_ImgOnly_p-me_3/D_NC2016_Manipulation_ImgOnly_p-me_3.csv" --optOut
}


test_c2_7() {
    echo "  * Testing a query with the manipulation OptOut case -- IsOptOut *  "
    echo_and_run python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2016-manipulation-index.csv" \
        				       -r "NC2016-manipulation-ref.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2016_Manipulation_ImgOnly_p-me_3/D_NC2016_Manipulation_ImgOnly_p-me_3.csv" \
               -qm "Operation==['PasteSplice']" --optOut
}

test_c2_8() {
    echo "  * Testing with the manipulation OptOut case -- ProbeStatus*  "
    echo_and_run python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2016-manipulation-index.csv" \
        				       -r "NC2016-manipulation-ref.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2016_Manipulation_ImgOnly_p-me_4/D_NC2016_Manipulation_ImgOnly_p-me_4.csv" --optOut
}

test_c2_9() {
    echo "  * Testing a query with the manipulation OptOut case -- ProbeStatus*  "
    echo_and_run python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2016-manipulation-index.csv" \
        				       -r "NC2016-manipulation-ref.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2016_Manipulation_ImgOnly_p-me_4/D_NC2016_Manipulation_ImgOnly_p-me_4.csv" \
               -qm "Operation==['PasteSplice']" --optOut
}

test_c2_10() {
    echo "  * Testing system output test case for EventRepurpose *  "
    echo_and_run python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "MFC2018-eventrepurpose-index.csv" \
        				       -r "MFC2018-eventrepurpose-ref.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_MFC2018_EventRepurpose_ImgOnly_p-me_1/D_MFC2018_EventRepurpose_ImgOnly_p-me_1.csv"
}

test_c3_1() {
    echo "  * Testing with the manipulation case with full index *  "
    echo_and_run python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2016-manipulation-index.csv" \
        				       -r "NC2016-manipulation-ref.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2016_Manipulation_ImgOnly_p-me_1/D_NC2016_Manipulation_ImgOnly_p-me_1.csv"
}

test_c3_2() {
    echo "  * Testing with the manipulation case with sub index (1 less) *  "
    echo_and_run python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2016-manipulation-index_sub.csv" \
        				       -r "NC2016-manipulation-ref.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2016_Manipulation_ImgOnly_p-me_1/D_NC2016_Manipulation_ImgOnly_p-me_1.csv"
}

test_c3_3() {
    echo "  * Testing with the splice case with full index *  "
    echo_and_run python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t splice \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2016-splice-index.csv" \
        				       -r "NC2016-splice-ref.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2016_Splice_ImgOnly_p-me_1/D_NC2016_Splice_ImgOnly_p-me_1.csv"
}

test_c3_4() {
    echo "  * Testing with the splice case with sub index (2 less) *  "
    echo_and_run python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t splice \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2016-splice-index_sub.csv" \
        				       -r "NC2016-splice-ref.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2016_Splice_ImgOnly_p-me_1/D_NC2016_Splice_ImgOnly_p-me_1.csv"
}

test_c4_1() {
    echo "  * Testing all data without queries - Manipulation *  "
    echo_and_run python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2017-manipulation-index-jt.csv" \
        				       -r "NC2017-manipulation-ref-jt.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2017_Manipulation_ImgOnly_c-me_3/D_NC2017_Manipulation_ImgOnly_c-me_3.csv"
}

test_c4_2() {
    echo "  * Testing a query existed in all ProbeFileIDs - Manipulation *  "
    echo_and_run python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
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
    echo_and_run python2 DetectionScorer.py -o "$compcheckfile_outdir/$checkfile_outdir_basename" \
                       -t manipulation \
                       --refDir "$testsuite_directory/sample/reference" \
                       -x "NC2017-manipulation-index-jt.csv" \
        				       -r "NC2017-manipulation-ref-jt.csv" \
                       --sysDir "$testsuite_directory/sample/" \
				       -s "D_NC2017_Manipulation_ImgOnly_c-me_3/D_NC2017_Manipulation_ImgOnly_c-me_3.csv" \
               -qm "Operation==['FillContentAwareFill']"
}



test_c5_1() {
    echo "  * Testing all the examples from the DetectionScorer ReadMe document *  "

    echo "  * Full scoring: rendering the ROC curve and the report table *  "
    echo_and_run python2 DetectionScorer.py -o $compcheckfile_outdir/$checkfile_outdir_basename-1 \
                        -t manipulation \
                        --refDir $testsuite_directory/reference \
                        -r NC2016-manipulation-ref.csv \
                        -x NC2016-manipulation-index.csv \
                        --sysDir $testsuite_directory/baseline \
                        -s Base_NC2016_Manipulation_ImgOnly_p-dct_02.csv \
                        --ci

    echo "  * Full scoring: rendering DET curve *  "
    echo_and_run python2 DetectionScorer.py -o $compcheckfile_outdir/$checkfile_outdir_basename-2 \
                          -t manipulation \
                          --refDir $testsuite_directory/reference \
                          -r NC2016-manipulation-ref.csv -x NC2016-manipulation-index.csv \
                          --sysDir $testsuite_directory/baseline \
                          -s Base_NC2016_Manipulation_ImgOnly_p-dct_02.csv  \
                          --plotType det

    echo "  * OptOut (IsOptOut =='N') scoring *  "
    echo_and_run python2 DetectionScorer.py -o $compcheckfile_outdir/$checkfile_outdir_basename-3 \
                          -t manipulation \
                          --refDir $testsuite_directory/reference \
                          -r NC2016-manipulation-ref.csv -x NC2016-manipulation-index.csv \
                          --sysDir $testsuite_directory/baseline \
                          -s Base_NC2016_Manipulation_ImgOnly_p-dct_02_optout.csv  \
                          --optOut --dLevel 0.1 --ci --plotType roc

    echo "  * Reduced (--noNum): legend without the number of target and non-target trials *  "
    echo_and_run python2 DetectionScorer.py -o $compcheckfile_outdir/$checkfile_outdir_basename-4 \
                    -t manipulation \
                    --refDir $testsuite_directory/reference \
                    -r NC2016-manipulation-ref.csv \
                    -x NC2016-manipulation-index.csv \
                    --sysDir $testsuite_directory/baseline \
                    -s Base_NC2016_Manipulation_ImgOnly_p-dct_02.csv \
                    --ci --noNum

    echo "  * Query (-q) with one query *  "
    echo_and_run python2 DetectionScorer.py -o $compcheckfile_outdir/$checkfile_outdir_basename-5 \
                      -t manipulation \
                      --refDir $testsuite_directory/reference \
                      -r NC2016-manipulation-ref.csv -x NC2016-manipulation-index.csv \
                      --sysDir $testsuite_directory/baseline \
                      -s Base_NC2016_Manipulation_ImgOnly_p-dct_02.csv  \
                      -q "Collection==['Nimble-SCI','Nimble-WEB']" --ci

    echo "  * Query (-q) with two queries *  "
    echo_and_run python2 DetectionScorer.py -o $compcheckfile_outdir/$checkfile_outdir_basename-6 \
                 -t manipulation \
                 --refDir $testsuite_directory/reference \
                 -r NC2016-manipulation-ref.csv -x NC2016-manipulation-index.csv \
                 --sysDir $testsuite_directory/baseline \
                 -s Base_NC2016_Manipulation_ImgOnly_p-dct_02.csv  \
                 -q "Collection==['Nimble-SCI'] & 300 <= ProbeWidth" "Collection==['Nimble-WEB'] & 300 <= ProbeWidth" \
                 --ci

    echo "  * Query for partition (-qp) with one partition *  "
    echo_and_run python2 DetectionScorer.py -o $compcheckfile_outdir/$checkfile_outdir_basename-7 \
                 -t manipulation \
                 --refDir $testsuite_directory/reference \
                 -r NC2016-manipulation-ref.csv -x NC2016-manipulation-index.csv \
                 --sysDir $testsuite_directory/baseline \
                 -s Base_NC2016_Manipulation_ImgOnly_p-dct_02.csv  \
                 -qp "Collection==['Nimble-SCI'] & 300 <= ProbeWidth" \
                 --ci

    echo "  * Query for partition (-qp) with two partitions  *  "
    echo_and_run python2 DetectionScorer.py -o $compcheckfile_outdir/$checkfile_outdir_basename-8 \
                -t manipulation \
                --refDir $testsuite_directory/reference \
                -r NC2016-manipulation-ref.csv \
                -x NC2016-manipulation-index.csv \
                --sysDir $testsuite_directory/baseline \
                -s Base_NC2016_Manipulation_ImgOnly_p-dct_02.csv \
                -qp "Collection==['Nimble-SCI','Nimble-WEB'] & 300 <= ProbeWidth" \
                --ci

    echo "  * Query for selective manipulation (-qm) with two queries   *  "
    echo_and_run python2 DetectionScorer.py -o $compcheckfile_outdir/$checkfile_outdir_basename-9 \
                -t manipulation \
                --refDir $testsuite_directory/reference \
                -r NC2016-manipulation-ref.csv \
                -x NC2016-manipulation-index.csv \
                --sysDir $testsuite_directory/baseline \
                -s Base_NC2016_Manipulation_ImgOnly_p-dct_02.csv \
                -qm "Collection==['Nimble-SCI'] & IsManipulationTypeRemoval==['Y']" "Collection==['Nimble-WEB'] & IsManipulationTypeRemoval==['Y']" \


    echo "  * --multiFigs with the query option    *  "
    echo_and_run python2 DetectionScorer.py -o $compcheckfile_outdir/$checkfile_outdir_basename-10 \
                -t manipulation \
                --refDir $testsuite_directory/reference \
                -r NC2016-manipulation-ref.csv \
                -x NC2016-manipulation-index.csv \
                --sysDir $testsuite_directory/baseline \
                -s Base_NC2016_Manipulation_ImgOnly_p-dct_02.csv \
                -qp "Collection==['Nimble-SCI','Nimble-WEB'] & 300 <= ProbeWidth" --multiFigs \
                --ci

    echo "  * Splice Task    *  "
    echo_and_run python2 DetectionScorer.py -o $compcheckfile_outdir/$checkfile_outdir_basename-11 \
                -t splice \
                --refDir $testsuite_directory/reference \
                -r NC2016-splice-ref.csv -x NC2016-splice-index.csv \
                --sysDir $testsuite_directory/baseline \
                -s Base_NC2016_Splice_ImgOnly_p-splice_01.csv \
                --ci

    echo "  * NC2017 Full scoring: rendering ROC curve   *  "
    echo_and_run python2 DetectionScorer.py -o $compcheckfile_outdir/$checkfile_outdir_basename-12 \
                -t manipulation \
                --refDir $testsuite_directory/ \
                -r reference/NC2017-manipulation-ref.csv \
                -x reference/NC2017-manipulation-index.csv \
                --sysDir $testsuite_directory/baseline \
                -s Base_NC2017_Manipulation_ImgOnly_p-copymove_01.csv \
                --ci

    echo "  * NC2017 Query (-q) with two queries    *  "
    echo_and_run python2 DetectionScorer.py -o $compcheckfile_outdir/$checkfile_outdir_basename-13 \
                -t manipulation \
                --refDir $testsuite_directory/ \
                -r reference/NC2017-manipulation-ref.csv \
                -x reference/NC2017-manipulation-index.csv \
                --sysDir $testsuite_directory/baseline \
                -s Base_NC2017_Manipulation_ImgOnly_p-copymove_01.csv \
                -q "(Purpose ==['remove'] and IsTarget == ['Y']) or IsTarget == ['N']" "(Purpose ==['clone'] and IsTarget == ['Y']) or IsTarget == ['N']"


    echo "  * NC2017 Query for selective manipulation (-qm) with the factor Purpose    *  "
    echo_and_run python2 DetectionScorer.py -o $compcheckfile_outdir/$checkfile_outdir_basename-14 \
                -t manipulation \
                --refDir $testsuite_directory/ \
                -r reference/NC2017-manipulation-ref.csv \
                -x reference/NC2017-manipulation-index.csv \
                --sysDir $testsuite_directory/baseline \
                -s Base_NC2017_Manipulation_ImgOnly_p-copymove_01.csv \
                -qm "Purpose==['remove']" "Purpose==['clone']"


    echo "  * NC2017 Query for selective manipulation (-qm) with the factor OperationArgument     *  "
    echo_and_run python2 DetectionScorer.py -o $compcheckfile_outdir/$checkfile_outdir_basename-15 \
                -t manipulation \
                --refDir $testsuite_directory/ \
                -r reference/NC2017-manipulation-ref.csv \
                -x reference/NC2017-manipulation-index.csv \
                --sysDir $testsuite_directory/baseline \
                -s Base_NC2017_Manipulation_ImgOnly_p-copymove_01.csv \
                -qm "OperationArgument==['people','face']" "OperationArgument==['man-made object','landscape']"


    echo "  * NC2017 Query for selective manipulation (-qm) with the mixed of factors  *  "
    echo_and_run python2 DetectionScorer.py -o $compcheckfile_outdir/$checkfile_outdir_basename-16 \
                -t manipulation \
                --refDir $testsuite_directory/ \
                -r reference/NC2017-manipulation-ref.csv \
                -x reference/NC2017-manipulation-index.csv \
                --sysDir $testsuite_directory/baseline \
                -s Base_NC2017_Manipulation_ImgOnly_p-copymove_01.csv \
                -qm "Purpose==['remove'] and Operation ==['FillContentAwareFill']"
}
