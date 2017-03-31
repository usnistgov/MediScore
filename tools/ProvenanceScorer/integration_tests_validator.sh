#!/bin/bash

export testsuite_directory=../../data/test_suite/provenanceValidatorTests

check_status() {
    status=$?
    if [ $status -ne 0 ]; then
	echo "*** FAILED ***"
	exit $status
    fi
}

check_fail() {
    status=$?
    if [ $status -ne 1 ]; then
	echo "*** FAILED ***"
	exit 1
    fi
}

run_test() {
    test=$1
    passfail=$2

    echo "** Running integration test '$test' **"
    $test
    if [[ $passfail==0 ]] ; then
        check_status
    else
        check_fail
    fi    

    echo "*** OK ***"
}

name_test_csv() {
    python ProvenanceValidator.py -x "$testsuite_directory/indexes/NC2017_Dev1-provenance-index.csv" \
			     	-s "$testsuite_directory/NameCheck_NC17_FuncTest_1/NameCheck_NC17_FuncTest_ProvenanceFiltering_ImgOnly_p-valid_1/NameCheck_NC17_FuncTest_ProvenanceFiltering_ImgOnly_p-valid_1.csv" \
        			-nc                          
}

name_test_fileNeqDir() {
    python ProvenanceValidator.py -x "$testsuite_directory/indexes/NC2017_Dev1-provenance-index.csv" \
			     	-s "$testsuite_directory/NameCheck_NC17_FuncTest_1/NameCheck_NC17_FuncTest2_Provenance_ImgOnly_p-valid_1/NameCheck_NC17_FuncTest_Provenance_ImgOnly_p-valid_1.csv" \
        			-nc                          
}

name_test_underscores() {
    python ProvenanceValidator.py -x "$testsuite_directory/indexes/NC2017_Dev1-provenance-index.csv" \
			     	-s "$testsuite_directory/NameCheck_NC17_FuncTest_1/NameCheck_NC17_Functionality_Test_3_ProvenanceFiltering_ImgOnly_p-invalid_1/NameCheck_NC17_Functionality_Test_3_Provenance_ImgOnly_p-invalid_1.csv" \
        			-nc                          
}

name_test_badtask() {
    python ProvenanceValidator.py -x "$testsuite_directory/indexes/NC2017_Dev1-provenance-index.csv" \
			     	-s "$testsuite_directory/NameCheck_NC17_FuncTest_1/NameCheck_NC17_FuncTest4_Manipulation_ImgOnly_p-valid_1/NameCheck_NC17_FuncTest4_Manipulation_ImgOnly_p-valid_1.csv" \
        			-nc                          
}

valid_test_min() {
    python ProvenanceValidator.py -x "$testsuite_directory/indexes/NC2017_Dev1-provenance-index.csv" \
			     	-s "$testsuite_directory/ValidCheck_NC17_FuncTest_2/ValidCheck_NC17_FuncTest_ProvenanceFiltering_ImgOnly_p-min_1/ValidCheck_NC17_FuncTest_ProvenanceFiltering_ImgOnly_p-min_1.csv" \
        			-nc
}

valid_test_conf() {
    python ProvenanceValidator.py -x "$testsuite_directory/indexes/NC2017_Dev1-provenance-index.csv" \
			     	-s "$testsuite_directory/ValidCheck_NC17_FuncTest_2/ValidCheck_NC17_FuncTest_Provenance_ImgOnly_p-conf_1/ValidCheck_NC17_FuncTest_Provenance_ImgOnly_p-conf_1.csv" \
        			-nc
}

valid_test_optout() {
    python ProvenanceValidator.py -x "$testsuite_directory/indexes/NC2017_Dev1-provenance-index.csv" \
			     	-s "$testsuite_directory/ValidCheck_NC17_FuncTest_2/ValidCheck_NC17_FuncTest_Provenance_ImgOnly_p-optout_1/ValidCheck_NC17_FuncTest_Provenance_ImgOnly_p-optout_1.csv" \
        			-nc
}

valid_test_all() {
    python ProvenanceValidator.py -x "$testsuite_directory/indexes/NC2017_Dev1-provenance-index.csv" \
			     	-s "$testsuite_directory/ValidCheck_NC17_FuncTest_2/ValidCheck_NC17_FuncTest_Provenance_ImgOnly_p-all_1/ValidCheck_NC17_FuncTest_Provenance_ImgOnly_p-all_1.csv" \
        			-nc
}

invalid_test_comma() {
    python ProvenanceValidator.py -x "$testsuite_directory/indexes/NC2017_Dev1-provenance-index.csv" \
			     	-s "$testsuite_directory/InvalidCheck_NC17_FuncTest_3/InvalidCheck_NC17_FuncTest_ProvenanceFiltering_ImgOnly_p-comma_3/InvalidCheck_NC17_FuncTest_ProvenanceFiltering_ImgOnly_p-comma_3.csv" \
        			-nc
}

invalid_test_diffidx() {
    python ProvenanceValidator.py -x "$testsuite_directory/indexes/NC2017_Dev1-provenance-index.csv" \
			     	-s "$testsuite_directory/InvalidCheck_NC17_FuncTest_3/InvalidCheck_NC17_FuncTest_ProvenanceFiltering_ImgOnly_p-diffidx_3/InvalidCheck_NC17_FuncTest_ProvenanceFiltering_ImgOnly_p-diffidx_3.csv" \
        			-nc
}

invalid_test_moreidx() {
    python ProvenanceValidator.py -x "$testsuite_directory/indexes/NC2017_Dev1-provenance-index.csv" \
			     	-s "$testsuite_directory/InvalidCheck_NC17_FuncTest_3/InvalidCheck_NC17_FuncTest_ProvenanceFiltering_ImgOnly_p-moreidx_3/InvalidCheck_NC17_FuncTest_ProvenanceFiltering_ImgOnly_p-moreidx_3.csv" \
        			-nc
}

invalid_test_badfilter() {
    python ProvenanceValidator.py -x "$testsuite_directory/indexes/NC2017_Dev1-provenance-index.csv" \
			     	-s "$testsuite_directory/InvalidCheck_NC17_FuncTest_3/InvalidCheck_NC17_FuncTest_ProvenanceFiltering_ImgOnly_p-badfilter_3/InvalidCheck_NC17_FuncTest_ProvenanceFiltering_ImgOnly_p-badfilter_3.csv" \
        			-nc
}

invalid_test_filtergraph() {
    python ProvenanceValidator.py -x "$testsuite_directory/indexes/NC2017_Dev1-provenance-index.csv" \
			     	-s "$testsuite_directory/InvalidCheck_NC17_FuncTest_3/InvalidCheck_NC17_FuncTest_Provenance_ImgOnly_p-filtergraph_3/InvalidCheck_NC17_FuncTest_Provenance_ImgOnly_p-filtergraph_3.csv" \
        			-nc
}

