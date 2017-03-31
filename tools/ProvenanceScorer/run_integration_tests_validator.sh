#!/bin/bash

source integration_tests_validator.sh

run_test name_test_csv 1
run_test name_test_fileNeqDir 1
run_test name_test_underscores 1
run_test name_test_badtask 1
run_test valid_test_min 0
run_test valid_test_conf 0
run_test valid_test_optout 0
run_test valid_test_all 0
run_test invalid_test_comma 1
run_test invalid_test_diffidx 1
run_test invalid_test_moreidx 1
run_test invalid_test_badfilter 1
run_test invalid_test_filtergraph 1
