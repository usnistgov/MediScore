#!/bin/bash

source integration_tests.sh

run_test test_1_0 "$testsuite_directory/checkfiles/test_1_0"
run_test test_1_0_direct "$testsuite_directory/checkfiles/test_1_0_direct"
run_test test_1_0and1_direct "$testsuite_directory/checkfiles/test_1_0and1_direct"
run_test test_1_1_direct "$testsuite_directory/checkfiles/test_1_1_direct"
run_test test_2_0 "$testsuite_directory/checkfiles/test_2_0"
run_test test_2_1 "$testsuite_directory/checkfiles/test_2_1"
run_test test_3_1 "$testsuite_directory/checkfiles/test_3_1"
run_test filtering_test_0 "$testsuite_directory/checkfiles/filtering_test_0"
run_test filtering_test_1 "$testsuite_directory/checkfiles/filtering_test_1"
run_test filtering_test_2 "$testsuite_directory/checkfiles/filtering_test_2"
run_test IsOptOut_test_0 "$testsuite_directory/checkfiles/IsOptOut_test_0"