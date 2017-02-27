#!/bin/bash

source integration_tests.sh

run_test test_1_0 "$testsuite_directory/checkfiles/test_1_0.csv"
run_test test_1_0_direct "$testsuite_directory/checkfiles/test_1_0_direct.csv"
run_test test_1_0and1_direct "$testsuite_directory/checkfiles/test_1_0and1_direct.csv"
run_test test_1_1_direct "$testsuite_directory/checkfiles/test_1_1_direct.csv"
run_test test_2_0 "$testsuite_directory/checkfiles/test_2_0.csv"
run_test test_2_1 "$testsuite_directory/checkfiles/test_2_1.csv"
run_test filtering_test_0 "$testsuite_directory/checkfiles/filtering_test_0.csv"
run_test filtering_test_1 "$testsuite_directory/checkfiles/filtering_test_1.csv"
