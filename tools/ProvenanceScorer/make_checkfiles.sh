#!/bin/bash

source integration_tests.sh

test_1_0 "$testsuite_directory/checkfiles/test_1_0.csv"
test_1_0_direct "$testsuite_directory/checkfiles/test_1_0_direct.csv"
test_1_0and1_direct "$testsuite_directory/checkfiles/test_1_0and1_direct.csv"
test_1_1_direct "$testsuite_directory/checkfiles/test_1_1_direct.csv"
test_2_0 "$testsuite_directory/checkfiles/test_2_0.csv"
test_2_1 "$testsuite_directory/checkfiles/test_2_1.csv"
filtering_test_0 "$testsuite_directory/checkfiles/filtering_test_0.csv"
filtering_test_1 "$testsuite_directory/checkfiles/filtering_test_1.csv"
