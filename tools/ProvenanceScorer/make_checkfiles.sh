#!/bin/bash

source integration_tests.sh

test_1_0 "$testsuite_directory/checkfiles/test_1_0"
test_1_0_direct "$testsuite_directory/checkfiles/test_1_0_direct"
test_1_0and1_direct "$testsuite_directory/checkfiles/test_1_0and1_direct"
test_1_1_direct "$testsuite_directory/checkfiles/test_1_1_direct"
test_2_0 "$testsuite_directory/checkfiles/test_2_0"
test_2_1 "$testsuite_directory/checkfiles/test_2_1"
test_3_1 "$testsuite_directory/checkfiles/test_3_1"
filtering_test_0 "$testsuite_directory/checkfiles/filtering_test_0"
filtering_test_1 "$testsuite_directory/checkfiles/filtering_test_1"
filtering_test_2 "$testsuite_directory/checkfiles/filtering_test_2"
filtering_test_0_b "$testsuite_directory/checkfiles/filtering_test_0_b"
filtering_test_1_b "$testsuite_directory/checkfiles/filtering_test_1_b"
filtering_test_2_b "$testsuite_directory/checkfiles/filtering_test_2_b"
IsOptOut_test_0 "$testsuite_directory/checkfiles/IsOptOut_test_0"