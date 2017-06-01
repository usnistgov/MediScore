#!/bin/bash

source detection_tests.sh

echo
echo "CASE 0: VALIDATING FULL SCORING WITH BASELINEs"
echo
run_test test_c1_1 "$testsuite_directory/checkfiles/test_c1_1"
run_test test_c1_2 "$testsuite_directory/checkfiles/test_c1_2"
run_test test_c1_3 "$testsuite_directory/checkfiles/test_c1_3"
