#!/bin/bash

source integration_tests.sh

#TODO: do this for video spatial

echo "PROCEEDING TO THOROUGHLY CHECK ALL CASES"
echo
echo "CASE 0: VALIDATING FULL SCORING"
echo

basic_test_wrapper
basic_collar_test_wrapper

