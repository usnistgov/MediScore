#!/bin/bash

exitstatus=0
set_status(){
    prev_exit_status=$?
    exitstatus=$(($prev_exit_status | $exitstatus))
} 

source integration_tests.sh

echo "PROCEEDING TO THOROUGHLY CHECK ALL CASES"
echo
echo "CASE 0: VALIDATING FULL SCORING"
echo

gen_masks
basic_test_wrapper
set_status
basic_test_noed_wrapper
set_status
basic_collar_test_wrapper
set_status
basic_video_oo_test_wrapper
set_status
err_test_wrapper
set_status

if [[ $exitstatus == 0 ]]; then
    echo "CASE 0 VALIDATED."
fi

echo
echo "CASE 1: VALIDATING SELECTIVE SCORING"
echo

selective_tests_wrapper
set_status

if [[ $exitstatus == 0 ]]; then
    echo "CASE 1 VALIDATED."
fi

rm_masks

exit $exitstatus
