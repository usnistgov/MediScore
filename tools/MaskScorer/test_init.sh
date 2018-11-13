#!/bin/bash

clean=TRUE
mypython='python2 -OO'

#TODO: define a test file

check_file(){
    ref_file=$1
    sys_file=$2
    comp_file_name=$3
    diff $ref_file $sys_file > $comp_file_name
    if ([ ! -f $comp_file_name ]); then
        echo
        echo "    !!!!! MASK SCORER TEST FAILED AT CASE 1 !!!!!    "
        echo "    Expected $comp_file_name. Failed to generate the file.     "
        echo
        exit 1
    fi    

    filter_comp="cat $comp_file_name | grep -v CVS"
    if test "`eval $filter_comp`" = ""; then
        if [ $clean = "TRUE" ]; then
            rm $sys_file
        fi
        rm $comp_file_name
        return 0
    else
        echo $comp_file_name
        cat $comp_file_name
        return 1
    fi
}

export check_file
export clean
export mypython

