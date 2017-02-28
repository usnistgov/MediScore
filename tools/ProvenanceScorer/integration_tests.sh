#!/bin/bash

export testsuite_directory=../../data/test_suite/provenanceScorerTests

check_status() {
    status=$?
    if [ $status -ne 0 ]; then
	echo "*** FAILED ***"
	exit $status
    fi
}

run_test() {
    test=$1
    checkfile_outdir=$2
    checkfile_outdir_basename=`basename $checkfile_outdir`
    compcheck_outdir=${3-compcheckfiles}
    compcheckfile_outdir="$compcheck_outdir/$checkfile_outdir_basename"

    echo "** Running integration test '$test' **"
    $test "$compcheckfile_outdir"
    check_status
    
    diff -r "$checkfile_outdir" "$compcheckfile_outdir"
    check_status
    
    echo "*** OK ***"
}

# Graph Building test 1_0
test_1_0() {
    ./ProvenanceGraphBuildingScorer.py -o "$1" \
				       -x "$testsuite_directory/test_case_1-provenancegraphbuilding-index.csv" \
				       -r "$testsuite_directory/test_case_1-provenance-ref.csv" \
				       -n "$testsuite_directory/test_case_1-provenance-node.csv" \
				       -w "$testsuite_directory/test_case_1-provenancegraphbuilding-world.csv" \
				       -R "$testsuite_directory/" \
				       -s "$testsuite_directory/test_case_1-system_output_0_index.csv" \
				       -S "$testsuite_directory/"
}

# Graph Building test 1_0_direct
test_1_0_direct() {
    ./ProvenanceGraphBuildingScorer.py -d -o "$1" \
				       -x "$testsuite_directory/test_case_1-provenancegraphbuilding-index.csv" \
				       -r "$testsuite_directory/test_case_1-provenance-ref.csv" \
				       -n "$testsuite_directory/test_case_1-provenance-node.csv" \
				       -w "$testsuite_directory/test_case_1-provenancegraphbuilding-world.csv" \
				       -R "$testsuite_directory/" \
				       -s "$testsuite_directory/test_case_1-system_output_0_index_direct.csv" \
				       -S "$testsuite_directory/"
}

# Graph Building test 1_0_direct
test_1_0and1_direct() {
    ./ProvenanceGraphBuildingScorer.py -d -t -o "$1" \
				       -x "$testsuite_directory/test_case_1-provenancegraphbuilding-index.csv" \
				       -r "$testsuite_directory/test_case_1-provenance-ref.csv" \
				       -n "$testsuite_directory/test_case_1-provenance-node.csv" \
				       -w "$testsuite_directory/test_case_1-provenancegraphbuilding-world.csv" \
				       -R "$testsuite_directory/" \
				       -s "$testsuite_directory/test_case_1-system_output_0and1_index_direct.csv" \
				       -S "$testsuite_directory/"
}

# Graph Building test 1_0_direct
test_1_1_direct() {
    ./ProvenanceGraphBuildingScorer.py -d -o "$1" \
				       -x "$testsuite_directory/test_case_1-provenancegraphbuilding-index.csv" \
				       -r "$testsuite_directory/test_case_1-provenance-ref.csv" \
				       -n "$testsuite_directory/test_case_1-provenance-node.csv" \
				       -w "$testsuite_directory/test_case_1-provenancegraphbuilding-world.csv" \
				       -R "$testsuite_directory/" \
				       -s "$testsuite_directory/test_case_1-system_output_1_index_direct.csv" \
				       -S "$testsuite_directory/"
}

# Graph Building test 2_0
test_2_0() {
    ./ProvenanceGraphBuildingScorer.py -o "$1" \
				       -x "$testsuite_directory/test_case_2-provenancegraphbuilding-index.csv" \
				       -r "$testsuite_directory/test_case_2-provenance-ref.csv" \
				       -n "$testsuite_directory/test_case_2-provenance-node.csv" \
				       -w "$testsuite_directory/test_case_2-provenancegraphbuilding-world.csv" \
				       -R "$testsuite_directory/" \
				       -s "$testsuite_directory/test_case_2-system_output_0_index.csv" \
				       -S "$testsuite_directory/"
}

# Graph Building test 2_1
test_2_1() {
    ./ProvenanceGraphBuildingScorer.py -o "$1" \
				       -x "$testsuite_directory/test_case_2-provenancegraphbuilding-index.csv" \
				       -r "$testsuite_directory/test_case_2-provenance-ref.csv" \
				       -n "$testsuite_directory/test_case_2-provenance-node.csv" \
				       -w "$testsuite_directory/test_case_2-provenancegraphbuilding-world.csv" \
				       -R "$testsuite_directory/" \
				       -s "$testsuite_directory/test_case_2-system_output_1_index.csv" \
				       -S "$testsuite_directory/"
}

# Filtering test 0 (2_0)
filtering_test_0() {
    ./ProvenanceFilteringScorer.py -o "$1" \
				   -x "$testsuite_directory/test_case_2-provenancegraphbuilding-index.csv" \
				   -r "$testsuite_directory/test_case_2-provenance-ref.csv" \
				   -n "$testsuite_directory/test_case_2-provenance-node.csv" \
				   -w "$testsuite_directory/test_case_2-provenancegraphbuilding-world.csv" \
				   -R "$testsuite_directory/" \
				   -s "$testsuite_directory/test_case_2-filtering-system_output_0_index.csv" \
				   -S "$testsuite_directory/"
}

# Filtering test 1 (2_1)
filtering_test_1() {
    ./ProvenanceFilteringScorer.py -o "$1" \
				   -x "$testsuite_directory/test_case_2-provenancegraphbuilding-index.csv" \
				   -r "$testsuite_directory/test_case_2-provenance-ref.csv" \
				   -n "$testsuite_directory/test_case_2-provenance-node.csv" \
				   -w "$testsuite_directory/test_case_2-provenancegraphbuilding-world.csv" \
				   -R "$testsuite_directory/" \
				   -s "$testsuite_directory/test_case_2-filtering-system_output_1_index.csv" \
				   -S "$testsuite_directory/"
}
