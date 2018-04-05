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

    # Replace paths in logfile
    log_fn="${compcheckfile_outdir}/log.txt"
    if [ -f "$log_fn" ]; then
	sed -e "s:${compcheckfile_outdir}/:${checkfile_outdir}/:g" "$log_fn" >"${log_fn}.new"
	mv "${log_fn}.new" "$log_fn"
    fi
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

# Graph Building test 1_0_undirect
test_1_0_undirect() {
    ./ProvenanceGraphBuildingScorer.py -o "$1" \
				       -x "$testsuite_directory/test_case_1-provenancegraphbuilding-index.csv" \
				       -r "$testsuite_directory/test_case_1-provenance-ref.csv" \
				       -n "$testsuite_directory/test_case_1-provenance-node.csv" \
				       -w "$testsuite_directory/test_case_1-provenancegraphbuilding-world.csv" \
				       -R "$testsuite_directory/" \
				       -s "$testsuite_directory/test_case_1-system_output_0_index_undirect.csv" \
				       -S "$testsuite_directory/" \
				       --undirected
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

# Graph Building test 2_1 undirected
test_2_1_undirect() {
    ./ProvenanceGraphBuildingScorer.py -o "$1" \
				       -x "$testsuite_directory/test_case_2-provenancegraphbuilding-index.csv" \
				       -r "$testsuite_directory/test_case_2-provenance-ref.csv" \
				       -n "$testsuite_directory/test_case_2-provenance-node.csv" \
				       -w "$testsuite_directory/test_case_2-provenancegraphbuilding-world.csv" \
				       -u \
				       -R "$testsuite_directory/" \
				       -s "$testsuite_directory/test_case_2-system_output_1_undirect_index.csv" \
				       -S "$testsuite_directory/"
}

# Graph Building test 3_1
test_3_1() {
    mkdir -p "$1"
    ./ProvenanceGraphBuildingScorer.py -o "$1" \
				       -c \
				       -v \
				       -x "$testsuite_directory/test_case_2-provenancegraphbuilding-index.csv" \
				       -r "$testsuite_directory/test_case_2-provenance-ref.csv" \
				       -n "$testsuite_directory/test_case_2-provenance-node.csv" \
				       -w "$testsuite_directory/test_case_2-provenancegraphbuilding-world.csv" \
				       -R "$testsuite_directory/" \
				       -s "$testsuite_directory/test_case_3-system_output_1_index.csv" \
				       -S "$testsuite_directory/" >&"$1/log.txt"
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

# Filtering test 2 (3_1)
filtering_test_2() {
    ./ProvenanceFilteringScorer.py -o "$1" \
				   -x "$testsuite_directory/test_case_3-provenancegraphbuilding-index.csv" \
				   -r "$testsuite_directory/test_case_3-provenance-ref.csv" \
				   -n "$testsuite_directory/test_case_3-provenance-node.csv" \
				   -w "$testsuite_directory/test_case_3-provenancegraphbuilding-world.csv" \
				   -R "$testsuite_directory/" \
				   -s "$testsuite_directory/test_case_3-filtering-system_output_1_index.csv" \
				   -S "$testsuite_directory/"
}

filtering_test_0_b() {
	./ProvenanceFilteringScorer.py -o "$1" \
				-S "$testsuite_directory/" \
				-R "$testsuite_directory/" \
				-x "$testsuite_directory/test_case_2-provenancegraphbuilding-index.csv" \
				-r "$testsuite_directory/test_case_2-provenance-ref.csv" \
				-n "$testsuite_directory/test_case_2-provenance-node.csv" \
				-w "$testsuite_directory/test_case_2-provenancegraphbuilding-world.csv" \
				-s "$testsuite_directory/test_case_2-filtering-system_output_0_index.csv" \
				--nodetype all
}

filtering_test_1_b() {
	./ProvenanceFilteringScorer.py -o "$1" \
				-S "$testsuite_directory/" \
				-R "$testsuite_directory/" \
				-x "$testsuite_directory/test_case_2-provenancegraphbuilding-index.csv" \
				-r "$testsuite_directory/test_case_2-provenance-ref.csv" \
				-n "$testsuite_directory/test_case_2-provenance-node.csv" \
				-w "$testsuite_directory/test_case_2-provenancegraphbuilding-world.csv" \
				-s "$testsuite_directory/test_case_2-filtering-system_output_1_index.csv" \
				--nodetype all
}

filtering_test_2_b() {
	./ProvenanceFilteringScorer.py -o "$1" \
				-S "$testsuite_directory/" \
				-R "$testsuite_directory/" \
				-x "$testsuite_directory/test_case_3-provenancegraphbuilding-index.csv" \
				-r "$testsuite_directory/test_case_3-provenance-ref.csv" \
				-n "$testsuite_directory/test_case_3-provenance-node.csv" \
				-w "$testsuite_directory/test_case_3-provenancegraphbuilding-world.csv" \
				-s "$testsuite_directory/test_case_3-filtering-system_output_1_index.csv" \
				--nodetype all
}

# Filtering test 2 (3_1)
IsOptOut_test_0() {
    ./ProvenanceFilteringScorer.py -o "$1" \
				   -x "$testsuite_directory/test_case_3-provenancegraphbuilding-index.csv" \
				   -r "$testsuite_directory/test_case_3-provenance-ref.csv" \
				   -n "$testsuite_directory/test_case_3-provenance-node.csv" \
				   -w "$testsuite_directory/test_case_3-provenancegraphbuilding-world.csv" \
				   -R "$testsuite_directory/" \
				   -s "$testsuite_directory/test_case_4-filtering-system_output_0_index.csv" \
				   -S "$testsuite_directory/"
}
