#!/usr/bin/env python2

import sys
import os
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib")
sys.path.append(lib_path)

import json
import argparse
from pandas import DataFrame, read_csv, merge, set_option
import errno
import collections

from ProvenanceMetrics import *

def err_quit(msg, exit_status=1):
    print(msg)
    exit(exit_status)

def build_logger(verbosity_threshold=0):
    def _log(depth, msg):
        if depth <= verbosity_threshold:
            print(msg)

    return _log

def load_json(json_fn):
    try:
        with open(json_fn, 'r') as json_f:
            return json.load(json_f)
    except IOError as ioerr:
        err_quit("{}. Aborting!".format(ioerr))

def load_csv(csv_fn, sep="|"):
    try:
        return read_csv(csv_fn, sep)
    except IOError as ioerr:
        err_quit("{}. Aborting!".format(ioerr))

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            err_quit("{}. Aborting!".format(exc))

# Returns ordered array of named tuples representing nodes
def system_out_to_ordered_nodes(system_out):
    node_w_confidence = collections.namedtuple('node_w_confidence', [ 'confidence', 'file' ])
    node_set_w_confidence = []
    sys_nodes_dict = {}
    for n in system_out["nodes"]:
        node_set_w_confidence.append(node_w_confidence(n["nodeConfidenceScore"], n["file"]))
        sys_nodes_dict[n["file"]] = n

    node_set_w_confidence.sort(reverse=True)
    return node_set_w_confidence, sys_nodes_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Score Medifor ProvenanceFiltering task output")
    parser.add_argument("-t", "--skip-trial-disparity-check", help="Skip check for trial disparity between INDEX_FILE and SYSTEM_OUTPUT_FILE", action="store_true")
    parser.add_argument("-o", "--output-dir", help="Output directory for scores", type=str, required=True)
    parser.add_argument("-x", "--index-file", help="Task Index file", type=str, required=True)
    parser.add_argument("-r", "--reference-file", help="Reference file", type=str, required=True)
    parser.add_argument("-n", "--node-file", help="Node file", type=str, required=True)
    parser.add_argument("-w", "--world-file", help="World file", type=str, required=True)
    parser.add_argument("-R", "--reference-dir", help="Reference directory", type=str, required=True)
    parser.add_argument("-s", "--system-output-file", help="System output file (i.e. <EXPID>.csv)", type=str, required=True)
    parser.add_argument("-S", "--system-dir", help="System output directory where system output json files can be found", type=str, required=True)
    parser.add_argument("-H", "--html-report", help="Generate an HTML report of the scores", action="store_true")
    parser.add_argument("-v", "--verbose", help="Toggle verbose log output", action="store_true")
    args = parser.parse_args()

    # Logger setup, could eventually support different levels of
    # verbosity
    verbosity_threshold = 1 if args.verbose else 0
    log = build_logger(verbosity_threshold)

    trial_index = load_csv(args.index_file)
    ref_file = load_csv(args.reference_file)
    nodes_file = load_csv(args.node_file)
    world_index = load_csv(args.world_file)

    abs_reference_dir = os.path.abspath(args.reference_dir)

    system_output_index = load_csv(args.system_output_file)

    def check_for_trial_disparity(only_warn_on_extraneous = False):
        # detect missing trials
        ref_probes = [ t.ProvenanceProbeFileID for t in trial_index.itertuples() ]
        remaining_sys_probes = [ t.ProvenanceProbeFileID for t in system_output_index.itertuples() ]

        missing_probes = []
        for probe_id in ref_probes:
            try:
                remaining_sys_probes.remove(probe_id)
            except ValueError:
                missing_probes.append(probe_id)

        errs = []
        if len(missing_probes) > 0:
            errs.append("Error, missing the following ProvenanceProbeFileIDs from the system output:\n{}".format("\n".join(map(lambda p: "\t" + p, missing_probes))))
        if len(remaining_sys_probes) > 0:
            if only_warn_on_extraneous:
                log(1, "Warning, found {} extraneous ProvenanceProbeFileIDs in the system output.".format(len(remaining_sys_probes)))
            else:
                errs.append("Error, extraneous ProvenanceProbeFileIDs in the system output:\n{}".format("\n".join(map(lambda p: "\t" + p, remaining_sys_probes))))

        if len(errs) > 0:
            errs.append("Aborting!")
            err_quit("\n".join(errs))

    check_for_trial_disparity(args.skip_trial_disparity_check)
    log(1, "Scoring {} trials ..".format(len(trial_index)))

    # Remove NonProcessed (or IsOptOut) trials
    if "IsOptOut" in system_output_index.columns:
        system_output_index = system_output_index.query("IsOptOut == ['Processed']")
    elif "ProvenanceProbeStatus" in system_output_index.columns:
        system_output_index = system_output_index.query("ProvenanceProbeStatus == ['Processed']")

    trial_index_ref = merge(trial_index, ref_file, on = "ProvenanceProbeFileID")
    trial_index_ref_sysout = merge(trial_index_ref, system_output_index, on = "ProvenanceProbeFileID")

    world_nodes = merge(nodes_file, world_index, on = ["WorldFileID", "WorldFileName"], how = "inner")

    output_records = []
    output_mapping_records = []

    for journal_fn, trial_index_ref_sysout_items in trial_index_ref_sysout.groupby("JournalFileName"):
        journal_path = os.path.join(abs_reference_dir, journal_fn)
        journal = load_json(journal_path)

        journal_node_lookup = {}
        for n in journal["nodes"]:
            journal_node_lookup[n["id"]] = n

        for trial in trial_index_ref_sysout_items.itertuples():
            log(1, "Working on ProvenanceProbeFileID: '{}' in JournalName: '{}'".format(trial.ProvenanceProbeFileID, trial.JournalName))
            system_out = load_json(os.path.join(args.system_dir, trial.ProvenanceOutputFileName))

            ref_nodes_dict = {}
            for n in world_nodes[world_nodes.ProvenanceProbeFileID == trial.ProvenanceProbeFileID].itertuples():
                ref_nodes_dict[n.WorldFileName] = journal_node_lookup[n.JournalNodeID]

            # This should only add a single node, the probe node,
            # doing this in a loop because Pandas
            for n in nodes_file[(nodes_file.ProvenanceProbeFileID == trial.ProvenanceProbeFileID) & (nodes_file.WorldFileName == trial.ProvenanceProbeFileName_x)].itertuples():
                ref_nodes_dict[n.WorldFileName] = journal_node_lookup[n.JournalNodeID]

            ordered_sys_nodes, full_sys_nodes_dict = system_out_to_ordered_nodes(system_out)

            out_rec = { "JournalName": trial.JournalName,
                        "ProvenanceProbeFileID": trial.ProvenanceProbeFileID,
                        "ProvenanceOutputFileName": trial.ProvenanceOutputFileName,
                        "NumSysNodes": len(ordered_sys_nodes),
                        "NumRefNodes": len(ref_nodes_dict.keys()) }

            def _worldfile_path_to_id(path):
                base, ext = os.path.splitext(os.path.basename(path))
                return base

            def _build_mapping(ref_nodes, sys_nodes):
                node_mapping = [ (nk, ref_nodes.get(nk, None), sys_nodes.get(nk, None)) for nk in set(ref_nodes.keys()) | set(sys_nodes.keys()) ]

                # a *_mapping file is a collection of (ref_*, sys_*)
                # tuples, where ref_* is None in the case of a FA and
                # sys_* is None in the case of a miss
                return node_mapping

            def _get_mapping(r, s):
                mapping = "Correct"
                if r == None:
                    mapping = "FalseAlarm"
                    if s == None:
                        # Should never have a case where both ref and
                        # sys are None, but let's check to be sure
                        raise ValueError("Shouldn't have None for both ref and sys")
                elif s == None:
                    mapping = "Missing"
                return mapping

            def _build_node_map_record(n, node_key, ref_node, sys_node):
                return { "JournalName": trial.JournalName,
                         "ProvenanceProbeFileID": trial.ProvenanceProbeFileID,
                         "ProvenanceOutputFileName": trial.ProvenanceOutputFileName,
                         "Measure": "NodeRecallAt{}".format(n),
                         "WorldFileID": _worldfile_path_to_id(node_key),
                         "NodeConfidence": sys_node["nodeConfidenceScore"] if sys_node != None else None,
                         "Mapping": _get_mapping(ref_node, sys_node) }

            def _corr_selector(t):
                k, r, s = t
                return (r != None and s != None)

            def _fa_selector(t):
                k, r, s = t
                return (r == None and s != None)

            def _miss_selector(t):
                k, r, s = t
                return (r != None and s == None)

            def _mapping_breakdown(node_mapping):
                return ({ k for k, r, s in filter(_corr_selector, node_mapping) },
                        { k for k, r, s in filter(_miss_selector, node_mapping) },
                        { k for k, r, s in filter(_fa_selector, node_mapping) })

            for n in [ 50, 100, 200 ]:
                sys_nodes_at_n = { node.file for node in ordered_sys_nodes[0:n] }

                sys_nodes_dict = { n: full_sys_nodes_dict[n] for n in sys_nodes_at_n }

                node_mapping = _build_mapping(ref_nodes_dict, sys_nodes_dict)
                output_mapping_records += sorted([ _build_node_map_record(n, *node_map) for node_map in node_mapping ])

                sys_nodes = set(sys_nodes_dict.keys())
                ref_nodes = set(ref_nodes_dict.keys())
                correct_nodes, missing_nodes, false_alarm_nodes = _mapping_breakdown(node_mapping)

                out_rec.update({ "NumCorrectNodesAt{}".format(n): len(correct_nodes),
                                 "NumMissingNodesAt{}".format(n): len(missing_nodes),
                                 "NumFalseAlarmNodesAt{}".format(n): len(false_alarm_nodes),
                                 "NodeRecallAt{}".format(n): node_recall(ref_nodes, sys_nodes) })

            output_records.append(out_rec)


    output_mapping_records_df = DataFrame(output_mapping_records, columns = ["JournalName",
                                                                             "ProvenanceProbeFileID",
                                                                             "ProvenanceOutputFileName",
                                                                             "Measure",
                                                                             "WorldFileID",
                                                                             "NodeConfidence",
                                                                             "Mapping"])
    output_records_df = DataFrame(output_records, columns = ["JournalName",
                                                             "ProvenanceProbeFileID",
                                                             "ProvenanceOutputFileName",
                                                             "NumSysNodes",
                                                             "NumRefNodes",
                                                             "NumCorrectNodesAt50",
                                                             "NumMissingNodesAt50",
                                                             "NumFalseAlarmNodesAt50",
                                                             "NumCorrectNodesAt100",
                                                             "NumMissingNodesAt100",
                                                             "NumFalseAlarmNodesAt100",
                                                             "NumCorrectNodesAt200",
                                                             "NumMissingNodesAt200",
                                                             "NumFalseAlarmNodesAt200",
                                                             "NodeRecallAt50",
                                                             "NodeRecallAt100",
                                                             "NodeRecallAt200"])
    aggregated = [{ "MeanNodeRecallAt50": output_records_df["NodeRecallAt50"].mean(),
                    "MeanNodeRecallAt100": output_records_df["NodeRecallAt100"].mean(),
                    "MeanNodeRecallAt200": output_records_df["NodeRecallAt200"].mean() }]
    output_agg_records_df = DataFrame(aggregated, columns = ["MeanNodeRecallAt50",
                                                             "MeanNodeRecallAt100",
                                                             "MeanNodeRecallAt200"])

    mkdir_p(args.output_dir)

    def _write_df_to_csv(name, df, out_fn):
        try:
            out_path = os.path.join(args.output_dir, out_fn)
            with open(out_path, 'w') as out_f:
                log(1, "Writing {} to '{}'".format(name, out_path))
                df.to_csv(path_or_buf=out_f, sep="|", index=False)
        except IOError as ioerr:
            err_quit("{}. Aborting!".format(ioerr))

    _write_df_to_csv("Trial Scores", output_records_df, "trial_scores.csv")
    _write_df_to_csv("Aggregate Scores", output_agg_records_df, "scores.csv")
    _write_df_to_csv("Node Mapping", output_mapping_records_df, "node_mapping.csv")

    if args.html_report == True:
        try:
            set_option('display.max_colwidth', -1) # Keep pandas from truncating our links
            report_out_path = os.path.join(args.output_dir, "report.html")
            log(1, "Writing HTML Report to '{}'".format(report_out_path))
            with open(report_out_path, 'w') as out_f:
                out_f.write("<h2>Aggregated Scores:</h2>")
                output_agg_records_df.to_html(buf=out_f, index=False)
                out_f.write("<br/><br/>")
                out_f.write("<h2>Trial Scores:</h2>")
                output_records_df.to_html(buf=out_f, index=False, columns=["JournalName",
                                                                           "ProvenanceProbeFileID",
                                                                           "ProvenanceOutputFileName",
                                                                           "NodeRecallAt50",
                                                                           "NodeRecallAt100",
                                                                           "NodeRecallAt200",
                                                                           "NumSysNodes",
                                                                           "NumRefNodes",
                                                                           "NumCorrectNodesAt50",
                                                                           "NumMissingNodesAt50",
                                                                           "NumFalseAlarmNodesAt50",
                                                                           "NumCorrectNodesAt100",
                                                                           "NumMissingNodesAt100",
                                                                           "NumFalseAlarmNodesAt100",
                                                                           "NumCorrectNodesAt200",
                                                                           "NumMissingNodesAt200",
                                                                           "NumFalseAlarmNodesAt200"])
        except IOError as ioerr:
            err_quit("{}. Aborting!".format(ioerr))
