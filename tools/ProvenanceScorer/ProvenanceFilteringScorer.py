#!/usr/bin/env python2

import sys
import os
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib")
sys.path.append(lib_path)

import json
import argparse
import pandas as pd
import errno
import collections

from ProvenanceMetrics import *

def err_quit(msg, exit_status=1):
    print(msg)
    exit(exit_status)

def load_json(json_fn):
    try:
        with open(json_fn, 'r') as json_f:
            return json.load(json_f)
    except IOError as ioerr:
        err_quit("{}. Aborting!".format(ioerr))

def load_csv(csv_fn, sep="|"):
    try:
        return pd.read_csv(csv_fn, sep)
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
    node_set_w_confidence = [ node_w_confidence(n["nodeConfidenceScore"], n["file"]) for n in system_out["nodes"] ]
    node_set_w_confidence.sort(reverse=True)
    return node_set_w_confidence

# Can't use just a hash here, as we need to enforce column order
def build_dataframe(columns, fields):
    df = pd.DataFrame(columns=columns)

    
    # Setting column data type one by one as pandas doesn't offer a
    # convenient way to do this
    for col, t in fields.items():
        df[col] = df[col].astype(t)

    return df

def build_provenancefiltering_agg_output_df():
    return build_dataframe(["MeanNodeRecallAt50",
                            "MeanNodeRecallAt100",
                            "MeanNodeRecallAt200"],
                           { "MeanNodeRecallAt50": float,
                             "MeanNodeRecallAt100": float,
                             "MeanNodeRecallAt200": float })

def build_provenancefiltering_nodemapping_df():
    return build_dataframe(["JournalName",
                            "ProvenanceProbeFileID",
                            "ProvenanceOutputFileName",
                            "Measure",
                            "WorldFileID",
                            "Mapping"],
                           { "JournalName": str,
                             "ProvenanceProbeFileID": str,
                             "ProvenanceOutputFileName": str,
                             "Measure": str,
                             "WorldFileID": str,
                             "Mapping": str })

def build_provenancefiltering_output_df():
    return build_dataframe(["JournalName",
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
                            "NodeRecallAt200"],
                           { "JournalName": str,
                             "ProvenanceProbeFileID": str,
                             "ProvenanceOutputFileName": str,
                             "NumSysNodes": int,
                             "NumRefNodes": int,
                             "NumCorrectNodesAt50": int,
                             "NumMissingNodesAt50": int,
                             "NumFalseAlarmNodesAt50": int,
                             "NumCorrectNodesAt100": int,
                             "NumMissingNodesAt100": int,
                             "NumFalseAlarmNodesAt100": int,
                             "NumCorrectNodesAt200": int,
                             "NumMissingNodesAt200": int,
                             "NumFalseAlarmNodesAt200": int,
                             "NodeRecallAt50": float,
                             "NodeRecallAt100": float,
                             "NodeRecallAt200": float })

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
    args = parser.parse_args()

    trial_index = load_csv(args.index_file)
    ref_file = load_csv(args.reference_file)
    nodes_file = load_csv(args.node_file)
    world_index = load_csv(args.world_file)

    system_output_index = load_csv(args.system_output_file)

    def check_for_trial_disparity():
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
            errs.append("Error, extraneous ProvenanceProbeFileIDs in the system output:\n{}".format("\n".join(map(lambda p: "\t" + p, remaining_sys_probes))))

        if len(errs) > 0:
            errs.append("Aborting!")
            err_quit("\n".join(errs))

    if not args.skip_trial_disparity_check:
        check_for_trial_disparity()
            
    trial_index_ref = pd.merge(trial_index, ref_file, on = "ProvenanceProbeFileID")
    trial_index_ref_sysout = pd.merge(trial_index_ref, system_output_index, on = "ProvenanceProbeFileID")

    world_nodes = pd.merge(nodes_file, world_index, on = "WorldFileID", how = "inner")

    output_records = build_provenancefiltering_output_df()
    output_mapping_records = build_provenancefiltering_nodemapping_df()
        
    for trial in trial_index_ref_sysout.itertuples():
        system_out = load_json(os.path.join(args.system_dir, trial.ProvenanceOutputFileName))
        
        probe_node_wfn = trial.ProvenanceProbeFileName_x
        world_set_nodes = { node.WorldFileName_x for node in world_nodes[world_nodes.ProvenanceProbeFileID == trial.ProvenanceProbeFileID].itertuples() }
        world_set_nodes.add(probe_node_wfn)

        ordered_sys_nodes = system_out_to_ordered_nodes(system_out)
            
        out_rec = { "JournalName": trial.JournalName,
                    "ProvenanceProbeFileID": trial.ProvenanceProbeFileID,
                    "ProvenanceOutputFileName": trial.ProvenanceOutputFileName,
                    "NumSysNodes": len(ordered_sys_nodes),
                    "NumRefNodes": len(world_set_nodes) }

        def _worldfile_path_to_id(path):
            base, ext = os.path.splitext(os.path.basename(path))
            return base
        
        def _build_node_map_record(node, mapping):
            return { "JournalName": trial.JournalName,
                     "ProvenanceProbeFileID": trial.ProvenanceProbeFileID,
                     "ProvenanceOutputFileName": trial.ProvenanceOutputFileName,
                     "Measure": "NodeRecallAt{}".format(n),
                     "WorldFileID": _worldfile_path_to_id(node),
                     "Mapping": mapping }

        for n in [ 50, 100, 200 ]:
            sys_nodes_at_n = { node.file for node in ordered_sys_nodes[0:n] }

            correct_nodes = sys_nodes_at_n & world_set_nodes
            missing_nodes = world_set_nodes - sys_nodes_at_n
            false_alarm_nodes = sys_nodes_at_n - world_set_nodes
            
            out_rec.update({ "NumCorrectNodesAt{}".format(n): len(correct_nodes),
                             "NumMissingNodesAt{}".format(n): len(missing_nodes),
                             "NumFalseAlarmNodesAt{}".format(n): len(false_alarm_nodes),
                             "NodeRecallAt{}".format(n): node_recall(world_set_nodes, sys_nodes_at_n) })

            for map_record in sorted([ _build_node_map_record(node, "Correct") for node in correct_nodes ] +
                                     [ _build_node_map_record(node, "Missing") for node in missing_nodes ] +
                                     [ _build_node_map_record(node, "FalseAlarm") for node in false_alarm_nodes ]):
                output_mapping_records = output_mapping_records.append(pd.Series(map_record), ignore_index=True)
            
        output_records = output_records.append(pd.Series(out_rec), ignore_index=True)

    output_agg_records = build_provenancefiltering_agg_output_df()
    aggregated = { "MeanNodeRecallAt50": output_records["NodeRecallAt50"].mean(),
                   "MeanNodeRecallAt100": output_records["NodeRecallAt100"].mean(),
                   "MeanNodeRecallAt200": output_records["NodeRecallAt200"].mean() }
    output_agg_records = output_agg_records.append(pd.Series(aggregated), ignore_index=True)

    mkdir_p(args.output_dir)

    def _write_df_to_csv(df, out_fn):
        try:
            with open(os.path.join(args.output_dir, out_fn), 'w') as out_f:
                df.to_csv(path_or_buf=out_f, sep="|", index=False)
        except IOError as ioerr:
            err_quit("{}. Aborting!".format(ioerr))
                
    _write_df_to_csv(output_records, "trial_scores.csv")
    _write_df_to_csv(output_agg_records, "scores.csv")
    _write_df_to_csv(output_mapping_records, "node_mapping.csv")

    if args.html_report == True:
        try:
            pd.set_option('display.max_colwidth', -1) # Keep pandas from truncating our links
            with open(os.path.join(args.output_dir, "report.html"), 'w') as out_f:
                out_f.write("<h2>Aggregated Scores:</h2>")
                output_agg_records.to_html(buf=out_f, index=False)
                out_f.write("<br/><br/>")
                out_f.write("<h2>Trial Scores:</h2>")
                output_records.to_html(buf=out_f, index=False)
        except IOError as ioerr:
            err_quit("{}. Aborting!".format(ioerr))
