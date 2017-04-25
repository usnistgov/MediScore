#!/usr/bin/env python2

import sys
import os
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib")
sys.path.append(lib_path)

import json
import argparse
import pandas as pd
import errno

from ProvenanceGraphBuilding import *
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
        
def antiforensic_donor_filter(edge_list):
    ins = group_by_fun(lambda e: e["target"], edge_list)

    def _filter(edge):
        return edge["op"] == "Donor" and len(filter(lambda e: e["op"][0:12] == "AntiForensic", ins.get(edge["target"], []))) > 0

    return map(_filter, edge_list)

def system_out_to_scorable(system_out):
    node_lookup = {}
    for i, node in enumerate(system_out["nodes"]):
        node_lookup[i] = node
    
    node_set = { n["file"] for n in system_out["nodes"] }
    edge_set = { (node_lookup[edge["source"]]["file"], node_lookup[edge["target"]]["file"]) for edge in system_out["links"] }
    return (node_set, edge_set)

# Can't use just a hash here, as we need to enforce column order
def build_dataframe(columns, fields):
    df = pd.DataFrame(columns=columns)

    
    # Setting column data type one by one as pandas doesn't offer a
    # convenient way to do this
    for col, t in fields.items():
        df[col] = df[col].astype(t)

    return df

def build_provenancegraphbuilding_agg_output_df():
    return build_dataframe(["Direct",
                            "MeanSimNLO",
                            "MeanSimNO",
                            "MeanSimLO",
                            "MeanNodeRecall"],
                           { "Direct": bool,
                             "MeanSimNLO": float,
                             "MeanSimNO": float,
                             "MeanSimLO": float,
                             "MeanNodeRecall": float })

def build_provenancegraphbuilding_nodemapping_df():
    return build_dataframe(["JournalName",
                            "ProvenanceProbeFileID",
                            "Direct",
                            "ProvenanceOutputFileName",
                            "WorldFileID",
                            "Mapping"],
                           { "JournalName": str,
                             "ProvenanceProbeFileID": str,
                             "Direct": bool,
                             "ProvenanceOutputFileName": str,
                             "WorldFileID": str,
                             "Mapping": str })

def build_provenancegraphbuilding_linkmapping_df():
    return build_dataframe(["JournalName",
                            "ProvenanceProbeFileID",
                            "Direct",
                            "ProvenanceOutputFileName",
                            "SourceWorldFileID",
                            "TargetWorldFileID",
                            "Mapping"],
                           { "JournalName": str,
                             "ProvenanceProbeFileID": str,
                             "Direct": bool,
                             "ProvenanceOutputFileName": str,
                             "SourceWorldFileID": str,
                             "TargetWorldFileID": str,
                             "Mapping": str })

def build_provenancegraphbuilding_output_df():
    return build_dataframe(["JournalName",
                            "ProvenanceProbeFileID",
                            "Direct",
                            "ProvenanceOutputFileName",
                            "NumSysNodes",
                            "NumSysLinks",
                            "NumRefNodes",
                            "NumRefLinks",
                            "NumCorrectNodes",
                            "NumMissingNodes",
                            "NumFalseAlarmNodes",
                            "NumCorrectLinks",
                            "NumMissingLinks",
                            "NumFalseAlarmLinks",
                            "SimNLO",
                            "SimNO",
                            "SimLO",
                            "NodeRecall"],
                           { "JournalName": str,
                             "ProvenanceProbeFileID": str,
                             "Direct": bool,
                             "ProvenanceOutputFileName": str,
                             "NumSysNodes": int,
                             "NumSysLinks": int,
                             "NumRefNodes": int,
                             "NumRefLinks": int,
                             "NumCorrectNodes": int,
                             "NumMissingNodes": int,
                             "NumFalseAlarmNodes": int,
                             "NumCorrectLinks": int,
                             "NumMissingLinks": int,
                             "NumFalseAlarmLinks": int,
                             "SimNLO": float,
                             "SimNO": float,
                             "SimLO": float,
                             "NodeRecall": float })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Score Medifor ProvenanceGraphBuilding task output")
    parser.add_argument("-d", "--direct", help="toggle direct path scoring", action="store_true")
    parser.add_argument("-t", "--skip-trial-disparity-check", help="Skip check for trial disparity between INDEX_FILE and SYSTEM_OUTPUT_FILE", action="store_true")
    parser.add_argument("-o", "--output-dir", help="Output directory for scores", type=str, required=True)
    parser.add_argument("-x", "--index-file", help="Task Index file", type=str, required=True)
    parser.add_argument("-r", "--reference-file", help="Reference file", type=str, required=True)
    parser.add_argument("-n", "--node-file", help="Node file", type=str, required=True)
    parser.add_argument("-w", "--world-file", help="World file", type=str, required=True)
    parser.add_argument("-R", "--reference-dir", help="Reference directory", type=str, required=True)
    parser.add_argument("-s", "--system-output-file", help="System output file (i.e. <EXPID>.csv)", type=str, required=True)
    parser.add_argument("-S", "--system-dir", help="System output directory where system output json files can be found", type=str, required=True)
    parser.add_argument("-p", "--plot-scored", help="Toggles graphical output of scored provenance graphs", action="store_true")
    parser.add_argument("-H", "--html-report", help="Generate an HTML report of the scores with plots (forces -p)", action="store_true")
    parser.add_argument("-T", "--thumbnail-cache-dir", help="Directory to use as thumbnail cache", type=str)
    args = parser.parse_args()

    mkdir_p(args.output_dir)
    figure_dir = os.path.join(os.path.abspath(args.output_dir), "figures")
    if args.html_report == True:
        args.plot_scored = True        
    # Import GraphVisualizer only if needed as it requires additional
    # dependencies
    if args.plot_scored:
        from GraphVisualizer import *
        mkdir_p(figure_dir)
    
    trial_index = load_csv(args.index_file)
    ref_file = load_csv(args.reference_file)
    nodes_file = load_csv(args.node_file)
    world_index = load_csv(args.world_file)

    abs_reference_dir = os.path.abspath(args.reference_dir)
    
    abs_thumb_cache_dir = None
    if args.thumbnail_cache_dir is not None:
        abs_thumb_cache_dir = os.path.abspath(args.thumbnail_cache_dir)
    
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

    output_records = build_provenancegraphbuilding_output_df()
    output_node_mapping_records = build_provenancegraphbuilding_nodemapping_df()
    output_link_mapping_records = build_provenancegraphbuilding_linkmapping_df()
    
    world_node_lookup = {}
    for world_node in world_nodes.itertuples():
        world_node_lookup[world_node.JournalNodeID] = world_node.WorldFileName_x
    
    for journal_fn, trial_index_ref_sysout_items in trial_index_ref_sysout.groupby("JournalFileName"):
        journal_path = os.path.join(abs_reference_dir, journal_fn)
        journal = load_json(journal_path)        
        
        node_lookup = {}
        node_index_by_id = {}
        for i, node in enumerate(journal["nodes"]):
            node_lookup[i] = node
            node_index_by_id[node["id"]] = i

        edge_lookup = {}
        original_edges = journal["links"]
        edge_filter_result = reject_edges(original_edges, [antiforensic_donor_filter])
        for tup in zip(edge_filter_result, enumerate(original_edges)):
            reject, (i, edge) = tup
            if reject:
                continue
            else:
                edge_lookup[i] = edge

        edge_records = [ EdgeRecord(v["source"], v["target"], Path(k, None)) for k, v in edge_lookup.items() ]
        # Check journal for cycles
        if detect_cycle(edge_records):
            err_quit("Detected a cycle in journal file.  Aborting!")

        for trial in trial_index_ref_sysout_items.itertuples():
            system_out = load_json(os.path.join(args.system_dir, trial.ProvenanceOutputFileName))
            
            probe_node_id = nodes_file[nodes_file.WorldFileID == trial.ProvenanceProbeFileID].iloc[0]["JournalNodeID"]
            probe_node_wfn = trial.ProvenanceProbeFileName_x
            world_set_nodes = { node_index_by_id[x] for x in world_nodes[world_nodes.ProvenanceProbeFileID == trial.ProvenanceProbeFileID]["JournalNodeID"] }
            world_set_nodes.add(node_index_by_id[probe_node_id])

            ref_graph = reduce_graph(edge_records, world_set_nodes)
            if args.direct:
                ref_graph = build_direct_graph(ref_graph, node_index_by_id[probe_node_id])

            def lookup_wfn(fileid):
                if fileid == probe_node_id:
                    return probe_node_wfn
                else:
                    return world_node_lookup[fileid]

            ref_nodes = ({ lookup_wfn(node_lookup[s]["id"]) for s, t, p in ref_graph } |
                         { lookup_wfn(node_lookup[t]["id"]) for s, t, p in ref_graph })
            ref_edges = { (lookup_wfn(node_lookup[s]["id"]), lookup_wfn(node_lookup[t]["id"])) for s, t, p in ref_graph }

            sys_nodes, sys_edges = system_out_to_scorable(system_out)

            # Could maybe put this in a "mapping" function
            correct_nodes = sys_nodes & ref_nodes
            fa_nodes = sys_nodes - ref_nodes
            missing_nodes = ref_nodes - sys_nodes
            correct_edges = sys_edges & ref_edges
            fa_edges = sys_edges - ref_edges
            missing_edges = ref_edges - sys_edges

            def _worldfile_path_to_id(path):
                base, ext = os.path.splitext(os.path.basename(path))
                return base

            def _build_node_map_record(node, mapping):
                return { "JournalName": trial.JournalName,
                         "ProvenanceProbeFileID": trial.ProvenanceProbeFileID,
                         "Direct": args.direct,
                         "ProvenanceOutputFileName": trial.ProvenanceOutputFileName,
                         "WorldFileID": _worldfile_path_to_id(node),
                         "Mapping": mapping }

            def _build_link_map_record(link, mapping):
                return { "JournalName": trial.JournalName,
                         "ProvenanceProbeFileID": trial.ProvenanceProbeFileID,
                         "Direct": args.direct,
                         "ProvenanceOutputFileName": trial.ProvenanceOutputFileName,
                         "SourceWorldFileID": _worldfile_path_to_id(link[0]),
                         "TargetWorldFileID": _worldfile_path_to_id(link[1]),
                         "Mapping": mapping }

            for map_record in sorted([ _build_node_map_record(node, "Correct") for node in correct_nodes ] +
                                     [ _build_node_map_record(node, "FalseAlarm") for node in fa_nodes ] +
                                     [ _build_node_map_record(node, "Missing") for node in missing_nodes ]):
                output_node_mapping_records = output_node_mapping_records.append(pd.Series(map_record), ignore_index=True)

            for map_record in sorted([ _build_link_map_record(link, "Correct") for link in correct_edges ] +
                                     [ _build_link_map_record(link, "FalseAlarm") for link in fa_edges ] +
                                     [ _build_link_map_record(link, "Missing") for link in missing_edges ]):
                output_link_mapping_records = output_link_mapping_records.append(pd.Series(map_record), ignore_index=True)

            
            out_rec = { "JournalName": trial.JournalName,
                        "ProvenanceProbeFileID": trial.ProvenanceProbeFileID,
                        "Direct": args.direct,
                        "ProvenanceOutputFileName": trial.ProvenanceOutputFileName,
                        "NumSysNodes": len(sys_nodes),
                        "NumSysLinks": len(sys_edges),
                        "NumRefNodes": len(ref_nodes),
                        "NumRefLinks": len(ref_edges),
                        "NumCorrectNodes": len(correct_nodes),
                        "NumMissingNodes": len(missing_nodes),
                        "NumFalseAlarmNodes": len(fa_nodes),
                        "NumCorrectLinks": len(correct_edges),
                        "NumMissingLinks": len(missing_edges),
                        "NumFalseAlarmLinks": len(fa_edges),
                        "SimNLO": SimNLO(ref_nodes, ref_edges, sys_nodes, sys_edges),
                        "SimNO": SimNO(ref_nodes, sys_nodes),
                        "SimLO": SimLO(ref_edges, sys_edges),
                        "NodeRecall": node_recall(ref_nodes, sys_nodes) }
            
            output_records = output_records.append(pd.Series(out_rec), ignore_index=True)

            # Plot our scored graph if requested
            if args.plot_scored:
                out_fn = os.path.join(figure_dir, "{}.png".format(trial.ProvenanceProbeFileID))
                render_provenance_graph_from_mapping(trial.ProvenanceProbeFileID,
                                                     correct_nodes,
                                                     fa_nodes,
                                                     missing_nodes,
                                                     correct_edges,
                                                     fa_edges,
                                                     missing_edges,
                                                     out_fn,
                                                     abs_reference_dir,
                                                     abs_thumb_cache_dir)

    output_agg_records = build_provenancegraphbuilding_agg_output_df()

    aggregated = { "Direct": args.direct,
                   "MeanSimNLO": output_records["SimNLO"].mean(),
                   "MeanSimNO": output_records["SimNO"].mean(),
                   "MeanSimLO": output_records["SimLO"].mean(),
                   "MeanNodeRecall": output_records["NodeRecall"].mean() }
    output_agg_records = output_agg_records.append(pd.Series(aggregated), ignore_index=True)
    
    def _write_df_to_csv(df, out_fn):
        try:
            with open(os.path.join(args.output_dir, out_fn), 'w') as out_f:
                df.to_csv(path_or_buf=out_f, sep="|", index=False)
        except IOError as ioerr:
            err_quit("{}. Aborting!".format(ioerr))

    _write_df_to_csv(output_records, "trial_scores.csv")
    _write_df_to_csv(output_agg_records, "scores.csv")
    _write_df_to_csv(output_node_mapping_records, "node_mapping.csv")
    _write_df_to_csv(output_link_mapping_records, "link_mapping.csv")

    if args.html_report == True:
        try:
            pd.set_option('display.max_colwidth', -1) # Keep pandas from truncating our links
            with open(os.path.join(args.output_dir, "report.html"), 'w') as out_f:
                out_f.write("<h2>Aggregated Scores:</h2>")
                output_agg_records.to_html(buf=out_f, index=False)
                out_f.write("<br/><br/>")
                out_f.write("<h2>Trial Scores:</h2>")
                output_records["Figure"] = output_records["ProvenanceProbeFileID"].map(lambda x: "<a href=\"figures/{0}.png\">link</a>".format(x))
                output_records.to_html(buf=out_f, index=False, escape=False)
        except IOError as ioerr:
            err_quit("{}. Aborting!".format(ioerr))
