#!/usr/bin/env python2

import sys
import os
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib")
sys.path.append(lib_path)

import json
import argparse
from pandas import DataFrame, read_csv, merge, set_option
import errno

from ProvenanceGraphBuilding import *
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

def antiforensic_donor_filter(edge_list):
    ins = group_by_fun(lambda e: e["target"], edge_list)

    def _filter(edge):
        return edge["op"] == "Donor" and len(filter(lambda e: e["op"][0:12] == "AntiForensic", ins.get(edge["target"], []))) > 0

    return map(_filter, edge_list)

def system_out_to_scorable(system_out):
    nl = { i: node for i, node in enumerate(system_out["nodes"]) }

    sys_node_dict = { n["file"]: n for n in system_out["nodes"] }
    sys_edge_dict = { (nl[e["source"]]["file"], nl[e["target"]]["file"]): (nl[e["source"]], nl[e["target"]], e) for e in system_out["links"] }

    return (sys_node_dict, sys_edge_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Score Medifor ProvenanceGraphBuilding task output")
    parser.add_argument("-d", "--direct", help="toggle direct path scoring", action="store_true")
    parser.add_argument("-t", "--skip-trial-disparity-check", help="Skip check for trial disparity between INDEX_FILE and SYSTEM_OUTPUT_FILE", action="store_true")
    parser.add_argument("-c", "--warn-on-system-cycle", help="Produces warnings if cycles detected in system output, rather than aborting", action="store_true")
    parser.add_argument("-o", "--output-dir", help="Output directory for scores", type=str, required=True)
    parser.add_argument("-x", "--index-file", help="Task Index file", type=str, required=True)
    parser.add_argument("-r", "--reference-file", help="Reference file", type=str, required=True)
    parser.add_argument("-n", "--node-file", help="Node file", type=str, required=True)
    parser.add_argument("-w", "--world-file", help="World file", type=str, required=True)
    parser.add_argument("-R", "--reference-dir", help="Reference directory", type=str, required=True)
    parser.add_argument("-s", "--system-output-file", help="System output file (i.e. <EXPID>.csv)", type=str, required=True)
    parser.add_argument("-u", "--undirected-graph", help="Toggles undirect graph support", action="store_true")
    parser.add_argument("-S", "--system-dir", help="System output directory where system output json files can be found", type=str, required=True)
    parser.add_argument("-p", "--plot-scored", help="Toggles graphical output of scored provenance graphs", action="store_true")
    parser.add_argument("-H", "--html-report", help="Generate an HTML report of the scores with plots (forces -p)", action="store_true")
    parser.add_argument("-T", "--thumbnail-cache-dir", help="Directory to use as thumbnail cache", type=str)
    parser.add_argument("-v", "--verbose", help="Toggle verbose log output", action="store_true")
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

    # Logger setup, could eventually support different levels of
    # verbosity
    verbosity_threshold = 1 if args.verbose else 0
    log = build_logger(verbosity_threshold)

    trial_index = load_csv(args.index_file)
    ref_file = load_csv(args.reference_file)
    nodes_file = load_csv(args.node_file)
    world_index = load_csv(args.world_file)

    abs_reference_dir = os.path.abspath(args.reference_dir)

    abs_thumb_cache_dir = None
    if args.thumbnail_cache_dir is not None:
        abs_thumb_cache_dir = os.path.abspath(args.thumbnail_cache_dir)

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

    trial_index_ref = merge(trial_index, ref_file, on = "ProvenanceProbeFileID")
    trial_index_ref_sysout = merge(trial_index_ref, system_output_index, on = "ProvenanceProbeFileID")

    world_nodes = merge(nodes_file, world_index, on = "WorldFileID", how = "inner")

    output_records = []
    output_node_mapping_records = []
    output_link_mapping_records = []

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
            log(1, "Working on ProvenanceProbeFileID: '{}' in JournalName: '{}'".format(trial.ProvenanceProbeFileID, trial.JournalName))
            system_out_path = os.path.join(args.system_dir, trial.ProvenanceOutputFileName)
            system_out = load_json(system_out_path)

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

            ref_nodes_dict = {}
            ref_edges_dict = {}
            for s, t, p in ref_graph:
                s_node = node_lookup[s]
                t_node = node_lookup[t]
                s_node_wfn = lookup_wfn(s_node["id"])
                t_node_wfn = lookup_wfn(t_node["id"])

                ref_nodes_dict[s_node_wfn] = s_node
                ref_nodes_dict[t_node_wfn] = t_node
                ref_edges_dict[(s_node_wfn, t_node_wfn)] = (s_node, t_node)

            sys_nodes_dict, sys_edges_dict = system_out_to_scorable(system_out)
            sys_edge_records = [ EdgeRecord(e[0], e[1], Path(e, None)) for e in sys_edges_dict.keys()]
            if detect_cycle(sys_edge_records):
                if args.warn_on_system_cycle:
                    log(1, "Warning, detected cycle(s) for system output file '{}'".format(system_out_path))
                else:
                    err_quit("Detected cycle(s) for system output file '{}', Aborting!".format(system_out_path))

            def _build_mapping(ref_nodes, ref_edges, sys_nodes, sys_edges, undirect_flag):
                node_mapping = [ (nk, ref_nodes.get(nk, None), sys_nodes.get(nk, None)) for nk in set(ref_nodes) | set(sys_nodes) ]
                if not undirect_flag:
                    edge_mapping = [ (ek, ref_edges.get(ek, None), sys_edges.get(ek, None)) for ek in set(ref_edges) | set(sys_edges) ]
                else:
                    edge_mapping = []
                    ref_set, sys_set = set(ref_edges), set(sys_edges)
                    for (a,b) in ref_set:
                       if (a,b) in sys_set:
                           sys_set.remove((a,b))
                           edge_mapping.append(((a,b), ref_edges[(a,b)], sys_edges[(a,b)]))
                       elif (b,a) in sys_set:
                           sys_set.remove((b,a))
                           edge_mapping.append(((a,b), ref_edges[(a,b)], sys_edges[(b,a)]))
                       else:
                           edge_mapping.append(((a,b), ref_edges[(a,b)], None))
                           
                    for edge in sys_set:
                       edge_mapping.append((edge, None, sys_edges[edge]))

                # a *_mapping file is a collection of (ref_*, sys_*)
                # tuples, where ref_* is None in the case of a FA and
                # sys_* is None in the case of a miss
                return (node_mapping, edge_mapping)

            node_mapping, edge_mapping = _build_mapping(ref_nodes_dict, ref_edges_dict, sys_nodes_dict, sys_edges_dict, args.undirected_graph)

            def _worldfile_path_to_id(path):
                base, ext = os.path.splitext(os.path.basename(path))
                return base

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

            def _build_node_map_record(node_key, ref_node, sys_node):
                return { "JournalName": trial.JournalName,
                         "ProvenanceProbeFileID": trial.ProvenanceProbeFileID,
                         "Direct": args.direct,
                         "ProvenanceOutputFileName": trial.ProvenanceOutputFileName,
                         "WorldFileID": _worldfile_path_to_id(node_key),
                         "NodeConfidence": sys_node["nodeConfidenceScore"] if sys_node != None else None,
                         "Mapping": _get_mapping(ref_node, sys_node) }

            def _build_link_map_record(link_key, ref_link, sys_link):
                link_key_s, link_key_t = link_key
                if sys_link != None:
                    sys_s_node, sys_t_node, sys_link_record = sys_link
                return { "JournalName": trial.JournalName,
                         "ProvenanceProbeFileID": trial.ProvenanceProbeFileID,
                         "Direct": args.direct,
                         "ProvenanceOutputFileName": trial.ProvenanceOutputFileName,
                         "SourceWorldFileID": _worldfile_path_to_id(link_key_s),
                         "TargetWorldFileID": _worldfile_path_to_id(link_key_t),
                         "LinkConfidence": sys_link_record["relationshipConfidenceScore"] if sys_link != None else None,
                         "Mapping": _get_mapping(ref_link, sys_link) }

            output_node_mapping_records += sorted([ _build_node_map_record(*node_map) for node_map in node_mapping ])
            output_link_mapping_records += sorted([ _build_link_map_record(*edge_map) for edge_map in edge_mapping ])

            def _corr_selector(t):
                k, r, s = t
                return (r != None and s != None)

            def _fa_selector(t):
                k, r, s = t
                return (r == None and s != None)

            def _miss_selector(t):
                k, r, s = t
                return (r != None and s == None)

            def _mapping_breakdown(node_mapping, edge_mapping):
                return ({ k for k, r, s in filter(_corr_selector, node_mapping) },
                        { k for k, r, s in filter(_miss_selector, node_mapping) },
                        { k for k, r, s in filter(_fa_selector, node_mapping) },
                        { k for k, r, s in filter(_corr_selector, edge_mapping) },
                        { k for k, r, s in filter(_miss_selector, edge_mapping) },
                        { k for k, r, s in filter(_fa_selector, edge_mapping) })

            sys_nodes = set(sys_nodes_dict.keys())
            sys_edges = set(sys_edges_dict.keys())
            ref_nodes = set(ref_nodes_dict.keys())
            ref_edges = set(ref_edges_dict.keys())
            correct_nodes, missing_nodes, fa_nodes, correct_edges, missing_edges, fa_edges = _mapping_breakdown(node_mapping, edge_mapping)

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

            output_records.append(out_rec)

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
                log(1, "Mapping figure saved to '{}'".format(out_fn))

    output_node_mapping_records_df = DataFrame(output_node_mapping_records, columns = ["JournalName",
                                                                                       "ProvenanceProbeFileID",
                                                                                       "Direct",
                                                                                       "ProvenanceOutputFileName",
                                                                                       "WorldFileID",
                                                                                       "NodeConfidence",
                                                                                       "Mapping"])
    output_link_mapping_records_df = DataFrame(output_link_mapping_records, columns = ["JournalName",
                                                                                       "ProvenanceProbeFileID",
                                                                                       "Direct",
                                                                                       "ProvenanceOutputFileName",
                                                                                       "SourceWorldFileID",
                                                                                       "TargetWorldFileID",
                                                                                       "LinkConfidence",
                                                                                       "Mapping"])
    output_records_df = DataFrame(output_records, columns = ["JournalName",
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
                                                             "NodeRecall"])

    aggregated = [{ "Direct": args.direct,
                    "MeanSimNLO": output_records_df["SimNLO"].mean(),
                    "MeanSimNO": output_records_df["SimNO"].mean(),
                    "MeanSimLO": output_records_df["SimLO"].mean(),
                    "MeanNodeRecall": output_records_df["NodeRecall"].mean() }]
    output_agg_records_df = DataFrame(aggregated, columns = ["Direct",
                                                             "MeanSimNLO",
                                                             "MeanSimNO",
                                                             "MeanSimLO",
                                                             "MeanNodeRecall"])

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
    _write_df_to_csv("Node Mapping", output_node_mapping_records_df, "node_mapping.csv")
    _write_df_to_csv("Link Mapping", output_link_mapping_records_df, "link_mapping.csv")

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
                output_records_df["Figure"] = output_records_df["ProvenanceProbeFileID"].map(lambda x: "<a href=\"figures/{0}.png\">link</a>".format(x))
                output_records_df.to_html(buf=out_f, index=False, escape=False, columns=["JournalName",
                                                                                         "ProvenanceProbeFileID",
                                                                                         "Direct",
                                                                                         "ProvenanceOutputFileName",
                                                                                         "Figure",
                                                                                         "SimNLO",
                                                                                         "SimNO",
                                                                                         "SimLO",
                                                                                         "NodeRecall",
                                                                                         "NumSysNodes",
                                                                                         "NumSysLinks",
                                                                                         "NumRefNodes",
                                                                                         "NumRefLinks",
                                                                                         "NumCorrectNodes",
                                                                                         "NumMissingNodes",
                                                                                         "NumFalseAlarmNodes",
                                                                                         "NumCorrectLinks",
                                                                                         "NumMissingLinks",
                                                                                         "NumFalseAlarmLinks"])
        except IOError as ioerr:
            err_quit("{}. Aborting!".format(ioerr))
