#!/usr/bin/env python2
import os
import json
import argparse
import errno
import numpy as np
import pandas as pd
# from arghelper import Args


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
        return pd.read_csv(csv_fn, sep)
    except IOError as ioerr:
        err_quit("{}. Aborting!".format(ioerr))


def write_df_to_csv(name, df, output_dir, out_fn):
    out_path = os.path.join(output_dir, out_fn)
    log(1, "Writing {} to '{}'".format(name, out_path))
    df.to_csv(out_path, sep="|", index=False, line_terminator='\n')


def get_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            err_quit("{}. Aborting!".format(exc))


def format_float(f):
    return round(f, 12)


def check_for_trial_disparity(trial_index, system_output_index, only_warn_on_extraneous=False):
    def probe_str(probe_set):
        return "\n".join(["\t{}".format(p) for p in probe_set])

    # detect missing trials and extra trials
    set_ref_probes = set(trial_index.ProvenanceProbeFileID)
    set_sys_probes = set(system_output_index.ProvenanceProbeFileID)
    missing_probes = set_ref_probes - set_sys_probes
    extraneous_sys_probes = set_sys_probes - set_ref_probes

    errs = []
    if missing_probes:
        errs.append("Error, missing the following ProvenanceProbeFileIDs from the system output:\n{}".format(probe_str(missing_probes)))
    if extraneous_sys_probes:
        if only_warn_on_extraneous:
            log(1, "Warning, found {} extraneous ProvenanceProbeFileIDs in the system output.".format(len(extraneous_sys_probes)))
        else:
            errs.append("Error, extraneous ProvenanceProbeFileIDs in the system output:\n{}".format(probe_str(extraneous_sys_probes)))
    if errs:
        errs.append("Aborting!")
        err_quit("\n".join(errs))


def create_data_dataframes(args, log):
    # TaskID, ProvenanceProbeFileID, ProvenanceProbeFileName, ProvenanceProbeWidth, ProvenanceProbeHeight
    trial_index = load_csv(args.index_file)
    # TaskID, ProvenanceProbeFileID, ProvenanceProbeFileName, BaseFileName, BaseBrowserFileName, JournalName, JournalFileName, JournalMD5
    ref_file = load_csv(args.reference_file)
    # ProvenanceProbeFileID, WorldFileID, WorldFileName, JournalNodeID
    nodes_file = load_csv(args.node_file)
    # TaskID, WorldFileID, WorldFileName, WorldWidth, WorldHeight
    world_index = load_csv(args.world_file)
    # ProvenanceProbeFileID, ProvenanceOutputFileName, ProvenanceProbeStatus
    system_output_index = load_csv(args.system_output_file)

    check_for_trial_disparity(trial_index, system_output_index, args.skip_trial_disparity_check)
    log(1, "Scoring {} trials ..".format(len(trial_index)))
    # Remove NonProcessed (or IsOptOut) trials
    if "IsOptOut" in system_output_index.columns:
        system_output_index = system_output_index.query("IsOptOut == ['Processed', 'N']")
    elif "ProvenanceProbeStatus" in system_output_index.columns:
        system_output_index = system_output_index.query("ProvenanceProbeStatus == ['Processed', 'N']")

    trial_index_ref = pd.merge(trial_index, ref_file, on="ProvenanceProbeFileID")
    # TaskID_x, ProvenanceProbeFileID, ProvenanceProbeFileName_x, ProvenanceProbeWidth, ProvenanceProbeHeight,
    # TaskID_y, ProvenanceProbeFileName_y, BaseFileName, BaseBrowserFileName, JournalName, JournalFileName, JournalMD5, ProvenanceOutputFileName, ProvenanceProbeStatus'
    trial_index_ref_sysout = pd.merge(trial_index_ref, system_output_index, on="ProvenanceProbeFileID")
    # ProvenanceProbeFileID, WorldFileID, WorldFileName, JournalNodeID, TaskID, WorldWidth, WorldHeight
    world_nodes = pd.merge(nodes_file, world_index, on=["WorldFileID", "WorldFileName"], how="inner")

    return trial_index_ref_sysout, nodes_file, world_nodes


def build_mapping(x, key_1="WorldFileName", key_2="file"):
    if x[key_1] == x[key_2]:
        return "Correct"
    elif pd.isna(x[key_1]):
        return "FalseAlarm"
    elif pd.isna(x[key_2]):
        return "Missing"
    elif pd.isna(x[key_1]) and pd.isna(x[key_2]):
        return "WARNING_INCORRECT_ROW"


def compute_trial_score_mapping(trial, ref_nodes_df, sys_nodes_df, recalls_number, nodetype=None, return_outer_list=False):
    ret = []  # return of the function
    node_mapping_outer_join_list = []
    output_node_mapping_cols = ["JournalName", "ProvenanceProbeFileID", "ProvenanceOutputFileName"]

    if nodetype:
        ref_nodes_df = ref_nodes_df.query("nodetype == '{}'".format(nodetype))
    else:
        trial_node_mapping_df_list = []

    NumSysNodes, NumRefNodes = sys_nodes_df.shape[0], ref_nodes_df.shape[0]

    info_dict = {"JournalName": trial.JournalName,
                 "ProvenanceProbeFileID": trial.ProvenanceProbeFileID,
                 "ProvenanceOutputFileName": trial.ProvenanceOutputFileName,
                 "NumSysNodes": NumSysNodes,
                 "NumRefNodes": NumRefNodes,
                 "NodeType": "All" if not nodetype else nodetype}

    for n in recalls_number:
        node_mapping_df = pd.merge(left=ref_nodes_df[["WorldFileName", "JournalNodeID", "nodetype"]],
                                   right=sys_nodes_df_sorted_by_confidence[["file", "nodeConfidenceScore"]][:n],
                                   how='outer',
                                   left_on="WorldFileName",
                                   right_on="file")
        node_mapping_outer_join_list.append(node_mapping_df)
        node_mapping_df["Mapping"] = node_mapping_df.apply(build_mapping, axis=1)

        if "WARNING_INCORRECT_ROW" in node_mapping_df["Mapping"]:
            err_quit("Error: compute_trial_score_mapping() -> the mapping dataframe columns contains invalid rows!")

        node_mapping_df_sorted = node_mapping_df.sort_values(by=["Mapping", "nodeConfidenceScore"], ascending=[True, False], inplace=False)

        Mapping_counts_at_n = node_mapping_df["Mapping"].value_counts().astype(int)

#        node_recall_at_n = np.float64(Mapping_counts_at_n.get("Correct",0)) / NumRefNodes

        info_dict.update({"NumCorrectNodesAt{}".format(n): Mapping_counts_at_n.get("Correct", 0),
                          "NumMissingNodesAt{}".format(n): Mapping_counts_at_n.get("Missing",0),
                          "NumFalseAlarmNodesAt{}".format(n): Mapping_counts_at_n.get("FalseAlarm", 0),
                          "NodeRecallAt{}".format(n): np.float64(Mapping_counts_at_n.get("Correct", 0)) / NumRefNodes})

        # We don't compute the full node mapping output if we have a specific nodetype
        if not nodetype:
            initial_data_output_node_mapping_df = [trial.JournalName, trial.ProvenanceProbeFileID, trial.ProvenanceOutputFileName]
            partial_output_node_mapping_df = pd.DataFrame([initial_data_output_node_mapping_df]*node_mapping_df.shape[0], columns=output_node_mapping_cols)
            partial_output_node_mapping_df["Measure"] = "NodeRecallAt{}".format(n)
            partial_output_node_mapping_df["WorldFileID"] = node_mapping_df_sorted.apply(lambda x: get_id(x["WorldFileName"]) if pd.notna(x["WorldFileName"]) else get_id(x["file"]), axis=1)
            partial_output_node_mapping_df[["NodeConfidence", "Mapping", "NodeType"]] = node_mapping_df_sorted[["nodeConfidenceScore", "Mapping", "nodetype"]]
            partial_output_node_mapping_df.sort_values(by=["Mapping", "NodeConfidence", "WorldFileID"], ascending=[True, True, True], inplace=True)
            trial_node_mapping_df_list.append(partial_output_node_mapping_df)

    if nodetype is not None:
        ret = [info_dict]
    else:
        ret = [info_dict, trial_node_mapping_df_list]

    if return_outer_list:
        ret.append(node_mapping_outer_join_list)

    return ret


def gen_output_files(args, output_scores, node_mapping_df_list, recalls_number, log, html_report=False):
    mkdir_p(args.output_dir)

    NodeRecallColumns = ["NodeRecallAt{}".format(n) for n in recalls_number]
    stats_columns = ["NumCorrectNodesAt{}", "NumMissingNodesAt{}", "NumFalseAlarmNodesAt{}"]
    output_scores_columns = ["JournalName", "ProvenanceProbeFileID", "ProvenanceOutputFileName", "NumSysNodes", "NumRefNodes", "NodeType"]
    output_scores_columns.extend([stat.format(recall) for recall in recalls_number for stat in stats_columns])
    output_scores_columns.extend(NodeRecallColumns)

    output_scores_df = pd.DataFrame(output_scores, columns=output_scores_columns)

    output_scores_df[NodeRecallColumns] = output_scores_df[NodeRecallColumns].applymap(format_float)
    write_df_to_csv("Scores Output", output_scores_df, args.output_dir, "trial_scores.csv")

    # We compute the mean of each node recall per node type
    output_scores_mean_df = output_scores_df[["NodeType"] + NodeRecallColumns].groupby(["NodeType"]).mean().reset_index()
    write_df_to_csv("Scores Output Mean", output_scores_mean_df, args.output_dir, "scores.csv")

    output_node_mapping_df = pd.concat(node_mapping_df_list, ignore_index=True)
    write_df_to_csv("New Nodes Mapping", output_node_mapping_df, args.output_dir, "node_mapping.csv")

    if html_report:
        pd.set_option('display.max_colwidth', 100000) # Keep pandas from truncating our links
        report_out_path = os.path.join(args.output_dir, "report.html")
        log(1, "Writing HTML Report to '{}'".format(report_out_path))

        with open(report_out_path, 'w') as out_f:
            out_f.write("<h2>Scores Mean:</h2>")
            output_scores_mean_df.to_html(buf=out_f, index=False)
            out_f.write("<br/><br/>")
            out_f.write("<h2>Trial Scores:</h2>")
            output_scores_df.to_html(buf=out_f, index=False, columns=output_scores_columns)

    return output_scores_df


if __name__ == '__main__':
    ide_debug = False
    all_nodetypes = ["donor", "base", "final", "interim"]
    all_nodes_flags = ["*", "all"]

    if not ide_debug:
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
        parser.add_argument("--nodetype", help="Specify a list of node types", nargs='+', choices=all_nodetypes+all_nodes_flags)
        args = parser.parse_args()
        args_dict = vars(args)
    # else:
    #     wd = "/Users/tnk12/Documents/MEDIFOR/Provenance_NodeType/"
    #     # wd = "C:\\Users\\tim-k\\Documents\\Dev\\Provenance_NodeType"
    #     testsuite_directory = os.path.join(wd, "provenanceScorerTests")
    #     output_dir = os.path.join(wd, "Score_Output")
    #     args = Args(wd, testsuite_directory, output_dir, "2")
    #     args_dict = vars(args)

    if args.nodetype and set(all_nodes_flags) & set(args.nodetype):
        args_dict["nodetype"] = all_nodetypes

    # Logger setup, could eventually support different levels of verbosity
    verbosity_threshold = 1 if args.verbose else 0
    log = build_logger(verbosity_threshold)

    recalls_number = [50, 100, 200, 300]
    output_scores = []
    output_mapping_records = []
    node_mapping_df_list = []
#    full_node_mapping_outer_join_list = []

    # Compute the dataframes we need based on the reference and system data
    trial_index_ref_sysout, nodes_file, world_nodes = create_data_dataframes(args, log)

    for journal_fn, trial_index_ref_sysout_items in trial_index_ref_sysout.groupby("JournalFileName"):

        journal_path = os.path.join(args.reference_dir, journal_fn)
        journal = load_json(journal_path)
        journal_df = pd.DataFrame.from_dict(journal["nodes"])

        for trial in trial_index_ref_sysout_items.itertuples():
            log(1, "Working on ProvenanceProbeFileID: '{}' in JournalName: '{}'".format(trial.ProvenanceProbeFileID, trial.JournalName))

            # Generating the reference data
            trial_world_nodes = world_nodes[world_nodes.ProvenanceProbeFileID == trial.ProvenanceProbeFileID]

            ref_nodes_df = pd.merge(left=trial_world_nodes[["WorldFileName", "JournalNodeID"]],
                                    right=journal_df[["file", "id", "nodetype"]],
                                    left_on="JournalNodeID", right_on="id")

            # This should only add a single node, the probe node,
            trial_probe_node = nodes_file[(nodes_file.ProvenanceProbeFileID == trial.ProvenanceProbeFileID) & (nodes_file.WorldFileName == trial.ProvenanceProbeFileName_x)]
            probe_row = pd.DataFrame({"WorldFileName": [trial_probe_node.WorldFileName.iloc[0]],
                                      "JournalNodeID": [trial_probe_node.ProvenanceProbeFileID.iloc[0]],
                                      "file": [trial_probe_node.WorldFileName.iloc[0]],
                                      "id": [trial_probe_node.ProvenanceProbeFileID.iloc[0]],
                                      "nodetype": ['probe']})
            ref_nodes_df = ref_nodes_df.append(probe_row, ignore_index=True)

            # Generating the system data
            system_out = load_json(os.path.join(args.system_dir, trial.ProvenanceOutputFileName))
            sys_nodes_df = pd.DataFrame(system_out["nodes"])
            sys_nodes_df_sorted_by_confidence = sys_nodes_df.sort_values(by="nodeConfidenceScore", ascending=False, inplace=False)

            # Computing the scores and mapping
            # Overall
            trial_info_dict, trial_node_mapping_df_list = compute_trial_score_mapping(trial, ref_nodes_df, sys_nodes_df, recalls_number)
            node_mapping_df_list.extend(trial_node_mapping_df_list)
            output_scores.append(trial_info_dict)
#            full_node_mapping_outer_join_list.append(node_mapping_outer_join_list)

            # By nodetype if specified
            if args.nodetype:
                # Check that the nodetypes are in the reference
                ref_unique_nodetypes = ref_nodes_df.nodetype.unique()
                assert set(args.nodetype) & set(ref_unique_nodetypes) == set(args.nodetype), \
                    "The reference does not have those nodetypes : {}".format(list(set(args.nodetype) - (set(args.nodetype) & set(ref_unique_nodetypes))))

                for nodetype in args.nodetype:
                    trial_info_dict = compute_trial_score_mapping(trial, ref_nodes_df, sys_nodes_df, recalls_number, nodetype=nodetype)
                    output_scores.extend(trial_info_dict)

    ret = gen_output_files(args, output_scores, node_mapping_df_list, recalls_number, log, html_report=args.html_report)
