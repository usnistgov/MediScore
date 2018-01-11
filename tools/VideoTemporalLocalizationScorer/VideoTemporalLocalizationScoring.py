#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:27:22 2017

@author: tnk12
"""
import os
import argparse
import numpy as np
import pandas as pd
from intervalcompute import IntervalCompute as IC
from TemporalVideoScoring import VideoScoring
from ast import literal_eval

def interval_list_string_python_parsing(s):
    """Convert a string [[x0,x1], [x2,x3]] in a numpy array using the python parser engine
    """
    return np.array(literal_eval(s))

def interval_list_string_fast_parsing(s, datatype=float):
    """Convert a string [[x0,x1], [x2,x3]] in a numpy array
    :param datatype: datatype to cast the string number
    :Note: this function is about 10 times faster than the regular python parser engine
    Here is the benchmark using IPython %timeit magic function:
    def gen_long_interval_list_string(n):
        L = [[float(i),i+1] for i in range(0,n,2)]
        return "[{}]".format(", ".join([str(_) for _ in L]))
    >>> S = gen_long_interval_list_string(1000000)
    >>> %timeit np.array(literal_eval(S))
    >>> %timeit interval_list_string_fast_parsing(S,datatype=float)
    """
    return np.array([[datatype(x[0]),datatype(x[1])] for x in (y.split(',') for y in s.replace(" ", "")[2:-2].split('],['))])


class VTLScorer():
    def __init__(self, 
                 path_ref, 
                 path_index,
                 path_journalmask, 
                 path_probejournaljoin, 
                 global_interval=[0,999999999999], 
                 delimiter = "|", 
                 collars=None,
                 log=True,
                 path_log="./log.txt"):

        self.delimiter = "|"
        self.path_ref = path_ref
        self.path_index = path_index
        self.path_journalmask = path_journalmask
        self.path_probejournaljoin = path_probejournaljoin
        self.df_ref = pd.read_csv(path_ref, sep = delimiter)
        self.df_index = pd.read_csv(path_index, sep = delimiter)
        self.df_journalmask = pd.read_csv(path_journalmask, sep = delimiter)
        self.df_probejournaljoin = pd.read_csv(path_probejournaljoin, sep = delimiter)
        
        self.df_ref_probe_journal_merge = pd.merge(self.df_probejournaljoin,self.df_journalmask, on=["JournalName","StartNodeID","EndNodeID"])
        self.df_ref_probe_journal_index_merge = pd.merge(self.df_ref_probe_journal_merge, self.df_index, on=["ProbeFileID"])
        self.global_interval = global_interval
        self.collars = collars
        self.add_collars = collars is not None
        
        self.Scorer = VideoScoring()
        self.path_log = path_log

        if log:
            self.log_file = open(self.path_log, 'w')
            self.writelog = lambda line: self.log_file.write(line + "\n")
        else:   
            self.writelog = lambda *a: None
    
    def score_probes(self, path_system_output, query=None, compute_mean=False):
        Results = None

        df_system_output = pd.read_csv(path_system_output, sep = self.delimiter)

        if (query == "*") or (query == ["*"]) or (query is None):
            Results = self.compute_probes_MCC(self.df_ref_probe_journal_index_merge, df_system_output)
            Results['query'] = 0
            Results.set_index('query', append=True, inplace=True)
            Results.reorder_levels(['query', 'ProbeFileID'])
            Results = Results.reorder_levels(['query','ProbeFileID'])
        else:
            assert (isinstance(query, list)), "query should be a list ({})".format(query)

            queries_df_list = []
            for q in query:

                if q != "*":
                    df_probe_journal_merge_selection = self.df_ref_probe_journal_index_merge.query(q)
                else:
                    df_probe_journal_merge_selection = self.df_ref_probe_journal_index_merge

                queries_df_list.append(self.compute_probes_MCC(df_probe_journal_merge_selection, df_system_output))

            # Results = pd.concat(queries_df_list, keys=query, names=['query'])
            Results = pd.concat(queries_df_list, keys=np.arange(len(query)), names=['query'])

        return Results

    def compute_probes_MCC(self, df_ref, df_sys):
                    
        probes_selection = df_ref.ProbeFileID.drop_duplicates()
        probes_selection_scores_df = pd.DataFrame(np.zeros(len(probes_selection)), index=probes_selection, columns=["MCC"])

        for ProbeFileID in probes_selection:
            self.writelog("ProbeFileID = {}".format(ProbeFileID))
            collars = None

            # We first get the system intervals
            assert (ProbeFileID in df_sys.ProbeFileID.values), "ProbeFileID ({}) is missing in the system output file".format(ProbeFileID)
            # System probe info
            sys_probe = df_sys.query("ProbeFileID == '{}'".format(ProbeFileID))

            if sys_probe.ProbeStatus.values == "Processed":
                SysVideoFramesSeries = sys_probe.VideoFrameSegments
                SysVideoFramesSeries_list = [interval_list_string_fast_parsing(x,datatype=int) for x in SysVideoFramesSeries.values if x != "[]"]
                if SysVideoFramesSeries_list:
                    sys_intervals = IC.compute_intervals_union(SysVideoFramesSeries_list)
                else:
                    sys_intervals = np.array([[]])
                self.writelog("sys_intervals = {}".format(sys_intervals))
                # sys_intervals = IC.gen_random_intervals(3, 610)

                # We get any OptOut video region
                SysVideoFramesOptOutSeries = sys_probe.VideoFrameOptOutSegments
                SysVideoFramesOptOutSeries_list = [interval_list_string_fast_parsing(x,datatype=int) for x in SysVideoFramesOptOutSeries.values if x != "[]"]
                if SysVideoFramesOptOutSeries_list:
                    SNS = IC.compute_intervals_union(SysVideoFramesOptOutSeries_list)
                else:
                    SNS = None
                self.writelog("SNS = {}".format(SNS))

            else:
                self.writelog("This Probe is OptOut")
                probes_selection_scores_df.drop(ProbeFileID, inplace = True)
                continue

            # We get the reference intervals
            RefVideoFramesSeries = df_ref.query("ProbeFileID == '{}'".format(ProbeFileID)).VideoFrame
            RefVideoFramesSeries_list = [interval_list_string_fast_parsing(x,datatype=int) for x in RefVideoFramesSeries.values if x != "[]"]

            if RefVideoFramesSeries_list:
                ref_intervals = IC.compute_intervals_union(RefVideoFramesSeries_list)
            else:
                ref_intervals = np.array([[]])
            self.writelog("ref_intervals = {}".format(ref_intervals))

            # We get the total number of frame
            assert (ProbeFileID in self.df_index.ProbeFileID.values), "ProbeFileID ({}) is missing in the index file ({})".format(ProbeFileID, self.path_index)
            FrameCount = self.df_index.query("ProbeFileID == '{}'".format(ProbeFileID)).FrameCount.values[0]
            global_range = [1, FrameCount]
            self.writelog("global_range = {}".format(global_range))

            if self.add_collars:
                collars = IC.compute_collars(ref_intervals, self.collars, crop_to_range = global_range)
            self.writelog("collars = {}".format(collars))
            
            # We compute the confusion metrics
            confusion_vector, all_intervals, all_interval_in_seq_array = self.Scorer.compute_confusion_map(ref_intervals, 
                                                                                                      sys_intervals, 
                                                                                                      global_range, 
                                                                                                      collars=collars,
                                                                                                      SNS=SNS)
            self.writelog("confusion_vector = {}".format(confusion_vector))
            self.writelog("all_intervals = {}".format(all_intervals))
            Counts = self.Scorer.count_confusion_value(all_intervals, confusion_vector)
            MCC = self.Scorer.compute_MCC(*[Counts[v] for v in ["TP", "TN", "FP", "FN"]])
            self.writelog("Counts = {}\nMCC = {}".format(Counts, MCC))
                        
            probes_selection_scores_df.loc[ProbeFileID] = MCC
            self.writelog("\n")
        return probes_selection_scores_df
    

if __name__ == '__main__':
    import sys
    # If the script is run from an IDE
    if len(sys.argv) == 1:
        
        class Parameters():
            def __init__(self):
                self.path_data = "/Users/tnk12/Documents/MEDIFOR/Scoring/Data/Reference/Ref2/"
                self.path_ref = self.path_data + "MFC18_Dev1-manipulation-video-ref.csv"
                self.path_index = self.path_data + "MFC18_Dev1-manipulation-video-index_2.csv"
                self.path_journalmask = self.path_data + "MFC18_Dev1-manipulation-video-ref-journalmask.csv"
                self.path_probejournaljoin = self.path_data + "MFC18_Dev1-manipulation-video-ref-probejournaljoin.csv"
                self.collars = 5
                self.query = ["*"]
                self.dump_dataframe = False
                self.dump_dataframe_file_name = "./df_scores_probes.pkl"
                self.output_path = "./scores/"
                self.log = "True"
                
        parameters = Parameters()
        
    else:
        parser = argparse.ArgumentParser(description='Video Temporal Localization Scoring System')
        parser.add_argument('-r', '--path_ref', help='path to the manipulation video reference csv file')
        parser.add_argument('-i', '--path_index', help='path to the manipulation video index csv file')
        parser.add_argument('-j', '--path_journalmask', help='path to the manipulation video ref journal mask csv file')
        parser.add_argument('-p', '--path_probejournaljoin', help='path to the manipulation video ref probe journal join csv file')
        parser.add_argument('-s', '--path_sysout', help='path to the system output file')
        parser.add_argument('-c', '--collars', help='collar value to add to each side of the reference intervals', default=None, type=int)
        parser.add_argument('-q', '--query', nargs="+", help="""give a sequence of criteria over the probe selection. Each query should be between double quotes. "*" select all""",default='*')
        parser.add_argument('-d', '--dump_dataframe', help='path to the file where the dataframe will be dumped',action='store_true')
        parser.add_argument('-o', '--output_path', help='path to the folder where the scores will be dumped',default="./scores_output/")
        parser.add_argument('-l', '--log', help='enable a log output', action='store_true')
        parser.add_argument('--dump_dataframe_file_name', help='name of the dumped dataframe', default='df_scores_probes.pkl')

        parameters = parser.parse_args()

    # print("Scorer initialialisation..")
    Scorer = VTLScorer(parameters.path_ref, 
                       parameters.path_index,
                       parameters.path_journalmask, 
                       parameters.path_probejournaljoin, 
                       collars = parameters.collars,
                       log = parameters.log)

    # print("Scoring probes...")
    Results = Scorer.score_probes(parameters.path_sysout, query=parameters.query)
    # print("query = {}".format(parameters.query))

    if not os.path.exists(parameters.output_path):
        os.makedirs(parameters.output_path)

    if parameters.dump_dataframe:
        Results.to_pickle(os.path.join(parameters.output_path, parameters.dump_dataframe_file_name))

    # Rounding the output to match 12 digits precision instead of the new 16 digits in numpy 1.14
    f_format = lambda x: round(x, 12)

    Results["MCC"] = Results["MCC"].apply(f_format)
    Results.to_csv(os.path.join(parameters.output_path, "scores_probes.csv"), 
                   sep="|")

    nb_of_query = len(parameters.query)
    #Computing the average per query
    Results_overall = pd.DataFrame(np.zeros(nb_of_query), columns=["MCC"])
    if nb_of_query > 1:
        for i,q in enumerate(parameters.query):
            Results_overall.loc[i] = Results.loc[i,"MCC"].mean()
    else:
        Results_overall.loc[0] = Results["MCC"].mean()

    Results_overall["MCC"] = Results_overall["MCC"].apply(f_format)
    Results_overall.to_csv(os.path.join(parameters.output_path, "scores.csv"), index_label=['query'], sep="|")

    # Query table join
    query_table_join = pd.DataFrame(np.zeros(nb_of_query), columns=["query"])
    query_table_join['query'] = parameters.query
    query_table_join.to_csv(os.path.join(parameters.output_path, "query_table_join.csv"), index_label = ['id'], sep="|") 

    print("Done.")
    
    