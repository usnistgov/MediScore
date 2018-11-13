#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:49:17 2017

@author: tnk12
"""
import numpy as np
from intervalcompute import IntervalCompute as IC 
from itertools import groupby
from collections import defaultdict


class VideoScoring():
    
    def __init__(self,collars=None,max_range=100):
        
        self.red = (1,0,0)
        self.green = (128/255,1,0)
        self.blue = (0,0,1)
        self.yellow = (1,1,0)
        self.red_hex = "#FF0000"
        self.green_hex = "#80FF00"
        self.blue_hex = "#0000FF"
        self.yellow_hex = "#FFFF00"
        self.white_hex = "#FFFFFF"
        self.black_hex = "#000000"
        self.max_range = max_range
        
        self.confusion_mapping = {0 : ["TN", self.white_hex], 
                                  1 : ["FN",self.blue_hex], 
                                  2 : ["FP", self.red_hex], 
                                  3 : ["TP", self.green_hex], 
                                 -1 : ["X", self.yellow_hex],
                                 -2 : ["Y", self.black_hex]}
        
        self.collars = None
        self.ref_intervals = None
        self.sys_intervals = None
        
    
    
    def compute_confusion_map(self, ref_intervals, sys_intervals, global_interval, collars=None, SNS=None,verbose=False):
        """This function is a wrapper to aggreate_intervals, in order to perform a confusion measure
        between an intervals list of reference and the interval list from a system output.
        :param ref_intervals: numpy.array
        :param sys_intervals: numpy.array
        :param global_interval: the global range where the measure is performed
        :param collars: an interval array representing the collars
        :param SNS: an interval array representing a system no_score zone
        """
        
        
        # Creation of the intervals_sequence
        interval_param_list = [ref_intervals, sys_intervals, collars, SNS]
        intervals_sequence = [IC.compute_intervals_union([i]) for i in interval_param_list if i is not None]
        
        # We remove all None and 0-length sub-intervals
        tmp_intervals_sequence = []
        for i in intervals_sequence:
            if i.size != 0:
                a = i[i[:,0] != i[:,1]]
                if a.size != 0:
                    tmp_intervals_sequence.append(a)
        intervals_sequence = tmp_intervals_sequence

        # Compute the overlap between all intervals sets
        confusion_vector, all_intervals, all_interval_in_seq_array, weights = IC.aggregate_intervals(intervals_sequence, 
                                                                                                  global_interval, 
                                                                                                  print_results=verbose)
        
        if collars is not None or SNS is not None: 
 
            # To put negative values in the confusion vector (np.uint64), we need to cast it in signed
            confusion_vector_masked = confusion_vector.astype(np.int64, copy=True)

            # Apply overrided value for no-scores zones
            if collars is not None:
                collars_mask = all_interval_in_seq_array[2].astype(bool)
                confusion_vector_masked[collars_mask] = -1
                
            if SNS is not None:
                SNS_idx = 3 if collars is not None else 2
                SNS_mask = all_interval_in_seq_array[SNS_idx].astype(bool)
                confusion_vector_masked[SNS_mask] = -2

            # In the case of no-score zone, we compress the intervals containing consecutive and identical values
            confusion_vector_compressed, sizes_compression = zip(*[(k, len(list(g))) for k, g in groupby(confusion_vector_masked)])
            
            sizes_cs = np.cumsum(sizes_compression)
            start_indexes_compressions = sizes_cs - sizes_compression
            end_indexes_compression = sizes_cs - 1
            all_intervals_compressed = []
            for start, end in zip(start_indexes_compressions, end_indexes_compression):
                if start != end:
                    start_first, _ = all_intervals[start]
                    _ ,end_last = all_intervals[end]
                    all_intervals_compressed.append([start_first,end_last])
                else:
                    all_intervals_compressed.append(all_intervals[start])

            all_interval_in_seq_array_compressed = all_interval_in_seq_array[:,start_indexes_compressions]
            return (np.array(confusion_vector_compressed), confusion_vector), np.array(all_intervals_compressed), all_interval_in_seq_array_compressed

        return confusion_vector, all_intervals, all_interval_in_seq_array
    
    def count_confusion_value(self, all_intervals, confusion_vector, confusion_mapping=None, mapping = True):
        Counts = defaultdict(int)
        
        if confusion_mapping is None: 
            confusion_mapping = self.confusion_mapping            
        
        values = confusion_vector
        if mapping: 
            values = [confusion_mapping[x][0] for x in confusion_vector]
            
        for n, val in zip(np.diff(all_intervals), values):
            Counts[val] += n[0]
        return Counts
    
    @staticmethod
    def compute_MCC(TP, TN, FP, FN):
        numerator = ((TP*TN)-(FP*FN))
        denominator = np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        return numerator / denominator if denominator != 0 else 0
    
    
if __name__ == '__main__':
    # Reference and system ouput random generation
    max_range = 20
    # ref_seed, sys_seed = 42, 55
    # ref_intervals = gen_random_intervals(4, max_range, random_seed = ref_seed)
    # sys_intervals = gen_random_intervals(5, max_range, random_seed = sys_seed)
    # ref_intervals = gen_random_intervals(4, max_range)
    # sys_intervals = gen_random_intervals(5, max_range)
    collar = 1
    # global_interval = [0,max_range+10]
    global_interval = [0,25]
    ref_intervals = np.array([[10,18]])
    sys_intervals = np.array([[5,12], [17, 20]])
    print("Ref : {}".format(ref_intervals.tolist()))
    print("Sys : {}".format(sys_intervals.tolist()))
    
    collars_intervals = IC.compute_collars(ref_intervals, collar, crop_to_range = global_interval)
    # print(collars_intervals)
    SNS = np.array([[4,9], [13,16]])
    
    
    Scorer = VideoScoring()
    confusion_vector, all_intervals, all_interval_in_seq_array = Scorer.compute_confusion_map(ref_intervals, 
                                                                                       sys_intervals, 
                                                                                       global_interval, 
                                                                                       collars=collars_intervals,
                                                                                       SNS=SNS)
    
    confusion_vector_mapped = [Scorer.confusion_mapping[x][0] for x in confusion_vector]
    confusion_intervals_mapped = list(zip(all_intervals.tolist(),confusion_vector_mapped))
    
    # print(confusion_vector)
    # print(confusion_vector_mapped)
    # print(confusion_intervals_mapped)
    CR = confusion_vector, all_intervals, Scorer.confusion_mapping
    
    figure_width = 15
    IC.display_confusion_scoring(IC.compute_intervals_union(ref_intervals),
                                  IC.compute_intervals_union(sys_intervals), 
                                  global_interval, 
                                  confusion_results=CR, 
                                  figsize=(figure_width,int(figure_width/4)),
                                  colors = ["#9933FF","#FF00FF"])
                                            
    Counts = Scorer.count_confusion_value(all_intervals, confusion_vector)
    MCC = Scorer.compute_MCC(*[Counts[v] for v in ["TP", "TN", "FP", "FN"]])
    print("MCC = {}".format(MCC))
