#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from pandas import DataFrame, Index
from datacontainer import DataContainer
from metrics import Metrics


class MediForDataContainer(DataContainer):
    """Class representing a set of trials and containing:
       - FNR array
       - FPR array
       - TPR array
       - EER metric
       - AUC metric
       - Confidence Interval for AUC
    """

    def __init__(self, fa_array, fn_array, threshold, label=None, line_options=None, fa_label=None, fn_label=None):
        super(MediForDataContainer, self).__init__(fa_array, fn_array, threshold, label=label, line_options=line_options, fa_label="PFA", fn_label="PMiss")
        self.t_num = None
        self.nt_num = None
        self.scores = None
        self.gt = None
        self.trr = None
        self.eer = None
        self.auc = None
        self.auc_at_fpr = None
        self.d = None
        self.dpoint = None
        self.b = None
        self.bpoint = None
        self.tpr_at_fpr = None
        self.ci_level = None
        self.auc_ci_lower = None
        self.auc_ci_upper = None
        self.auc_at_fpr_ci_lower = None
        self.auc_at_fpr_ci_upper = None
        self.tpr_at_fpr_ci_lower = None
        self.tpr_at_fpr_ci_upper = None
        self.fpr_stop = None
        self.sys_res = None

    def _attributes_set_print(self, attributes, verbose=True):
        if verbose:
            display_list = ["Attributes set:"]
            for attr in attributes:
                if isinstance(attr, (list, tuple)):
                    attr_name, attr_value = attr
                    display_list.append("  - '{}' -> {}".format(attr_name, attr_value))
                else:
                    display_list.append("  - '{}'".format(attr))
            print("\n".join(display_list))

    def set_groundtruth(self, groundtruth, target_label="Y", non_target_label="N", verbose=True):
        self.target_label = target_label
        self.non_target_label = non_target_label
        self.gt = groundtruth
        self._attributes_set_print(['gt', ("target_label", repr(self.target_label)), ("non_target_label", repr(self.non_target_label))], verbose=verbose)

    def set_scores(self, scores, verbose=True):
        self.scores = scores
        self._attributes_set_print(['scores'], verbose=verbose)

    def set_target_stats(self, verbose=True):
        if self.gt is not None:
            self.t_num = np.count_nonzero(self.gt == self.target_label)
            self.nt_num = np.count_nonzero(self.gt == self.non_target_label)
            self._attributes_set_print([('t_num', self.t_num), ('nt_num', self.nt_num)], verbose=verbose)
        else:
            print("Error [set_target_stats]: Missing attributes.\nPlease set the ground truth array by calling set_groundtruth() before calling this method.")
    
    def set_trial_reponse_rate(self, total_trial, verbose=True):
        if (self.t_num is None or self.nt_num is None):
            print("Error [set_trial_response_rate]: Missing attributes.\nPlease set the target statistics by calling set_target_stats() before calling this method.")
        else:
            if total_trial != 0:
                self.trr = round((self.t_num + self.nt_num) / total_trial, 2)
                self._attributes_set_print([('trr', self.trr)], verbose=verbose)
            else:
                print("Error: the total number of trial without optout must be different from 0")

    def set_eer(self, verbose=True):
        self.eer = Metrics.compute_eer(self.fa, self.fn)
        self._attributes_set_print([('eer', self.eer)], verbose=verbose)

    def set_auc(self, verbose=True):
        self.auc = Metrics.compute_auc(self.fa, 1 - self.fn, fpr_stop=1)
        self._attributes_set_print([('auc', self.auc)], verbose=verbose)

    def set_auc_at_fpr(self, fpr_stop=1, verbose=True):
        self.fpr_stop = fpr_stop
        self.auc_at_fpr = Metrics.compute_auc(self.fa, 1 - self.fn, fpr_stop=fpr_stop)
        self._attributes_set_print([('auc_at_fpr', self.auc_at_fpr), ('fpr_stop', self.fpr_stop)], verbose=verbose)

    def set_dprime(self, dLevel=0.0, verbose=True):
        self.d, self.dpoint, self.b, self.bpoint = Metrics.compute_dprime(self.fa, 1 - self.fn, d_level=dLevel)
        self._attributes_set_print([('d', self.d), ('dpoint', self.dpoint), ('b', self.b), ('bpoint', self.bpoint)], verbose=verbose)

    def set_sys_res(self, sys_res="all", verbose=True):
        self.sys_res = sys_res
        self._attributes_set_print([('sys_res', self.sys_res)], verbose=verbose)

    def set_tpr_at_fpr(self, fpr_stop=1, verbose=True):
        inter_point = Metrics.linear_interpolated_point(self.fa, 1 - self.fn, fpr_stop)
        self.tpr_at_fpr = inter_point[0][1]
        self._attributes_set_print([('tpr_at_fpr', self.tpr_at_fpr)], verbose=verbose)

    def set_ci(self, ciLevel=0.9, fpr_stop=1, verbose=True):
        if self.scores is not None:
            self.ci_level = ciLevel
            self.fpr_stop = fpr_stop
            self.auc_ci_lower, self.auc_ci_upper, self.auc_at_fpr_ci_lower, self.auc_at_fpr_ci_upper, self.tpr_at_fpr_ci_lower, self.tpr_at_fpr_ci_upper = Metrics.compute_ci(self.scores, self.gt, self.ci_level, self.fpr_stop, target_label=self.target_label)
            self._attributes_set_print([('ci_level', self.ci_level), ('fpr_stop', self.fpr_stop), 
                                        ('auc_ci_lower', self.auc_ci_lower), ('auc_ci_upper', self.auc_ci_upper), 
                                        ('auc_at_fpr_ci_lower', self.auc_at_fpr_ci_lower), ('auc_at_fpr_ci_upper', self.auc_at_fpr_ci_upper), 
                                        ('tpr_at_fpr_ci_lower', self.tpr_at_fpr_ci_lower), ('tpr_at_fpr_ci_upper', self.tpr_at_fpr_ci_upper)],
                                        verbose=verbose)
        else:
            print("Error [set_ci]: Missing attributes.\nPlease set the scores array by calling set_scores() before calling this method.")

    def setter_standard(self, gt, scores, total_trial, target_label="Y", non_target_label="N", verbose=True):
        self.set_groundtruth(gt, target_label=target_label, non_target_label=non_target_label, verbose=verbose)
        self.set_scores(scores, verbose=verbose)
        self.set_target_stats(verbose=verbose)
        self.set_trial_reponse_rate(total_trial, verbose=verbose)
        self.set_eer(verbose=verbose)
        self.set_auc(verbose=verbose)
        self.set_auc_at_fpr(verbose=verbose)
        self.set_dprime(verbose=verbose)
        self.set_sys_res(verbose=verbose)
        self.set_tpr_at_fpr(verbose=verbose)
        self.set_ci(verbose=verbose)

    def arrays_to_dataframe(self, fa_label=None, fn_label=None):
        fa_col_name = fa_label if fa_label is not None else (self.fa_label if self.fa_label is not None else "FPR")
        fn_col_name = fn_label if fn_label is not None else (self.fn_label if self.fn_label is not None else "FNR")
        table = DataFrame({fa_col_name:self.fa,
                           fn_col_name:self.fn,
                           "threshold":self.threshold})
        return table

    def metrics_to_dataframe(self, orientation="vertical"):
        """ Render CSV table using Pandas Data Frame
        """
        metric_labels = ['TRR', 'SYS_RESPONSE', 'AUC', 'EER', 'FAR_STOP', 'AUC@FAR', 'CDR@FAR', 'CI_LEVEL', 
                         'AUC_CI_LOWER', 'AUC_CI_UPPER', 'AUC_CI_LOWER@FAR', 'AUC_CI_UPPER@FAR', 'CDR_CI_LOWER@FAR', 'CDR_CI_UPPER@FAR']
        metrics = [self.trr, self.sys_res, self.auc, self.eer, self.fpr_stop, self.auc_at_fpr, self.tpr_at_fpr, self.ci_level,
                   self.auc_ci_lower, self.auc_ci_upper, self.auc_at_fpr_ci_lower, self.auc_at_fpr_ci_upper, self.tpr_at_fpr_ci_lower, self.tpr_at_fpr_ci_upper]
        
        if orientation.lower() == "vertical":
            index = Index(metric_labels, name="Metrics")
            table = DataFrame(metrics, index=index, columns=["Value"])
            return table.round(6)
        elif orientation.lower() == "horizontal":
            columns = Index(metric_labels, name="Metrics")
            table = DataFrame([metrics], index=["Value"], columns=columns)
            return table.round(6)
        else:
            print("Error: Unknown orientation '{}', please choose from 'vertical', 'horizontal'".format(orientation))

        

    @staticmethod
    def set_from_old_dm(dm):
        dc = MediForDataContainer(dm.fpr, dm.fnr, dm.thres, label=None, line_options=None, fa_label="PFA", fn_label="Pmiss")
        used_attributes_names = set(["fpr", "fnr", "thres"])
        attributes_list = set(dm.__dict__.keys()) - used_attributes_names
        for attribute_name in attributes_list:
            setattr(dc, attribute_name, getattr(dm, attribute_name))
        return dc