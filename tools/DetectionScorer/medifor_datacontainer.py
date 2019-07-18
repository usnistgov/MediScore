#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from datacontainer import DataContainer


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

    def set_groundtruth(self, groundtruth):
        self.gt = groundtruth
        print("Attributes set: 'gt'")

    def set_scores(self, scores):
        self.scores = scores
        print("Attributes set: 'scores'")

    def set_target_stats(self, target_label="Y", non_target_label="N"):
        print("Computing the number of target and non-target based on the provided ground truth labels")
        if self.gt is not None:
            self.t_num = np.count_nonzero(gt == target_label)
            self.nt_num = np.count_nonzero(gt == non_target_label)
            print("Attributes set: 't_num', 'nt_num'")
        else:
            print("Error [set_target_stats]: Missing attributes.\nPlease set the ground truth array by calling set_groundtruth() before.")
    
    def set_trial_reponse_rate(self, total_trial):
        if (self.t_num is None or self.nt_num is None):
            print("Error [set_trial_response_rate]: Missing attributes.\nPlease set the target statistics by using the method set_target_stats() before calling this method.")
        else:
            if total_trial != 0:
                self.trr = round((self.t_num + self.nt_num) / total_trial, 2)
                print("Attributes set: 'trr")
            else:
                print("Error: the total number of trial without optout must be different from 0")

    def set_eer(self):
        self.eer = Metrics.compute_eer(self.fa, self.fn)
        print("Attributes set: 'eer'")

    def set_auc(self):
        self.auc = Metrics.compute_auc(self.fa, 1 - self.fn, fpr_stop=1)
        print("Attributes set: 'auc'")

    def set_auc_at_fpr(self, fpr_stop=1):
        self.fpr_stop = fpr_stop
        self.auc_at_fpr = Metrics.compute_auc(self.fa, 1 - self.fn, fpr_stop=fpr_stop)
        print("Attributes set: 'auc_at_fpr', 'fpr_stop'")

    def set_dprime(self, dLevel=0.0):
        self.d, self.dpoint, self.b, self.bpoint = Metrics.compute_dprime(self.fa, self.tpr, d_level=dLevel)
        print("Attributes set: 'd', 'dpoint', 'b', 'bpoint'")

    def set_sys_res(self, sys_res):
        self.sys_res = sys_res
        print("Attributs set: 'sys_res'")

    def set_ci(self, ciLevel=0.9, fpr_stop=1):
        if self.scores is not None:
            self.ci_level = ciLevel
            self.fpr_stop = fpr_stop
            self.auc_ci_lower, self.auc_ci_upper, self.auc_at_fpr_ci_lower, self.auc_at_fpr_ci_upper, self.tpr_at_fpr_ci_lower, self.tpr_at_fpr_ci_upper = Metrics.compute_ci(score, gt, ciLevel, fpr_stop)
            print("Attributes set: 'ci_level', 'fpr_stop', 'auc_ci_lower', 'auc_ci_upper', 'auc_at_fpr_ci_lower', 'auc_at_fpr_ci_upper', 'tpr_at_fpr_ci_lower', 'tpr_at_fpr_ci_upper'")
        else:
            print("Error [set_ci]: Missing attributes.\nPlease set the scores array by calling the method set_scores() before.")

    def render_table(self):
        """ Render CSV table using Pandas Data Frame
        """
        from collections import OrderedDict
        from pandas import DataFrame
        data = OrderedDict([
        ('TRR', self.trr),
        ('SYS_RESPONSE', self.sys_res),
        ('AUC', self.auc),
        ('EER', self.eer),
        ('FAR_STOP', self.fpr_stop),
        ('AUC@FAR', self.auc_at_fpr),
        ('CDR@FAR', self.tpr_at_fpr),
        ('CI_LEVEL', self.ci_level),
        ('AUC_CI_LOWER', self.auc_ci_lower),
        ('AUC_CI_UPPER', self.auc_ci_upper),
        ('AUC_CI_LOWER@FAR', self.auc_at_fpr_ci_lower),
        ('AUC_CI_UPPER@FAR', self.auc_at_fpr_ci_upper),
        ('CDR_CI_LOWER@FAR', self.tpr_at_fpr_ci_lower),
        ('CDR_CI_UPPER@FAR', self.tpr_at_fpr_ci_upper)
        ])

        my_table = DataFrame(data, index=['0'])

        return my_table.round(6)
        #my_table.to_csv(file_name, index = False)