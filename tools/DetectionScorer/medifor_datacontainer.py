#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from data_container import DataContainer


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
        self.ci_level = ciLevel
        self.auc_ci_lower = 0
        self.auc_ci_upper = 0
        self.auc_at_fpr_ci_lower = 0
        self.auc_at_fpr_ci_upper = 0
        self.tpr_at_fpr_ci_lower = 0
        self.tpr_at_fpr_ci_upper = 0
        self.fpr_stop = None
        self.sys_res = None

    def __repr__(self):
        """Print from interpretor"""
        return "MediForDataContainer: eer({}), auc({}), ci_level({}), auc_ci_lower({}), auc_ci_upper({}), pauc_ci_lower({}), pauc_ci_upper({}),tpr_ci_lower({}), tpr_ci_upper({}),".format(self.eer, self.auc, self.ci_level, self.auc_ci_lower, self.auc_ci_upper, self.auc_at_fpr_ci_lower , self.auc_at_fpr_ci_upper, self.tpr_at_fpr_ci_lower, self.tpr_at_fpr_ci_upper)

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

    def get_eer(self):
        if self.eer == -1:
            self.eer = Metrics.compute_eer(self)
        return self.eer

    def get_auc(self):
        if self.auc == -1:
            self.auc = Metrics.compute_auc(self)
        return self.auc

    def get_ci(self):
        self.auc_ci_lower, self.auc_ci_upper, self.auc_at_fpr_ci_lower , self.auc_at_fpr_ci_upper, self.tpr_at_fpr_ci_lower, self.tpr_at_fpr_ci_upper = Metrics.compute_ci(self)
        return self.auc_ci_lower, self.auc_ci_upper, self.auc_at_fpr_ci_lower , self.auc_at_fpr_ci_upper, self.tpr_at_fpr_ci_lower, self.tpr_at_fpr_ci_upper

    def set_eer(self, eer):
        self.eer = eer

    def set_auc(self, auc):
        self.aux = auc