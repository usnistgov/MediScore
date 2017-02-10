#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
#import scipy.stats as st
#import sys
#import time

class detMetrics:
    """Class representing a set of trials and containing:
       - FNR array
       - FPR array
       - TPR array
       - EER metric
       - AUC metric
       - Confidence Interval for AUC
    """

    def __init__(self, score, gt, fpr_stop = 1, isCI=False, ciLevel=0.9):
        """Constructor"""
#        s = time.time()
#        print("sklearn: Computing points...")
#        sys.stdout.flush()
        self.fpr, self.tpr, self.fnr, self.thres = self.compute_points_sk(score, gt)
#        print("({0:.1f}s)".format(time.time() - s))

#        s = time.time()
#        print("Manual: Computing points...")
#        sys.stdout.flush()
#        self.fpr2, self.tpr2, self.fnr2, self.thres2 = self.compute_points_donotuse(score, gt)
#        print("({0:.1f}s)".format(time.time() - s))

        self.eer = Metrics.compute_eer(self.fpr, self.fnr)
        self.auc = Metrics.compute_auc(self.fpr, self.tpr, fpr_stop)
        self.d, self.dpoint, self.b, self.bpoint = Metrics.compute_dprime(self.fpr, self.tpr)
        self.a, self.apoint = Metrics.compute_aprime(self.fpr, self.tpr)
        #print ("fpr_stop test:".format(fpr_stop))

        self.ci_lower = 0
        self.ci_upper = 0
        self.ci_tpr = 0

        if isCI:
            self.ci_lower, self.ci_upper, self.ci_tpr = Metrics.compute_ci(score, gt, ciLevel)

        self.fpr_stop = fpr_stop
#        detMetrics.dm_id += 1

    def __repr__(self):
        """Print from interpretor"""
        return "DetMetrics: eer({}), auc({}), ci_lower({}), ci_upper({}), ci_tpr({})".format(self.eer, self.auc, self.ci_lower,self.ci_upper,self.ci_tpr)

    #TODO: optimize the speed (maybe vertorization?)
    def compute_points_donotuse(self, score, gt): # do not match with R results
        """ computes false positive rate (FPR) and false negative rate (FNR)
        given trial scores and their ground-truth.
        score: system output scores
        gt: ground-truth for given trials
        output:
        """
        from collections import Counter
        score_sorted = np.sort(score)
        val = score_sorted[::-1]
        binary = gt[np.argsort(score)[::-1]] # Since the score was sorted, the ground-truth needs to be reallocated by the index sorted
        # use the unique scores as a threshold value
        t = np.unique(score_sorted)[::-1]
        total = len(t)
        fpr, tpr, fnr = np.zeros(total), np.zeros(total), np.zeros(total)
        fp, tp, fn, tn = np.zeros(total), np.zeros(total), np.zeros(total), np.zeros(total)
        yes = binary == 'Y'
        no = np.invert(yes)
        counts = Counter(binary)
        n_N, n_Y = counts['N'], counts['Y']
        for i in range(0, total):
            tn[i] = np.logical_and(val < t[i], no).sum()
            fn[i] = np.logical_and(val < t[i], yes).sum()
            tp[i] = n_Y - fn[i]
            fp[i] = n_N - tn[i]
    #    print("tp = {},\ntn = {},\nfp = {},\nfn = {}".format(tp,tn,fp,fn))
        # Compute true positive rate for current threshold
        tpr = tp / (tp + fn)
        # Compute false positive rate for current threshold
        fpr = fp / (fp + tn)
        # Compute false negative rate for current threshold.
        fnr = 1 - tpr     # fnr = 1 - tpr
        # print("tpr = {}, fnr = {}\n".format(tpr[i],fpr[i]))
        return fpr, tpr, fnr, t

    def compute_points_sk(self, score, gt):
        """ computes false positive rate (FPR) and false negative rate (FNR)
        given trial scores and their ground-truth using the sklearn package.
        score: system output scores
        gt: ground-truth for given trials
        """
        from sklearn.metrics import roc_curve
#        label = np.zeros(len(gt))
#        #label =  np.where(gt=='Y', 1, 0)
#        yes_mask = np.array(gt == 'Y')#TODO: error here
#        label[yes_mask] = 1
        # TODO:  Print the number of trials (target and non-target) here

        label = np.where(gt=='Y', 1, 0)
        target_num = label[label==1].size
        nontarget_num = label[label==0].size
        print("Total# ({}),  Target# ({}),  NonTarget# ({}) \n".format(label.size, target_num, nontarget_num))

        fpr, tpr, thres = roc_curve(label, score)
        fnr = 1 - tpr
        return fpr, tpr, fnr, thres

    def write(self, file_name):
        """ Save the Dump files (formatted in a binary) that contains
        a list of FAR, FPR, TPR, threshold, AUC, and EER values.
        file_name: Dump file name
        """
        import pickle
        dmFile = open(file_name, 'wb')
        pickle.dump(self, dmFile)
        dmFile.close()

    def render_table(self):
        """ Render CSV table using Pandas Data Frame
        """
        from collections import OrderedDict
        from pandas import DataFrame
        data = OrderedDict([('AUC',self.auc),('FAR_STOP',self.fpr_stop),('EER',self.eer),('AUC_CI_LOWER',self.ci_lower), ('AUC_CI_UPPER',self.ci_upper)])
        my_table = DataFrame(data,index=['0'])

        return my_table.round(6)
        #my_table.to_csv(file_name, index = False)

    def get_eer(self):
        if self.eer == -1:
            self.eer = Metrics.compute_eer(self)
        return self.eer

    def get_auc(self):
        if self.auc == -1:
            self.auc = Metrics.compute_auc(self)
        return self.eer

    def get_ci(self):
        self.ci_lower, self.ci_upper, self.ci_tpr = Metrics.compute_ci(self)
        return self.ci_lower, self.ci_upper, self.ci_tpr

    def set_eer(self, eer):
        self.eer = eer

    def set_auc(self, auc):
        self.aux = auc


def load_dm_file(path):
    """ Load Dump (DM) files
        path: DM file name along with the path
    """
    import pickle
    file = open(path, 'rb')
    myObject = pickle.load(file)
    file.close()
    return myObject


class Metrics:

    @staticmethod
    def compute_auc(fpr, tpr, fpr_stop=1):
        """ Computes the under area curve (AUC) given FPR and TPR values
        fpr: false positive rates
        tpr: true positive rates
        fpr_stop: fpr value for calculating partial AUC"""
        width = [x - fpr[i] for i, x in enumerate(fpr[1:]) if fpr[i+1] <= fpr_stop]
        height = [(x+tpr[i])/2 for i, x in enumerate(tpr[1:])]
        p_height = height[0:len(width)]
        auc = sum([width[i]*p_height[i] for i in range(0, len(width))])
        return auc

    @staticmethod
    def compute_eer(fpr, fnr):
        """ computes the equal error rate (EER) given FNR and FPR values
        fpr: false positive rates
        fnr: false negative rates"""
        errdif = [abs(fpr[j] - fnr[j]) for j in range(0, len(fpr))]
        idx = errdif.index(min(errdif))
        eer = np.mean([fpr[idx], fnr[idx]])
        return eer

    # TODO: need to validate this and make command-line inputs
    @staticmethod
    def compute_ci(score, gt, ci_level):
        """ compute the confidence interval for AUC
        score: system output scores
        gt: ground-truth for given trials
        lower_bound: lower bound percentile
        upper_bound: upper bound percentile"""
        from sklearn.metrics import roc_auc_score
#        from sklearn.metrics import roc_curve
#        score = score.astype(np.float64)
#        mean = np.mean(score)
#        size = len(score) - 1
#        return abs(mean - st.t.interval(prob, size, loc=mean, scale=st.sem(score))[1])

        lower_bound = round((1.0 - float(ci_level))/2.0, 3)
        upper_bound = round((1.0 - lower_bound), 3)
        n_bootstraps = 500
        rng_seed = 77  # control reproducibility
        bootstrapped_auc = []
#        bootstrapped_tpr = []

        rng = np.random.RandomState(rng_seed)
        indices = np.copy(score.index.values)
        #print("Original indices {}".format(indices))
        for i in range(n_bootstraps):
            # bootstrap by sampling with replacement on the prediction indices
            new_indices = rng.choice(indices, len(indices))
            #print("New indices {}".format(new_indices))
            label = np.where(gt[new_indices] == 'Y', 1, 0)
            #print("New label {}".format(label))
            if np.unique(label).size < 2:
                #print("Ignore: we need at least one positive and one negative sample for ROC AUC.")
                continue

            #auc = roc_auc_score(label, score_shuffled.values)
            auc = roc_auc_score(label, score[new_indices].values)
            bootstrapped_auc.append(auc)
            #TODO: need pdf and ecdf distribution at each FPR
            #see the paper: Nonparametic confidence intervals for receiver operating characteristic curves (2.4)
            #fpr, tpr, thres = roc_curve(label[indices], score[indices])
            #bootstrapped_tpr.append(tpr)
            #print("Bootstrap #{} AUC: {:0.3f}".format(i + 1, auc))

        sorted_aucs = sorted(bootstrapped_auc)
        #print("sorted AUCS {}".format(sorted_aucs))

        # Computing the lower and upper bound of the 90% confidence interval (default)
        # You can change the bounds percentiles to 0.025 and 0.975 to get
        # a 95% confidence interval instead.
        ci_lower = sorted_aucs[int(lower_bound * len(sorted_aucs))]
        ci_upper = sorted_aucs[int(upper_bound * len(sorted_aucs))]

        ci_tpr = 0 #TODO: after calculating CI for each TPR
        #print("Confidence interval for AUC: [{:0.5f} - {:0.5f}]".format(ci_lower, ci_upper))
        return ci_lower, ci_upper, ci_tpr

    # Calculate d-prime and beta
    @staticmethod
    def compute_dprime(fpr, tpr):
        """ computes the d-prime given TPR and FPR values
        tpr: true positive rates
        fpr: false positive rates"""
        from scipy.stats import norm
        from math import exp

        Z = norm.ppf
        d = []
        beta = []
        for i in range(0, len(fpr)):
            # Starting d' calculation
            #avoid d' infinity
            if tpr[i] == 1: tpr[i] =0.9975
            if fpr[i] == 1: fpr[i] =0.9975
            if tpr[i] == 0: tpr[i] =0.0025
            if fpr[i] == 0: fpr[i] =0.0025
            d.append(Z(tpr[i]) - Z(fpr[i]))
            beta.append(exp(Z(fpr[i])**2 - Z(tpr[i])**2)/2)

        #d = [ Z(tpr[i]) - Z(fpr[i]) for i in range(0, len(fpr)) ]
        #beta = [ exp(Z(fpr[i])**2 - Z(tpr[i])**2)/2 for i in range(0, len(fpr)) ]
        #c = [ -(Z(tpr[i]) - Z(fpr[i]))/2 for i in range(0, len(fpr)) ]
        d_idx = d.index(max(d))
        d_max_point = (fpr[d_idx], tpr[d_idx])

        b_idx = beta.index(max(beta))
        b_max_point = (fpr[b_idx], tpr[b_idx])

        #print("beta{}".format(beta))
#        print("d- {} dmax- {} idx- {} bpoint- {}".format(d, max(d), d_idx, d_max_point))
#        print("b- {} amax- {} idx- {} bpoint- {}".format(beta, max(beta), b_idx, b_max_point))

        return max(d), d_max_point, max(beta), b_max_point


    @staticmethod
    def compute_aprime(fpr, tpr):
        """ computes the d-prime given TPR and FPR values
        tpr: true positive rates
        fpr: false positive rates"""
        from scipy.stats import norm
        from math import exp

        Z = norm.ppf
        a = []
        for i in range(0, len(fpr)):

            # Starting a' calculation
            if(fpr[i] <=0.5 and tpr[i] >=0.5):
                a.append(0.75 + (tpr[i]-fpr[i]/4.0) - fpr[i]*(1.0-tpr[i]))
            elif(fpr[i] <= tpr[i] and tpr[i] <=0.5):
                a.append(0.75 + (tpr[i]-fpr[i]/4.0) - fpr[i]/(4.0*tpr[i]))
            else:
                a.append(0.75 + (tpr[i]-fpr[i]/4.0) - (1.0-tpr[i])/(4.0*(1.0-fpr[i])))


        a_idx = a.index(max(a))
        a_max_point = (fpr[a_idx], tpr[a_idx])

        #print("a- {} amax- {} idx- {} apoint- {}".format(a, max(a), a_idx, a_max_point))
#        print("tpr{}".format(tpr))
#        print("fpr{}".format(fpr))

        return max(a), a_max_point

#fpr1 = [.0228,.0668,.1587,.3085,.5,.6915,.8413,.9332,.9772,.9938,.9987]
#tpr1 = [0.0013,.0062,.0228,.0668,.1587,.3085,.5,.6915,.8413,.9332,.9772]
