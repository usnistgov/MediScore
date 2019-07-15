class Metrics:

    @staticmethod
    def compute_points_sk(score, gt):
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

        label = np.where(gt == 'Y', 1, 0)
        target_num = label[label == 1].size
        nontarget_num = label[label == 0].size
        # print("Total# ({}),  Target# ({}),  NonTarget# ({}) \n".format(
        #    label.size, target_num, nontarget_num))

        fpr, tpr, thres = roc_curve(label, score)
        fnr = 1 - tpr
        return fpr, tpr, fnr, thres, target_num, nontarget_num

    @staticmethod
    def compute_auc(fpr, tpr, fpr_stop=1):
        """ Computes the under area curve (AUC) given FPR and TPR values
        fpr: false positive rates
        tpr: true positive rates
        fpr_stop: fpr value for calculating partial AUC"""
        width = [x - fpr[i] for i, x in enumerate(fpr[1:]) if fpr[i + 1] <= fpr_stop]
        height = [(x + tpr[i]) / 2 for i, x in enumerate(tpr[1:])]
        p_height = height[0:len(width)]
        auc = sum([width[i] * p_height[i] for i in range(0, len(width))])
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
    def compute_ci(score, gt, ci_level, fpr_stop):
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

        lower_bound = round((1.0 - float(ci_level)) / 2.0, 3)
        upper_bound = round((1.0 - lower_bound), 3)
        n_bootstraps = 500
        rng_seed = 77  # control reproducibility
        bootstrapped_auc = []
        bootstrapped_pauc = []
        bootstrapped_tpr = []

        rng = np.random.RandomState(rng_seed)
        indices = np.copy(score.index.values)
        #print("Original indices {}".format(indices))
        for i in range(n_bootstraps):
            # bootstrap by sampling with replacement on the prediction indices
            new_indices = rng.choice(indices, len(indices))
            fpr, tpr, fnr, thres, t_num, nt_num = Metrics.compute_points_sk(
                score[new_indices].values, gt[new_indices])

            auc = Metrics.compute_auc(fpr, tpr, 1)
            pauc = Metrics.compute_auc(fpr, tpr, fpr_stop)
            inter_points = Metrics.linear_interpolated_point(fpr, tpr, fpr_stop)
            tpr_at_fpr = inter_points[0][1]

            # print("Bootstrap #{} FPR_stop {}, AUC: {:0.3f}".format(i + 1, fpr_stop, auc))

            #print("New indices {}".format(new_indices))
            #label = np.where(gt[new_indices] == 'Y', 1, 0)
            #print("New label {}".format(label))
            # if np.unique(label).size < 2:
            #print("Ignore: we need at least one positive and one negative sample for ROC AUC.")
            #    continue

            #auc = roc_auc_score(label, score[new_indices].values)

            bootstrapped_auc.append(auc)
            bootstrapped_pauc.append(pauc)
            bootstrapped_tpr.append(tpr_at_fpr)
            # TODO: need pdf and ecdf distribution at each FPR
            # see the paper: Nonparametic confidence intervals for receiver operating characteristic curves (2.4)
            #fpr, tpr, thres = roc_curve(label[indices], score[indices])
            # bootstrapped_tpr.append(tpr)
            # print("Bootstrap #{} AUC: {:0.3f}".format(i + 1, auc))

        sorted_aucs = sorted(bootstrapped_auc)
        sorted_paucs = sorted(bootstrapped_pauc)
        sorted_tprs = sorted(bootstrapped_tpr)
        #print("sorted AUCS {}".format(sorted_aucs))

        # Computing the lower and upper bound of the 90% confidence interval (default)
        # You can change the bounds percentiles to 0.025 and 0.975 to get
        # a 95% confidence interval instead.
        auc_ci_lower = sorted_aucs[int(lower_bound * len(sorted_aucs))]
        auc_ci_upper = sorted_aucs[int(upper_bound * len(sorted_aucs))]
        pauc_ci_lower = sorted_paucs[int(lower_bound * len(sorted_paucs))]
        pauc_ci_upper = sorted_paucs[int(upper_bound * len(sorted_paucs))]
        tpr_ci_lower = sorted_tprs[int(lower_bound * len(sorted_tprs))]
        tpr_ci_upper = sorted_tprs[int(upper_bound * len(sorted_tprs))]

        #print("Confidence interval for AUC: [{:0.5f} - {:0.5f}]".format(ci_lower, ci_upper))
        return auc_ci_lower, auc_ci_upper, pauc_ci_lower, pauc_ci_upper, tpr_ci_lower, tpr_ci_upper

    # Calculate d-prime and beta
    @staticmethod
    def compute_dprime(fpr, tpr, d_level=0.0):
        """ computes the d-prime given TPR and FPR values
        tpr: true positive rates
        fpr: false positive rates"""
        from scipy.stats import norm
        from math import exp

        #fpr_a = [0, .0228,.0668,.1587,.3085,.5,.6915,.8413,.9332,.9772,.9938,.9987, 1]
        #tpr_a = [0, 0.0013,.0062,.0228,.0668,.1587,.3085,.5,.6915,.8413,.9332,.9772, 1]
        #d_level = 0.2

        def range_limit(n, minn, maxn):
            return max(min(maxn, n), minn)

        d = []
        d_max = None
        d_max_idx = None
        beta = []
        beta_max = None
        beta_max_idx = None
        Z = norm.ppf
        mask = []
        for idx, x in enumerate(fpr):
            d.append(Z(range_limit(tpr[idx], 0.00001, .99999)) -
                     Z(range_limit(fpr[idx], 0.00001, 0.99999)))
            beta.append(exp((Z(range_limit(fpr[idx], 0.00001, 0.99999))
                             ** 2 - Z(range_limit(tpr[idx], 0.00001, .99999))**2) / 2))
            if (tpr[idx] >= d_level and tpr[idx] <= 1 - d_level and fpr[idx] >= d_level and fpr[idx] <= 1 - d_level):
                if (d_max == None or d_max < d[idx]):
                    d_max = d[idx]
                    d_max_idx = idx
                if (beta_max == None or beta_max < d[idx]):
                    beta_max = d[idx]
                    beta_max_idx = idx
                mask.append(1)
            else:
                mask.append(0)
#        print("d_level- {} \ntpr- {} \nfpr- {} \nmask- {} \nd- {} \ndmax- {} \nidx- {} \n".format(d_level, fpr, tpr, mask, d, d_max, d_max_idx))

        if (d_max_idx == None):
            return None, (0, 0), None, (0, 0)
        return d_max, (fpr[d_max_idx], tpr[d_max_idx]), beta_max, (fpr[beta_max_idx], tpr[beta_max_idx])

    @staticmethod
    def compute_aprime(fpr_a, tpr_a):
        """ computes the d-prime given TPR and FPR values
        tpr: true positive rates
        fpr: false positive rates"""
        from scipy.stats import norm
        from math import exp

        fpr = list(fpr_a)
        tpr = list(tpr_a)
        Z = norm.ppf
        a = []
        for i in range(0, len(fpr)):

            # Starting a' calculation
            if(fpr[i] <= 0.5 and tpr[i] >= 0.5):
                a.append(0.75 + (tpr[i] - fpr[i] / 4.0) - fpr[i] * (1.0 - tpr[i]))
            elif(fpr[i] <= tpr[i] and tpr[i] <= 0.5):
                a.append(0.75 + (tpr[i] - fpr[i] / 4.0) - fpr[i] / (4.0 * tpr[i]))
            else:
                a.append(0.75 + (tpr[i] - fpr[i] / 4.0) - (1.0 - tpr[i]) / (4.0 * (1.0 - fpr[i])))

        a_idx = a.index(max(a))
        a_max_point = (fpr[a_idx], tpr[a_idx])

        #print("a- {} amax- {} idx- {} apoint- {}".format(a, max(a), a_idx, a_max_point))
#        print("tpr{}".format(tpr))
#        print("fpr{}".format(fpr))

        return max(a), a_max_point

    @staticmethod
    def linear_interpolated_point(x, y, x0):
        # given a list or numpy array of x and y, compute the y for some x0.
        # currently only applicable to interpolating ROC curves

        xy = list(zip(x, y))
        xy.sort(key=lambda x: x[0])

        if (len(x) == 0) or (len(y) == 0):
            print("ERROR: no data in x or y to interpolate.")
            exit(1)
        elif len(x) != len(y):
            print("ERROR: x and y are not the same length.")
            exit(1)

        # find x0 in the set of x's
        tuples = [p for p in xy if p[0] == x0]
        if len(tuples) > 0:
            return tuples
        else:
            # find the largest x smaller than x0
            smaller = [p for p in xy if p[0] < x0]
            ix_x01 = len(smaller) - 1  # index for the relevant point

            # if nothing is in the list of smaller points
            if ix_x01 == -1:
                x01 = 0
                y01 = 0
                x02 = xy[0][0]
                y02 = xy[0][1]
                if x02 != x01:
                    y0 = y01 + (y02 - y01) / (x02 - x01) * (x0 - x01)
                else:
                    y0 = (y02 + y01) / 2
                return [(x0, y0)]

            x01 = xy[ix_x01][0]
            y01 = xy[ix_x01][1]
            ix_x02 = ix_x01 + 1

            # check to see if there is a next x. If not, let it be (1,1)
            try:
                x02 = xy[ix_x02][0]
                y02 = xy[ix_x02][1]
            except IndexError:
                x02 = 1
                y02 = 1

            # linear interpolate
            if x02 != x01:
                y0 = y01 + (y02 - y01) / (x02 - x01) * (x0 - x01)
            else:
                y0 = (y02 + y01) / 2

            # return a single set of tuples to maintain format
            return [(x0, y0)]

    # TODO: optimize the speed (maybe vertorization?)
    @staticmethod
    def compute_points_donotuse(score, gt):  # do not match with R results
        """ computes false positive rate (FPR) and false negative rate (FNR)
        given trial scores and their ground-truth.
        score: system output scores
        gt: ground-truth for given trials
        output:
        """
        from collections import Counter
        score_sorted = np.sort(score)
        val = score_sorted[::-1]
        # Since the score was sorted, the ground-truth needs to be reallocated by the index sorted
        binary = gt[np.argsort(score)[::-1]]
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