#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import os # os.system("pause") for windows command line
import sys
import numpy as np
import pandas as pd
import os  # os.system("pause") for windows command line

#ib_path = "../../lib"
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib")
sys.path.append(lib_path)
import detMetrics as dm


# fake data #1
def unittestMetrics():
    d1 = {'fname': ['F18.jpg', 'F166.jpg', 'F86.jpg', 'F172.jpg'],
          'score': [0.034536792, 0.020949942, 0.016464296, 0.014902585],
          'gt': ["Y", "N", "Y", "N"]}
    df1 = pd.DataFrame(d1)

    df1fpr = np.array([0, 0.5, 0.5, 1])
    df1tpr = np.array([0.5, 0.5, 1, 1])

    # check points and AUC
    DM1 = dm.detMetrics(df1['score'], df1['gt'], fpr_stop=1, isCI=False, ciLevel=0.9)
    # scikit-learn Metrics
    np.testing.assert_almost_equal(DM1.fpr, df1fpr)
    np.testing.assert_almost_equal(DM1.tpr, df1tpr)
    # manual calculation by unique values
    # np.testing.assert_almost_equal(DM1.fpr2, df1fpr)
    # np.testing.assert_almost_equal(DM1.tpr2, df1tpr)

    dauc = 0.75
    np.testing.assert_almost_equal(DM1.auc, dauc)

    # check partial AUC for following fpr_stop
    fpr_stop_list1 = [0.1, 0.3, 0.5, 0.7, 0.9]
    partial_aucs1 = [0, 0, 0.25, 0.25, 0.25]
    # AUC
    for i in range(0, len(fpr_stop_list1)):
        np.testing.assert_almost_equal(dm.Metrics.compute_auc(
            df1fpr, df1tpr, fpr_stop_list1[i]), partial_aucs1[i])
    # EER
    df1eer = 0.5
    df1fnr = [1 - df1tpr[i] for i in range(0, len(df1tpr))]
    np.testing.assert_almost_equal(dm.Metrics.compute_eer(df1fpr, df1fnr), df1eer)

    # TODO: ci test

    # fake data #2. Contains duplicate scores.
    d2 = {'fname': ['F18.jpg', 'F166.jpg', 'F165.jpg', 'F86.jpg', 'F87.jpg', 'F88.jpg', 'F172.jpg'],
          'score': [0.034536792, 0.020949942, 0.020949942, 0.016464296, 0.016464296, 0.016464296, 0.014902585],
          'gt': ["Y", "N", "N", "Y", "N", "Y", "N"]}
    df2 = pd.DataFrame(d2)

    df2fpr = [0, 0.5, 0.75, 1]
    df2tpr = [1. / 3, 1. / 3, 1, 1]

    DM2 = dm.detMetrics(df2['score'], df2['gt'], fpr_stop=1)
    # scikit-learn Metrics
    np.testing.assert_almost_equal(DM2.fpr, df2fpr)
    np.testing.assert_almost_equal(DM2.tpr, df2tpr)
    # manual calculation by unique values
    # np.testing.assert_almost_equal(DM2.fpr2, df2fpr)
    # np.testing.assert_almost_equal(DM2.tpr2, df2tpr)

    eps = 10**-12
    dauc2 = 7. / 12
    np.testing.assert_allclose(dm.Metrics.compute_auc(df2fpr, df2tpr, fpr_stop=1), dauc2)

    # check partial AUC for following thresholds
    fpr_stop_list2 = [0.2, 0.4, 0.6, 0.8, 1]
    partial_aucs2 = [0, 0, 1. / 6, 1. / 3, 7. / 12]

    for j in range(0, len(fpr_stop_list2)):
        np.testing.assert_allclose(dm.Metrics.compute_auc(
            df2fpr, df2tpr, fpr_stop_list2[j]), partial_aucs2[j])

    df2eer = 7. / 12
    df2fnr = [1 - df2tpr[i] for i in range(0, len(df2tpr))]
    np.testing.assert_allclose(dm.Metrics.compute_eer(df2fpr, df2fnr), df2eer)

    # check CD@FAR
    DM3 = dm.detMetrics(df1['score'], df1['gt'], fpr_stop=0.5)
    np.testing.assert_almost_equal(DM3.tpr_at_fpr, 0.5)

    # check linear_interpolated_point
    df4fpr = [0, 1]
    df4tpr = [0, 1]
    np.testing.assert_almost_equal(
        dm.Metrics.linear_interpolated_point(df4fpr, df4tpr, 0.5)[0], (0.5, 0.5))

    print("All detection scorer unit test successfully complete.\n")


if __name__ == '__main__':

    # Boolean for disabling the command line
    debug_mode_ide = False

    # Command-line mode
    if not debug_mode_ide:
        unittestMetrics()
    else:

        print("Debug mode: Starting Unit Tests ...\n")

        # fake data #1
        d1 = {'fname': ['F18.jpg', 'F166.jpg', 'F86.jpg', 'F172.jpg'],
              'score': [0.034536792, 0.020949942, 0.016464296, 0.014902585],
              'gt': ["Y", "N", "Y", "N"]}
        df1 = pd.DataFrame(d1)

        df1fpr = np.array([0, 0.5, 0.5, 1])
        df1tpr = np.array([0.5, 0.5, 1, 1])

        # check points and AUC
        DM1 = dm.detMetrics(df1['score'], df1['gt'], fpr_stop=1)
        # scikit-learn Metrics
        np.testing.assert_almost_equal(DM1.fpr, df1fpr)
        np.testing.assert_almost_equal(DM1.tpr, df1tpr)
        # manual calculation by unique values
        # np.testing.assert_almost_equal(DM1.fpr2, df1fpr)
        # np.testing.assert_almost_equal(DM1.tpr2, df1tpr)

        dauc = 0.75
        np.testing.assert_almost_equal(DM1.auc, dauc)

        # check partial AUC for following fpr_stop
        fpr_stop_list1 = [0.1, 0.3, 0.5, 0.7, 0.9]
        partial_aucs1 = [0, 0, 0.25, 0.25, 0.25]
        # AUC
        for i in range(0, len(fpr_stop_list1)):
            np.testing.assert_almost_equal(dm.Metrics.compute_auc(
                df1fpr, df1tpr, fpr_stop_list1[i]), partial_aucs1[i])
        # EER
        df1eer = 0.5
        df1fnr = [1 - df1tpr[i] for i in range(0, len(df1tpr))]
        np.testing.assert_almost_equal(dm.Metrics.compute_eer(df1fpr, df1fnr), df1eer)

        # TODO: ci test

        # fake data #2. Contains duplicate scores.
        d2 = {'fname': ['F18.jpg', 'F166.jpg', 'F165.jpg', 'F86.jpg', 'F87.jpg', 'F88.jpg', 'F172.jpg'],
              'score': [0.034536792, 0.020949942, 0.020949942, 0.016464296, 0.016464296, 0.016464296, 0.014902585],
              'gt': ["Y", "N", "N", "Y", "N", "Y", "N"]}
        df2 = pd.DataFrame(d2)

        df2fpr = [0, 0.5, 0.75, 1]
        df2tpr = [1. / 3, 1. / 3, 1, 1]

        DM2 = dm.detMetrics(df2['score'], df2['gt'], fpr_stop=1)
        # scikit-learn Metrics
        np.testing.assert_almost_equal(DM2.fpr, df2fpr)
        np.testing.assert_almost_equal(DM2.tpr, df2tpr)
        # manual calculation by unique values
        # np.testing.assert_almost_equal(DM2.fpr2, df2fpr)
        # np.testing.assert_almost_equal(DM2.tpr2, df2tpr)

        eps = 10**-12
        dauc2 = 7. / 12
        np.testing.assert_allclose(dm.Metrics.compute_auc(df2fpr, df2tpr, fpr_stop=1), dauc2)

        # check partial AUC for following thresholds
        fpr_stop_list2 = [0.2, 0.4, 0.6, 0.8, 1]
        partial_aucs2 = [0, 0, 1. / 6, 1. / 3, 7. / 12]

        for j in range(0, len(fpr_stop_list2)):
            np.testing.assert_allclose(dm.Metrics.compute_auc(
                df2fpr, df2tpr, fpr_stop_list2[j]), partial_aucs2[j])

        df2eer = 7. / 12
        df2fnr = [1 - df2tpr[i] for i in range(0, len(df2tpr))]
        np.testing.assert_allclose(dm.Metrics.compute_eer(df2fpr, df2fnr), df2eer)

        print("All detection scorer unit tests successfully complete.\n")
