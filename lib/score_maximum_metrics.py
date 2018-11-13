import pandas as pd
import numpy as np
import sys
import os
from detMetrics import Metrics as dmets
#from plotROC import plotROC,detPackage
from constants import *

"""
Description: Computes the list of maximum metrics for each probe listed in thres_scores.
             The maximum metrics are the metrics obtained for the global threshold that yields the
             largest average of the choice metric given in the 
"""

confusion_measure_names = ["TP","TN","FP","FN"]

class max_metrics_scorer():
    def __init__(self,tlist,tscores,choice_metric="MCC",metric_list=["MCC","NMM","BWL1"]):
        """
        * Inputs:
            - thres_list: a list of the numeric thresholds computed over the list of probes
            - thres_scores: a dictionary of dataframes of metrics for each threshold computed
                            over the mask indexed by probe ID.
            - choice_metric: the metric to maximize over to obtain the largest average.
            - metric_list: The list of metrics that depend on the confusion metrics computed
                           on each threshold.
        """
        self.thres_list = list(sorted(set(tlist)))
        self.thres_scores = tscores
        assert choice_metric in metric_list, "Error: Choice metric {} is not in metric list {}.".format(choice_metric,metric_list)
        self.choice_metric = choice_metric
        self.metric_list = metric_list
        
    #TODO; parallel read them all in from the list of rows in score_df
    def preprocess_threshold_metrics(self):
        probe_thres_mets = self.thres_scores
        probe_thres_mets_new = {}
        all_thresholds = np.array(self.thres_list)
#        ['Threshold','NMM','MCC','BWL1','TP','TN','FP','FN','BNS','SNS','PNS','N']
        for p in self.thres_scores:
            thres_mets_df = probe_thres_mets[p]
            if thres_mets_df.shape[0] == 0: #a safeguard
                continue
            partial_thresholds = thres_mets_df['Threshold']
            sample_row = thres_mets_df.iloc[0]
            filled_index = np.digitize(all_thresholds,partial_thresholds,right=False) - 1
            black_rows = filled_index == -1
            black_thresholds = all_thresholds[black_rows]

            gt_pos = sample_row['TP'] + sample_row['FN']
            gt_neg = sample_row['FP'] + sample_row['TN']

            thres_mets_new_df = thres_mets_df.iloc[filled_index]
            thres_mets_new_df.at[black_thresholds,['TP','FP','TN','FN']] = gt_pos,gt_neg,0,0
            if sample_row['N'] == 0:
                thres_mets_new_df.at[:,self.metric_list] = np.nan
            else:
                #NOTE: these are all specific values for NMM, MCC, and BWL1 respectively
                if gt_pos == 0:
                    thres_mets_new_df.at[black_thresholds,self.metric_list] = np.nan,0,float(gt_neg)/sample_row['N']
                else:
                    thres_mets_new_df.at[black_thresholds,self.metric_list] = max([float(gt_pos - gt_neg)/gt_pos,-1]),0,float(gt_neg)/sample_row['N']
            
            if gt_pos == 0:
                thres_mets_new_df['TPR'] = np.nan
            else:
                thres_mets_new_df['TPR'] = thres_mets_new_df["TP"].astype(float)/gt_pos
            if gt_neg == 0:
                thres_mets_new_df['FPR'] = np.nan
            else:
                thres_mets_new_df['FPR'] = thres_mets_new_df["FP"].astype(float)/gt_neg

            #reassign the thresholds and index to all_thresholds
            thres_mets_new_df.index = all_thresholds
            probe_thres_mets_new[p] = thres_mets_new_df

        return probe_thres_mets_new

    def gen_pixel_probe_ROCs(self,task,score_task,roc_values,out_root):
        aucs = {}
        roc_values.to_csv(os.path.join(out_root,"thresMets_pixelprobe.csv"),sep="|",index=False)
        for pfx in ['Pixel','Probe']:
            tpr_name = ''.join([pfx,'TPR'])
            fpr_name = ''.join([pfx,'FPR'])
            roc_pfx = pfx
            if pfx == 'Probe':
                roc_pfx = 'Mask'
            if (roc_values[tpr_name].count() > 0) and (roc_values[fpr_name].count() > 0):
                p_roc_values = roc_values[[fpr_name,tpr_name]]
                p_roc_values = p_roc_values.append(pd.DataFrame([[0,0],[1,1]],columns=[fpr_name,tpr_name]),ignore_index=True)
                p_roc = p_roc_values.sort_values(by=[fpr_name,tpr_name],ascending=[True,True]).reset_index(drop=True)
                fpr = p_roc[fpr_name]
                tpr = p_roc[tpr_name]
                myauc = dmets.compute_auc(fpr,tpr)
                aucs[''.join([roc_pfx,'AverageAUC'])] = myauc #store in scoredf to tack onto average dataframe later
        
                #compute confusion measures by using the totals across all probes
#                confsum = scoredf[['OptimumPixelTP','OptimumPixelTN','OptimumPixelFP','OptimumPixelFN']].sum(axis=0)

                #TODO: move plotting to HTML report module
                confsum = roc_values[['TP','TN','FP','FN']].iloc[0]
#                mydets = detPackage(tpr,
#                                    fpr,
#                                    1,
#                                    0,
#                                    myauc,
#                                    confsum['TP'] + confsum['FN'],
#                                    confsum['FP'] + confsum['TN'])
##                                    confsum['OptimumPixelTP'] + confsum['OptimumPixelFN'],
##                                    confsum['OptimumPixelFP'] + confsum['OptimumPixelTN'])

                if task == 'manipulation':
                    plot_name = '_'.join([roc_pfx.lower(),'average_roc'])
                    plot_title = ' '.join([roc_pfx,'Average ROC'])
                elif task == 'splice':
                    plot_name = '_'.join([roc_pfx.lower(),'average_roc_{}'.format(score_task)])
                    plot_title = ' '.join([score_task.capitalize(),roc_pfx,'Average ROC'])
#                plotROC(mydets,plot_name,plot_title,out_root)
            else:
                aucs[''.join([roc_pfx,'AverageAUC'])] = np.nan
        return aucs

    def score_max_metrics(self,scoredf,task="manipulation",score_task="probe",probe_id_field="ProbeFileID",sbin=-10,log_dir='.'):
        """
        * Description: the top-level function that scores the maximum metrics.
        """
        max_cols = ["Maximum{}".format(m) for m in self.metric_list] + ["MaximumPixel{}".format(p) for p in confusion_measure_names ]
        max_cols.append("MaximumThreshold")
        
        #if there's nothing to score in scoredf, return it
        opt_choice_metric = "Optimum{}".format(self.choice_metric)
#        if task == 'splice':
#            if score_task == 'probe':
#                opt_choice_metric = 'p{}'.format(opt_choice_metric)
#            if score_task == 'donor':
#                opt_choice_metric = 'd{}'.format(opt_choice_metric)

        if scoredf.query("{} > -2".format(opt_choice_metric)).count()[opt_choice_metric] == 0:
            auc_cols = ['PixelAverageAUC','MaskAverageAUC']
            all_cols = max_cols + auc_cols
            for col in all_cols:
                scoredf[col] = np.nan
            return scoredf

        maxThreshold = -10
        scoredf['PixelAverageAUC'] = np.nan
        scoredf['MaskAverageAUC'] = np.nan
        
        #preprocess and then proceed to compute 
        probe_thres_mets_preprocess = self.preprocess_threshold_metrics()
        probe_thres_mets_agg = pd.concat(probe_thres_mets_preprocess.values(),keys=probe_thres_mets_preprocess.keys(),names=[probe_id_field,'Threshold'])
        thres_mets_sum = probe_thres_mets_agg.sum(level=[1])
        thres_mets_sum['PixelTPR'] = thres_mets_sum['TP']/(thres_mets_sum['TP'] + thres_mets_sum['FN'])
        thres_mets_sum['PixelFPR'] = thres_mets_sum['FP']/(thres_mets_sum['FP'] + thres_mets_sum['TN'])
        thres_mets_sum[['ProbeTPR','ProbeFPR']] = probe_thres_mets_agg[['TPR','FPR']].mean(level=[1])
        maxThreshold = thres_mets_sum[self.choice_metric].idxmax()

#        roc_values = self.parallelize(roc_values,self.runROCvals,scoreAvgROCPerProc,1,top_procs=top_procs,top_procs_apply=top_procs_apply)
#        maxThreshold = roc_values['avgMCC'].idxmax()

        #TODO: include main
        aucs = self.gen_pixel_probe_ROCs(task,score_task,thres_mets_sum,log_dir)
        auc_keys = aucs.keys()
        for pfx in ['Pixel','Mask']:
            auc_name = ''.join([pfx,'AverageAUC'])
            scoredf[auc_name] = aucs[auc_name]

        #join roc_values to scoredf
        if (sbin >= -1) and (maxThreshold > -10):
            #with the maxThreshold, set MaximumMCC for everything. Join that dataframe with this one
            scoredf['MaximumThreshold'] = maxThreshold
            #access the probe_thres_mets_agg for the threshold
            maxMCCdf = probe_thres_mets_agg.xs(maxThreshold,level=1)
            maxMCCdf[probe_id_field] = maxMCCdf.index
            maxMCCdf.drop_duplicates(inplace=True)
            #erase all rows where self.choice_metric is np.nan
            maxMCCdf[maxMCCdf[self.choice_metric].isnull()] = np.nan

            maxMCCdf.rename(columns={'NMM':'MaximumNMM',
                                     'MCC':'MaximumMCC',
                                     'BWL1':'MaximumBWL1',
                                     'TP':'MaximumPixelTP',
                                     'TN':'MaximumPixelTN',
                                     'FP':'MaximumPixelFP',
                                     'FN':'MaximumPixelFN'},inplace=True)
            scoredf = scoredf.merge(maxMCCdf[[probe_id_field,'MaximumNMM','MaximumMCC','MaximumBWL1','MaximumPixelTP','MaximumPixelTN','MaximumPixelFP','MaximumPixelFN']],on=[probe_id_field],how='left')
        else:
            for col in max_cols:
                scoredf[col] = np.nan

        return scoredf

    def gen_pixel_probe_ROCs(self,task,score_task,roc_values,out_root):
        aucs = {}
        roc_values.to_csv(os.path.join(out_root,"thresMets_pixelprobe.csv"),sep="|",index=False)
        for pfx in ['Pixel','Probe']:
            tpr_name = ''.join([pfx,'TPR'])
            fpr_name = ''.join([pfx,'FPR'])
            roc_pfx = pfx
            if pfx == 'Probe':
                roc_pfx = 'Mask'
            if (roc_values[tpr_name].count() > 0) and (roc_values[fpr_name].count() > 0):
                p_roc_values = roc_values[[fpr_name,tpr_name]]
                p_roc_values = p_roc_values.append(pd.DataFrame([[0,0],[1,1]],columns=[fpr_name,tpr_name]),ignore_index=True)
                p_roc = p_roc_values.sort_values(by=[fpr_name,tpr_name],ascending=[True,True]).reset_index(drop=True)
                fpr = p_roc[fpr_name]
                tpr = p_roc[tpr_name]
                myauc = dmets.compute_auc(fpr,tpr)
                aucs[''.join([roc_pfx,'AverageAUC'])] = myauc #store in scoredf to tack onto average dataframe later
        
                #compute confusion measures by using the totals across all probes
#                confsum = scoredf[['OptimumPixelTP','OptimumPixelTN','OptimumPixelFP','OptimumPixelFN']].sum(axis=0)

                #TODO: move plotting to HTML report module
                confsum = roc_values[['TP','TN','FP','FN']].iloc[0]
#                mydets = detPackage(tpr,
#                                    fpr,
#                                    1,
#                                    0,
#                                    myauc,
#                                    confsum['TP'] + confsum['FN'],
#                                    confsum['FP'] + confsum['TN'])
##                                    confsum['OptimumPixelTP'] + confsum['OptimumPixelFN'],
##                                    confsum['OptimumPixelFP'] + confsum['OptimumPixelTN'])

                if task == 'manipulation':
                    plot_name = '_'.join([roc_pfx.lower(),'average_roc'])
                    plot_title = ' '.join([roc_pfx,'Average ROC'])
                elif task == 'splice':
                    plot_name = '_'.join([roc_pfx.lower(),'average_roc_{}'.format(score_task)])
                    plot_title = ' '.join([score_task.capitalize(),roc_pfx,'Average ROC'])
#                plotROC(mydets,plot_name,plot_title,out_root)
            else:
                aucs[''.join([roc_pfx,'AverageAUC'])] = np.nan
        return aucs

    def score_max_metrics(self,scoredf,task="manipulation",score_task="probe",probe_id_field="ProbeFileID",sbin=-10,log_dir='.'):
        """
        * Description: the top-level function that scores the maximum metrics.
        """
        max_cols = ["Maximum{}".format(m) for m in self.metric_list] + ["MaximumPixel{}".format(p) for p in confusion_measure_names ]
        max_cols.append("MaximumThreshold")
        
        #if there's nothing to score in scoredf, return it
        opt_choice_metric = "Optimum{}".format(self.choice_metric)
#        if task == 'splice':
#            if score_task == 'probe':
#                opt_choice_metric = 'p{}'.format(opt_choice_metric)
#            if score_task == 'donor':
#                opt_choice_metric = 'd{}'.format(opt_choice_metric)

        if scoredf.query("{} > -2".format(opt_choice_metric)).count()[opt_choice_metric] == 0:
            auc_cols = ['PixelAverageAUC','MaskAverageAUC']
            all_cols = max_cols + auc_cols
            for col in all_cols:
                scoredf[col] = np.nan
            return scoredf

        maxThreshold = -10
        scoredf['PixelAverageAUC'] = np.nan
        scoredf['MaskAverageAUC'] = np.nan
        
        #preprocess and then proceed to compute 
        probe_thres_mets_preprocess = self.preprocess_threshold_metrics()
        probe_thres_mets_agg = pd.concat(probe_thres_mets_preprocess.values(),keys=probe_thres_mets_preprocess.keys(),names=[probe_id_field,'Threshold'])
        thres_mets_sum = probe_thres_mets_agg.sum(level=[1])
        thres_mets_sum['PixelTPR'] = thres_mets_sum['TP']/(thres_mets_sum['TP'] + thres_mets_sum['FN'])
        thres_mets_sum['PixelFPR'] = thres_mets_sum['FP']/(thres_mets_sum['FP'] + thres_mets_sum['TN'])
        thres_mets_sum[['ProbeTPR','ProbeFPR']] = probe_thres_mets_agg[['TPR','FPR']].mean(level=[1])
        maxThreshold = thres_mets_sum[self.choice_metric].idxmax()

#        roc_values = self.parallelize(roc_values,self.runROCvals,scoreAvgROCPerProc,1,top_procs=top_procs,top_procs_apply=top_procs_apply)
#        maxThreshold = roc_values['avgMCC'].idxmax()

        #TODO: include main
        aucs = self.gen_pixel_probe_ROCs(task,score_task,thres_mets_sum,log_dir)
        auc_keys = aucs.keys()
        for pfx in ['Pixel','Mask']:
            auc_name = ''.join([pfx,'AverageAUC'])
            scoredf[auc_name] = aucs[auc_name]

        #join roc_values to scoredf
        if (sbin >= -1) and (maxThreshold > -10):
            #with the maxThreshold, set MaximumMCC for everything. Join that dataframe with this one
            scoredf['MaximumThreshold'] = maxThreshold
            #access the probe_thres_mets_agg for the threshold
            maxMCCdf = probe_thres_mets_agg.xs(maxThreshold,level=1)
            maxMCCdf[probe_id_field] = maxMCCdf.index
            maxMCCdf.drop_duplicates(inplace=True)
            #erase all rows where self.choice_metric is np.nan
            maxMCCdf[maxMCCdf[self.choice_metric].isnull()] = np.nan

            maxMCCdf.rename(columns={'NMM':'MaximumNMM',
                                     'MCC':'MaximumMCC',
                                     'BWL1':'MaximumBWL1',
                                     'TP':'MaximumPixelTP',
                                     'TN':'MaximumPixelTN',
                                     'FP':'MaximumPixelFP',
                                     'FN':'MaximumPixelFN'},inplace=True)
            scoredf = scoredf.merge(maxMCCdf[[probe_id_field,'MaximumNMM','MaximumMCC','MaximumBWL1','MaximumPixelTP','MaximumPixelTN','MaximumPixelFP','MaximumPixelFN']],on=[probe_id_field],how='left')
        else:
            for col in max_cols:
                scoredf[col] = np.nan

        return scoredf

if __name__ == '__main__':
    print "Main not engineered yet."
