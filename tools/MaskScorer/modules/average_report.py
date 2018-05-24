import numpy as np
import pandas as pd
import sys
import os
import numbers
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../lib")
sys.path.append(lib_path)
import Partition_mask18 as pt
from myround import myround

nonelist = ['',None,np.nan]

def round_df(my_df,metlist,precision=16,round_modes=['sd']):
    df_cols = my_df.columns.values.tolist()
    final_metlist = [met for met in metlist if met in df_cols] #intersection
    my_df[final_metlist] = my_df[final_metlist].applymap(lambda n: myround(n,precision,round_modes))
    return my_df

def fields2ints(df,fields):
    for f in fields:
        df[f] = df[f].dropna().apply(lambda x: str(int(x)))
    return df

def is_number(n):
    if isinstance(n,numbers.Number):
        if not np.isnan(n):
            return True
    return False

#TODO: turn this into an object

def average_report(task,score_df,metrics,constant_metrics,factor_mode,query,outroot_and_pfx,optout=False,precision=16,round_modes=['sd']):
    if task == 'manipulation':
        #TODO: expand to thresholds and the like?
        primary_met_absent = 'OptimumMCC' not in metrics
        primary_met = 'OptimumMCC'
    elif task == 'splice':
        primary_met_absent = ('pOptimumMCC' not in metrics) or ('dOptimumMCC' not in metrics)
        primary_met = '(pOptimumMCC,dOptimumMCC)'

    if primary_met_absent:
        print("Error: {} is not in the metrics provided.".format(primary_met))
        return 1

    outroot = os.path.dirname(outroot_and_pfx)
    prefix = os.path.basename(outroot_and_pfx)

    df_heads = score_df.columns.values.tolist()
    constant_mets = {}
    for m in constant_metrics:
        constant_mets[m] = myround(score_df[m].iloc[0],precision,round_modes) if score_df.shape[0] > 0 else ''

    optout_mode = 0
    if optout:
        optout_mode = 1 if 'IsOptOut' in df_heads else 2

    if task == 'manipulation':
        scorequery = "Scored=='Y'"
        lastcols = ['TRR','totalTrials','ScoreableTrials','totalOptIn','totalOptOut','optOutScoring']
    elif task == 'splice':
        scorequery = "(ProbeScored=='Y') or (DonorScored=='Y')"
        lastcols = ['pTRR','dTRR','totalTrials','ScoreableProbeTrials','ScoreableDonorTrials']
        oo_cols = ['totalProbeOptIn','totalProbeOptOut','totalDonorOptIn','totalDonorOptOut']
        lastcols.extend(oo_cols)
        lastcols.append('optOutScoring')

    total_trials = score_df.shape[0]
    scoredonly_df = score_df.query(scorequery)
    scoreable_trials = scoredonly_df.shape[0]
    #TODO: Should TRR be for only the average queried or should it be a metric for the perimage dataset?
    #TODO: Fix the implementation of the partitioner? Recompute TRR in there?
    my_partition = pt.Partition(task,scoredonly_df,query,factor_mode,metrics)
    df_list = my_partition.render_table(metrics)
    if len(df_list) == 0:
        first_cols = ['TaskID','Query']
        mets_and_stddev = []
        for m in metrics:
            mets_and_stddev.append(m)
            mets_and_stddev.append("stddev_%s" % m)
        all_cols = first_cols + mets_and_stddev + constant_metrics + lastcols
        empty_df = pd.DataFrame(columns=all_cols)
        for m in constant_metrics:
            empty_df[m] = constant_mets[m]
        return empty_df

    #depending on the factor_mode, score
    if factor_mode == 'q':
        #a_df get the headers of temp_df and tack entries on one after the other
        a_df = pd.DataFrame(columns=df_list[0].columns)
        for i,temp_df in enumerate(df_list):
            temp_df = postprocess_avg_report(task,temp_df,score_df,metrics,constant_mets,lastcols,optout,precision,round_modes)
            if temp_df is 0:
                continue
            temp_df.to_csv(path_or_buf="{}_{}.csv".format(os.path.join(outroot,'_'.join([prefix,'mask_scores'])),i),sep="|",index=False)
            a_df = a_df.append(temp_df,ignore_index=True)
    else:
        #one data frame
        a_df = df_list[0]
        a_df = postprocess_avg_report(task,a_df,score_df,metrics,constant_mets,lastcols,optout,precision,round_modes)
        if a_df is not 0:
            a_df.to_csv(path_or_buf=os.path.join(outroot,"_".join([prefix,"mask_score.csv"])),sep="|",index=False)
        
    return a_df

#postprocessing with TRR and misc optOut information after averaging
def postprocess_avg_report(task,a_df,score_df,metrics,constant_mets,lastcols,optout=False,precision=16,round_modes=['sd']):
    if a_df is 0:
        return 0

    for m in constant_mets:
        a_df[m] = constant_mets[m]

    total_trials = score_df.shape[0]
    a_df['totalTrials'] = total_trials
    a_df_heads = a_df.columns.values.tolist()
    
    df_heads = score_df.columns.values.tolist()
    max_heads = [ h for h in a_df_heads if 'MaximumThreshold' in h]
    act_heads = [ h for h in a_df_heads if 'ActualThreshold' in h]

    if 'IsOptOut' in df_heads:
        optout_mode = 1
    elif 'ProbeStatus' in df_heads:
        optout_mode = 2
    score_oo_cols = 'IsOptOut' if optout_mode == 1 else 'ProbeStatus'
    
    if task == 'manipulation':
        pfx_set = ['']
    elif task == 'splice':
        pfx_set = ['p','d']
        
    for pfx in pfx_set:
        opt_t = '%sOptimumThreshold' % pfx
        max_t = '%sMaximumThreshold' % pfx
        act_t = '%sActualThreshold' % pfx

        a_df.loc[:,opt_t] = a_df[opt_t].dropna().astype(int).astype(str)
        if (act_t in a_df_heads) and (a_df.shape[0] > 0):
            if is_number(a_df[act_t].iloc[0]):
                a_df.loc[:,max_heads] = a_df[max_heads].dropna().astype(int).astype(str)
                a_df.loc[:,act_heads] = a_df[act_heads].dropna().astype(int).astype(str)
            else:
                a_df.loc[:,max_heads] = ''
                a_df.loc[:,act_heads] = ''
        else:
            a_df.at[:,max_heads] = ''
            a_df.at[:,act_heads] = ''

        if optout_mode == 1:
            undesirables = ['Y','Localization']
        elif optout_mode == 2:
            if (pfx == 'p') or (pfx == ''):
                undesirables = ['OptOutAll','OptOutLocalization']
            elif pfx == 'd':
                undesirables = ['OptOutLocalization']
                score_oo_cols = 'DonorStatus'

        if pfx == 'p':
            loctype = 'Probe'
        elif pfx == 'd':
            loctype = 'Donor'
        elif pfx == '':
            loctype = ''

        n_optout = score_df.query("{}=={}".format(score_oo_cols,undesirables)).shape[0]
        scoreable_trials = score_df.query("({}Scored=='Y') and ({}OptimumMCC!='')".format(loctype,pfx)).shape[0]
        a_df['Scoreable%sTrials' % loctype] = scoreable_trials
        a_df['%sTRR' % pfx] = float(scoreable_trials)/total_trials if total_trials > 0 else np.nan
        a_df['total%sOptOut' % loctype ] = n_optout
        a_df['total%sOptIn' % loctype ] = total_trials - n_optout
    a_df = fields2ints(a_df,['totalTrials','Scoreable%sTrials' % loctype,'total%sOptOut' % loctype,'total%sOptIn' % loctype])

    a_df['optOutScoring'] = 'Y' if optout else 'N'

    a_df_heads = a_df.columns.values.tolist()
    met_heads = [ f for f in a_df_heads if f not in lastcols]
    heads = met_heads[:]
#    heads.extend(constant_mets.keys())
    heads.extend(lastcols)

#    stdev_mets = [met for met in met_heads if 'stddev' in met]
#    my_metrics_to_be_scored = metrics + stdev_mets + lastcols
#    float_mets = [met for met in my_metrics_to_be_scored if ('Threshold' not in met) or ('stddev' in met)]
    a_df = round_df(a_df,a_df_heads,precision=precision)
    a_df = a_df[heads]
    return a_df

if __name__ == '__main__':
    import argparse
    #TODO: then pass in the arguments
    #TODO: reference and query arguments
