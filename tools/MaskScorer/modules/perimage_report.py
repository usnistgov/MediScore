import sys
import os
import numpy as np
import pandas as pd
import multiprocessing
from collections import OrderedDict

#lib_path = os.path.join(os.path.abspath(__file__),'../../../lib')
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../../lib')
sys.path.append(lib_path)
from constants import *
from masks import mask,refmask,refmask_color
from maskMetrics import maskMetrics as mask_metrics
from detMetrics import Metrics as dmets
from score_maximum_metrics import max_metrics_scorer
from myround import myround

pd.options.mode.chained_assignment = None

def printerr(string,verbose=None,exitcode=1):
    if verbose is not 0:
        print(string)
    exit(exitcode)

manip_metric_cols = ['OptimumThreshold','OptimumNMM','OptimumMCC','OptimumBWL1',
                    'GWL1','AUC','EER',
                    'PixelAverageAUC','MaskAverageAUC',
                    'OptimumPixelTP','OptimumPixelTN','OptimumPixelFP','OptimumPixelFN',
                    'PixelN','PixelBNS','PixelSNS','PixelPNS',
                    'MaximumThreshold','MaximumNMM','MaximumMCC','MaximumBWL1',
                    'MaximumPixelTP','MaximumPixelTN','MaximumPixelFP','MaximumPixelFN',
                    'ActualThreshold','ActualNMM','ActualMCC','ActualBWL1',
                    'ActualPixelTP','ActualPixelTN','ActualPixelFP','ActualPixelFN'
                   ]

base_metrics = ["NMM","MCC","BWL1","GWL1","AUC","EER","TP","TN","FP","FN","N","BNS","SNS","PNS","Threshold"]

blank_metrics_defaults = {"NMM":np.nan,"MCC":np.nan,"BWL1":np.nan,
                          "GWL1":np.nan,"AUC":np.nan,"EER":np.nan,
                          "TP":np.nan,"TN":np.nan,"FP":np.nan,"FN":np.nan,"Threshold":np.nan,
                          "N":np.nan,"BNS":np.nan,"SNS":np.nan,"PNS":np.nan,
                          "ActualNMM":np.nan,"ActualMCC":np.nan,"ActualBWL1":np.nan,
                          "ActualPixelTP":np.nan,"ActualPixelTN":np.nan,"ActualPixelFP":np.nan,"ActualPixelFN":np.nan,"ActualThreshold":np.nan,"Scored":"N"}

def fields2ints(df,fields):
    for f in fields:
        df.at[:,f] = df[f].dropna().apply(lambda x: str(int(x)))
    return df

def get_ordered_df_headers(taskmode,all_cols,opt_out_cols,base_metric_cols,base_last_cols):
    """
    Return the ordered dataframe headers. Also useful for generating empty data frames.
    """
    task,mode = taskmode.split(":")
    metric_cols = base_metric_cols
    last_cols = base_last_cols
    if task == 'manipulation':
        param_ids = ['ProbeFileID']
        first_refs = ['TaskID','ProbeFileID','IsTarget','ConfidenceScore','ProbeFileName','ProbeMaskFileName']
        if 'ProbeBitPlaneMaskFileName' in all_cols:
            first_refs.append('ProbeBitPlaneMaskFileName')
        first_cols = first_refs + ['OutputProbeMaskFileName','Scored']
        first_cols.extend(opt_out_cols)
    elif task == 'splice':
        param_ids = ['ProbeFileID','DonorFileID']
        file_cols_template = ['%sFileName','%sMaskFileName','Output%sMaskFileName']
        if mode == 'base':
            metric_cols = [ 'p%s' % m for m in base_metric_cols ] + [ 'd%s' % m for m in base_metric_cols ]
            file_cols_template.append('%sScored')
            last_cols = [ 'Probe%s' % m for m in base_last_cols ] + [ 'Donor%s' % m for m in base_last_cols ]

        file_cols = [ m % 'Probe' if m != "%sMaskFileName" else "BinaryProbeMaskFileName" for m in file_cols_template ] + [ m % 'Donor' for m in file_cols_template ]
        if mode == 'stack':
            file_cols.append('Scored')
            param_ids.append("ScoredMask")

        first_cols = ['TaskID'] + param_ids + ['IsTarget','ConfidenceScore'] + file_cols + opt_out_cols
        if 'JournalName' in all_cols:
            first_cols.append('JournalName')

    remaining_cols = list(set(all_cols) - set(first_cols) - set(metric_cols) - set(last_cols))
    return first_cols + metric_cols + remaining_cols + last_cols

class localization_perimage_runner():
    def __init__(self,
                 task,
                 ref_df,
                 pjj_df,
                 jm_df,
                 index_df,
                 sys_df,
                 ref_dir,
                 sys_dir,
                 ref_bin,
                 sys_bin,
                 debug_mode=True
                 ):
        self.task = task
        self.ref_df = ref_df
        self.pjj_df = pjj_df
        self.jm_df = jm_df
        self.index_df = index_df
        self.sys_df = sys_df
        self.ref_dir = ref_dir
        self.sys_dir = sys_dir
        self.ref_bin = ref_bin
        self.sys_bin = sys_bin
        self.verbose = False
        self.debug_mode = debug_mode
        self.central_met = "MCC"

        allowed_task_types = ['manipulation','splice','camera']
        if self.task not in allowed_task_types:
            printerr("ERROR: Localization task type must be one of {}.".format(allowed_task_types))
        
        manager = multiprocessing.Manager()
        self.msg_queue = manager.Queue()

    #preprocess steps to clean up data. 
    def preprocess(self,
                   query,
                   opt_out,
                   usejpeg2000,
                   verbose,
                   speedup,
                   processors):
        self.set_parameters(opt_out,usejpeg2000,verbose,speedup,processors)

        syscols = self.sys_df.columns.values.tolist()
        refcols = self.ref_df.columns.values.tolist()

        index_noncols = [ c for c in self.index_df.columns.values.tolist() if (c not in syscols) and (c not in refcols)]
        self.merged_df = self.ref_df.merge(self.index_df).merge(self.sys_df).query("IsTarget == 'Y'").drop(index_noncols,axis=1)
        if self.task == 'camera':
            self.merged_df = self.merged_df.query("IsManipulated == 'Y'")
        self.journal_join_df = self.pjj_df.merge(self.jm_df)
        evalcols = ["Evaluated"] if self.task != 'splice' else ["ProbeEvaluated","DonorEvaluated"]
        for ec in evalcols:
            self.journal_join_df[ec] = "N" if query != "" else "Y" #NOTE: for convenience

        if opt_out:
            if (self.task in ['manipulation','camera']) or self.opt_out_column == "IsOptOut":
                self.merged_df = self.merged_df.query("{} not in {}".format(self.opt_out_column,self.undesirables))
            elif self.task == 'splice':
                self.merged_df = self.merged_df.query("(ProbeStatus not in {}) and (DonorStatus not in {})".format(self.undesirables,self.undesirables))

        maxprocs = max(multiprocessing.cpu_count() - 2,1)
        nrow = self.merged_df.shape[0]
        if (processors > nrow) and (nrow > 0):
            print("Warning: too many processors for rows in the data. Defaulting to rows in data ({}).".format(nrow))
            processors = nrow
        if processors > maxprocs:
            print("Warning: the machine does not have {} processors available. Defaulting to max ({}).".format(processors,maxprocs))
            processors = maxprocs
        self.processors = processors

        if query != "":
            #process the same way as with images
            try:
                big_df = self.merged_df.merge(self.journal_join_df,how='left').query(query)
            except query_exception:
                print("Error: The query '{}' doesn't seem to refer to a valid key. Please correct the query and try again.".format(query))
                exit(1)
            
            dummy_col = "StartNodeID"
	    self.merged_df = self.merged_df.merge(big_df[self.primary_fields + [dummy_col]],how="inner").dropna().drop(dummy_col,1).drop_duplicates()
            if len(self.merged_df) == 0:
                print("The query '{}' yielded no journal data over which image scoring may take place.".format(query))
                return

            #self.journal_join_df to be filtered according to queried manipulations
            journal_join_fields_all = ['JournalName','StartNodeID','EndNodeID']
            journal_bigdf_join_fields = journal_join_fields_all + self.primary_fields
            target_manips = self.journal_join_df.reset_index().merge(big_df[journal_bigdf_join_fields + [self.probe_mask_field]],how='left',on=journal_join_fields_all).set_index('index').dropna().drop(self.probe_mask_field,1).index
            for ec in evalcols:
                self.journal_join_df.loc[target_manips,ec] = 'Y'

    def set_parameters(self,
                       opt_out,
                       usejpeg2000,
                       verbose,
                       speedup,
                       processors):
        self.usejpeg2000 = usejpeg2000 if self.task in ['manipulation','camera'] else False
        self.verbose = verbose
        self.speedup = speedup

        self.probe_id_field = "ProbeFileID"
        self.probe_w_field = 'ProbeWidth'
        self.probe_h_field = 'ProbeHeight'
        self.probe_mask_field = "ProbeBitPlaneMaskFileName" if self.usejpeg2000 else "ProbeMaskFileName"
        self.sys_mask_field = "OutputProbeMaskFileName"
        self.probe_oopx_field = "ProbeOptOutPixelValue"

        if self.task in ['manipulation','camera']:
            self.evalcol = "Evaluated"
        elif self.task == 'splice':
            self.evalcol = "ProbeEvaluated"

        opt_out_mode = 0

        #assess optout values. TODO: put this into its own function
        nc17_oo_name = "IsOptOut"
        mfc18_oo_name = "ProbeStatus"
        
        syscols = self.sys_df.columns.values.tolist()
        if mfc18_oo_name in syscols:
            self.opt_out_column = mfc18_oo_name
            self.opt_out_cols = ["ProbeStatus","DonorStatus"] if self.task == 'splice' else [mfc18_oo_name]
            self.undesirables = ["OptOutAll","OptOutLocalization"]
            all_statuses = {'Processed','NonProcessed','OptOutAll','OptOutDetection','OptOutLocalization','FailedValidation'}
        elif nc17_oo_name in syscols:
            self.opt_out_column = nc17_oo_name
            self.opt_out_cols = [nc17_oo_name]
            self.undesirables = ["Y","Localization"]
            all_statuses = {'Y','N','Detection','Localization','FailedValidation'}
        else:
            print("Error: Expected 'ProbeStatus' or 'IsOptOut'. Neiter is found in the list of columns: {}".format(syscols))
            exit(1)

        probeStatuses = set(self.sys_df[self.opt_out_column].unique().tolist())
        if probeStatuses - all_statuses > set():
            print("ERROR: Status {} is not recognized for column {}.".format(probeStatuses - all_statuses,self.opt_out_column))
            exit(1)

        if (self.task == 'splice') and (self.opt_out_column == mfc18_oo_name):
            donorStatuses = set(self.sys_df['DonorStatus'].unique().tolist())
            all_donor_statuses = {'Processed','NonProcessed','OptOutLocalization','FailedValidation'}
            donor_diff_statuses = donorStatuses - all_donor_statuses
            if donor_diff_statuses != set():
                print("ERROR: Status {} is not recognized for column DonorStatus.".format(donor_diff_statuses))
                exit(1)

        if self.task == 'manipulation':
            self.primary_fields = ["ProbeFileID"] 
        elif self.task == 'splice':
            self.primary_fields = ["ProbeFileID","DonorFileID"] 
        elif self.task == 'camera':
            self.primary_fields = ["ProbeFileID","TrainCamID"] 

    def score_all_masks(self,
                        out_root,
                        query="",
                        query_mode="",
                        opt_out=False,
                        usejpeg2000=False,
                        eks=15,
                        dks=11,
                        ntdks=15,
                        nspx=-1,
                        pppns=False,
                        kernel='box',
                        precision=16,
                        verbose=False,
                        speedup=False,
                        processors=1):

        self.preprocess(query,opt_out,usejpeg2000,verbose,speedup,processors)
        #scores optimum metrics
        if self.task in ['manipulation','camera']:
            score_df = self.score_probe_run(self.merged_df,
                                            opt_out,
                                            eks,
                                            dks,
                                            ntdks,
                                            nspx,
                                            pppns,
                                            kernel,
                                            speedup,
                                            processors,
                                            log_dir = out_root)
            score_df = self.score_max_metrics(score_df,out_root)
            score_df = pd.merge(score_df,self.merged_df)
        elif self.task == 'splice':
            for score_mode in ["Probe","Donor"]:
                self.probe_id_field = "{}FileID".format(score_mode)
                self.probe_w_field = '{}Width'.format(score_mode)
                self.probe_h_field = '{}Height'.format(score_mode)
                self.probe_mask_field = "BinaryProbeMaskFileName" if score_mode == "Probe" else "DonorMaskFileName"
                self.sys_mask_field = "Output{}MaskFileName".format(score_mode)
                self.probe_oopx_field = "{}OptOutPixelValue".format(score_mode)
        
                self.evalcol = "{}Evaluated".format(score_mode)

                df_to_score = self.merged_df if score_mode == "Probe" else df_to_score.merge(self.merged_df)

                score_df = self.score_probe_run(df_to_score,
                                                opt_out,
                                                eks,
                                                dks,
                                                ntdks,
                                                nspx,
                                                pppns,
                                                kernel,
                                                speedup,
                                                processors,
                                                log_dir = out_root)
                score_df = self.score_max_metrics(score_df,out_root)
                #need to postprocess here and rename metrics
                met_pfx = score_mode[0].lower()
                score_df.rename(columns={m:"".join([met_pfx,m]) for m in manip_metric_cols},inplace=True)
                score_df.rename(columns={"Scored":"{}Scored".format(score_mode)},inplace=True)
                if score_mode == "Donor":
                    score_df = pd.merge(score_df,df_to_score)

                df_to_score = score_df
        
        self.update_journal_join_df(out_root)

        score_df = self.postprocess(score_df,manip_metric_cols,precision,verbose)
        #score_df.apply(lambda r: self.thresscores[r[self.probe_id_field]].to_csv(os.path.join(os.path.join(out_root,r[self.probe_id_field]),"thresMets.csv"),sep="|",index=False),axis=1)

        return score_df

    #one run of the mask scoring
    def score_probe_run(self,
                        df_to_score,
                        opt_out,
                        eks=15,
                        dks=11,
                        ntdks=15,
                        nspx=-1,
                        pppns=False,
                        kernel='box',
                        speedup=False,
                        processors=1,
                        log_dir = '.'):
        #TODO: parallelize over processors
        journal_join_query = " and ".join([ "(%s == '{}')" % p for p in self.primary_fields ])
        journal_join_df_cols = self.journal_join_df.columns.values.tolist()
        if self.task == 'splice':
            if "Donor" in self.probe_id_field:
                journal_join_df_cols.remove("ProbeEvaluated")
            else:
                journal_join_df_cols.remove("DonorEvaluated")

        score_df = df_to_score.apply(lambda r: localization_perimage_runner.score_one_mask(self.task,
                                                                                           r[self.probe_id_field],
                                                                                           os.path.join(self.ref_dir,r[self.probe_mask_field]) if r[self.probe_mask_field] != '' else '',
                                                                                           os.path.join(self.sys_dir,r[self.sys_mask_field]) if r[self.sys_mask_field] != '' else '',
                                                                                           journal_join_df=self.journal_join_df.query(journal_join_query.format(*[r[p] for p in self.primary_fields]))[journal_join_df_cols],
                                                                                           probe_id_field=self.probe_id_field,
                                                                                           central_met=self.central_met,
                                                                                           opt_out=opt_out,
                                                                                           probe_status=r[self.opt_out_column],
                                                                                           usejpeg2000=self.usejpeg2000,
                                                                                           eks=eks,
                                                                                           dks=dks,
                                                                                           ntdks=ntdks,
                                                                                           sys_bin=self.sys_bin,
                                                                                           nspx=nspx if not pppns else r[self.probe_oopx_field],
                                                                                           kernel=kernel,
                                                                                           speedup=speedup,
                                                                                           identifiers={"ProbeFileID":r["ProbeFileID"],"DonorFileID":r["DonorFileID"]} if self.task == 'splice' else {"ProbeFileID":r["ProbeFileID"]},
                                                                                           verbose=self.verbose,
                                                                                           debug_mode=self.debug_mode,
                                                                                           log_dir=os.path.join(log_dir,os.path.join("_".join([r["ProbeFileID"],r["DonorFileID"]]),
                                                                                                   'probe' if "Probe" in self.probe_id_field else "donor") if self.task == 'splice' else r[self.probe_id_field])
                                                                                           ),
                                                                                           axis=1,reduce=False)

        #Add columns to score_df if empty.
        if score_df.shape[0] == 0:
            score_df_cols = score_df.columns.values.tolist()
            new_cols = set(score_df_cols + blank_metrics_defaults.keys())
            score_df = pd.DataFrame(columns=new_cols)

	t_metric_list = ["MCC","NMM","BWL1","Threshold"]
        t_pixel_list = ["TP","TN","FP","FN"]
        all_pixel_list = ["N","BNS","SNS","PNS"]
        metric_dict = { m:"Optimum{}".format(m) for m in t_metric_list }
        pixel_dict = { p:"OptimumPixel{}".format(p) for p in t_pixel_list }
        pixel_dict.update({ p:"Pixel{}".format(p) for p in all_pixel_list })
        metric_dict.update(pixel_dict)
        score_df.rename(columns=metric_dict,inplace=True)

        #TODO: print all logs to a single log here
        if self.verbose:
            print("Output from all logs follow:")

        return score_df

    @staticmethod
    def __get_row_metrics(threshold_metrics,
                          central_metric,
                          blank_row,
                          sys_bin
                          ):
        """
        * Description: to be used only by score_one_mask.
        """
        actrow = blank_row.copy()

        t_max = threshold_metrics[central_metric].idxmax()

        #TODO: want to somehow assign pixels here if at least one row is not a blank row

        #all_metrics should be obtained from threshold_metrics directly
        if np.isnan(t_max):
            all_metrics = threshold_metrics.iloc[0]
        else:
            all_metrics = threshold_metrics.loc[t_max]

        for p in ["BNS","SNS","PNS","N"]:
            blank_row[p] = all_metrics[p]

        for p in ["TP","TN","FP","FN"]:
            blank_row[p] = all_metrics[p]

        if (threshold_metrics.shape[0] == 0) or (np.isnan(t_max)):
            return blank_row

        maxrow = threshold_metrics.loc[t_max]
        if isinstance(maxrow,pd.DataFrame):
            maxrow = maxrow.iloc[0]

        #add actual metrics
        if sys_bin >= -1:
            t_act = max([t for t in threshold_metrics["Threshold"].tolist() if t <= sys_bin])
            actrow = threshold_metrics.loc[threshold_metrics["Threshold"] == t_act]
            if isinstance(actrow,pd.DataFrame):
                actrow = actrow.iloc[0]

        actual_variants = ["MCC","NMM","BWL1","Threshold"]
        actual_variants_pix = ["TP","TN","FP","FN"]
        act_template = "Actual{}"
        for m in actual_variants:
            maxrow.loc[act_template.format(m)] = actrow[m]
        act_px_template = "ActualPixel{}"
        for m in actual_variants_pix:
            maxrow.loc[act_px_template.format(m)] = actrow[m]
        thres_invariants = ["GWL1","AUC","EER"]
        for m in thres_invariants:
            maxrow.loc[m] = all_metrics[m]

        #case no ground truth
        if (maxrow['TP'] + maxrow['FN'] == 0) or (maxrow['TN'] + maxrow['FP'] == 0):
#            for p in ["TP","TN","FP","FN","BNS","SNS","PNS","N"]:
#                blank_row[p] = maxrow[p]
            return blank_row

        maxrow["Scored"] = "Y"
        return maxrow

    @staticmethod
    def score_one_mask(task,
                       probe_file_id,
                       ref_mask_name,
                       sys_mask_name,
                       journal_join_df=None,
                       probe_id_field="ProbeFileID",
                       central_met="MCC",
                       opt_out=False,
                       probe_status="Processed",
                       undesirable_statuses=["OptOutLocalization","OptOutAll"],
                       usejpeg2000=False,
                       eks=15,
                       dks=11,
                       ntdks=15,
                       sys_bin=-10,
                       nspx=-1,
                       kernel='box',
                       speedup=False,
                       identifiers={},
                       verbose=False,
                       debug_mode=False,
                       log_dir='.'):

        blank_metrics = blank_metrics_defaults.copy()
        blank_metrics.update(identifiers)
        blank_row = pd.Series(blank_metrics)

        if not os.path.isdir(log_dir):
            os.system("mkdir -p {}".format(log_dir))
    
        #output each message to its own log file. Print out the log to an overall log later.
        log_fname = os.path.join(log_dir,"{}.log".format(probe_file_id))
        log_ptr = open(log_fname,'w')

        try:
            if opt_out and (probe_status in undesirable_statuses):
                return blank_row
    
            #read reference and system videos
            #if no reference mask, return blank row
            if ref_mask_name == '':
                return blank_row
            ref_mask = refmask(ref_mask_name) if usejpeg2000 else refmask_color(ref_mask_name)
            if journal_join_df is not None:
                evalcol = "{}Evaluated".format("Probe" if "Probe" in probe_id_field else "Donor") if task == 'splice' else "Evaluated"
                ref_mask.insert_journal_data(journal_join_df,evalcol=evalcol)
            if sys_mask_name != "":
                sys_mask = mask(sys_mask_name)
            else:
                #make blank sys masks
                white_mask_name = os.path.join(log_dir,'whitemask.png')
                white_matrix = 255*np.ones(ref_mask.shape[0:2],dtype=np.uint8)
                cv2.imwrite(white_mask_name,white_matrix,[cv2.IMWRITE_PNG_COMPRESSION,0])
                sys_mask = mask(white_mask_name)
    
            #compute no scores
            rwts,bns,sns = ref_mask.aggregateNoScore(eks,dks,ntdks,kernel)
            pns = sys_mask.pixelNoScore(nspx)
   
            #have ref_mask output journal info to log_dir
            ref_mask.journal_data_to_csv(os.path.join(log_dir,"{}-journal.csv".format(probe_file_id)))
 
            #compute scores
            metric_runner = mask_metrics(ref_mask,sys_mask,rwts & pns)
            all_metrics,threshold_metrics = metric_runner.get_all_metrics(sys_bin,bns,sns,pns,eks,dks,ntdks,kernel)
    
            threshold_metrics.to_csv(os.path.join(log_dir,"thresMets.csv"),sep="|",index=False)
            threshold_metrics = threshold_metrics[[c for c in base_metrics if c in threshold_metrics.columns.values.tolist()]]

            if sys_mask_name == "":
                os.system('rm {}'.format(white_mask_name))

            #return the row with the max central metric. 
            maxrow = localization_perimage_runner._localization_perimage_runner__get_row_metrics(threshold_metrics,central_met,blank_row,sys_bin)
            for idf in identifiers:
                maxrow.loc[idf] = identifiers[idf]
            log_ptr.write("{} {} has been processed.".format(probe_id_field,probe_file_id))
            log_ptr.close()
            return maxrow
        except:
            exc_type,exc_obj,exc_tb = sys.exc_info()
            log_ptr.write("Scoring run for {} {} encountered exception {} at line {}.".format(probe_id_field,probe_file_id,exc_type,exc_tb.tb_lineno))
            if debug_mode:
                log_ptr.close()
                raise

            log_ptr.close()
            #an unprocessable system mask should yield all minimum scores
            blank_row["NMM"] = -1
            blank_row["MCC"] = 0
            blank_row["BWL1"] = 1
            blank_row["GWL1"] = 1
            blank_row["Scored"] = "Y"
            return blank_row

    #score the same way as perimage scorer.
    def score_max_metrics(self,score_df,out_root):
        thresholds = []
        thresscores = {}
        probe_id_field = self.probe_id_field
        if self.task == 'splice':
            score_df["ProbeDonorID"] = score_df["ProbeFileID"] + "_" + score_df["DonorFileID"]
            probe_id_field = "ProbeDonorID"
        threshold_csv_name = 'thresMets.csv'
        score_task = 'donor' if "Donor" in self.probe_id_field else 'probe'
        if self.task == 'splice':
            threshold_csv_name = os.path.join(score_task,threshold_csv_name)
        for i,row in score_df.iterrows():
            probe_file_id = row[probe_id_field]
            thresholds_file_path = os.path.join(os.path.join(out_root,row[probe_id_field]),threshold_csv_name)
            if os.path.isfile(thresholds_file_path):
                thres_scores = pd.read_csv(thresholds_file_path,sep="|",index_col=False,header=0)
                thresholds.extend(thres_scores["Threshold"].unique().tolist())
                thresscores[probe_file_id] = thres_scores #NOTE: double-indexed. Requires both probe and donor ID.

        #score through maximum metrics scorer and output to out_root. 
        max_scorer = max_metrics_scorer(thresholds,thresscores)
        score_df = max_scorer.score_max_metrics(score_df,
                                                task=self.task,
                                                score_task=score_task,
                                                probe_id_field=probe_id_field,
                                                sbin=self.sys_bin,
                                                log_dir=out_root)

        if self.task == 'splice':
            score_df.drop("ProbeDonorID",axis=1,inplace=True)
        return score_df

    #update self.journal_join_df evalcols with updated journal info from the reference masks
    def update_journal_join_df_part(self,new_journal_df,evalcol="Evaluated"):
        file_ids = ["ProbeFileID","DonorFileID"] if self.task == 'splice' else ["ProbeFileID"]
        all_fields = file_ids + ["JournalName","StartNodeID","EndNodeID"]
        selection_query = " and ".join(["(%s == '{}')" % f for f in all_fields])
        for i,row in new_journal_df.iterrows():
            qvals = [row[f] for f in all_fields]
            self.journal_join_df.loc[self.journal_join_df.query(selection_query.format(*qvals)).index,evalcol] = row[evalcol]

    #read in the journal join df and update self.journal_join_df
    def update_journal_join_df(self,log_dir):
        modes = ["Probe","Donor"] if self.task == 'splice' else ["Probe"]
        file_ids = ["{}FileID".format(m) for m in modes]
        #if file does not exist, skip
        evalcol="Evaluated"
        for i,row in self.merged_df.iterrows():
            for m in modes:
                file_id = "{}FileID".format(m)
                evalcol = "{}Evaluated".format(m) if self.task == 'splice' else "Evaluated"
		filename = "-".join([row[file_id],"journal.csv"])
                if self.task == 'splice':
                    jpath = os.path.join(os.path.join(log_dir,"_".join([row["ProbeFileID"],row["DonorFileID"]])),m.lower())
                else:
                    jpath = os.path.join(log_dir,row[file_id])
                jpath = os.path.join(jpath,filename)
                if not os.path.isfile(jpath):
                    continue
                self.update_journal_join_df_part(pd.read_csv(jpath,sep="|",na_filter=False,index_col=False),evalcol=evalcol)

    def output_journal_join_df(self,journal_join_name):
        self.journal_join_df.to_csv(journal_join_name,sep="|",index=False)

    #TODO: print messages if verbose
    def postprocess(self,score_df,manip_metric_cols=manip_metric_cols,precision=16,verbose=False):
        score_df = score_df.merge(self.merged_df)
        manip_metric_cols_round = manip_metric_cols[:]

        if self.task == 'splice':
            manip_metric_cols_round = ["p{}".format(c) for c in manip_metric_cols] + ["d{}".format(c) for c in manip_metric_cols]

        #round according to precision
        score_df.at[:,manip_metric_cols_round] = score_df[manip_metric_cols_round].applymap(lambda x:myround(x,precision,['sd']))

        #reformat the pixels in score_df to be integers.
        pixel_cols = [c for c in manip_metric_cols_round if "Pixel" in c] + [t for t in manip_metric_cols_round if "Threshold" in t]
        score_df = fields2ints(score_df,pixel_cols)

        #Mark certain fields as not Scored.
        if self.task in ['manipulation','camera']:
            no_manip_idx = score_df.query("OptimumMCC==-2").index
            score_df.at[no_manip_idx,'Scored'] = 'N'
            score_df.at[no_manip_idx,'OptimumMCC'] = np.nan
        elif self.task == 'splice':
            no_manip_idx_p = score_df.query("pOptimumMCC==-2").index
            no_manip_idx_d = score_df.query("dOptimumMCC==-2").index
            score_df.at[no_manip_idx_p,'ProbeScored'] = 'N'
            score_df.at[no_manip_idx_p,'pOptimumMCC'] = np.nan
            score_df.at[no_manip_idx_d,'DonorScored'] = 'N'
            score_df.at[no_manip_idx_d,'dOptimumMCC'] = np.nan

        #TODO: Pixel metrics are secondary
        if self.task in ['manipulation','camera']:
            ordered_cols = get_ordered_df_headers(":".join([self.task,"base"]),score_df.columns.values.tolist(),self.opt_out_cols,manip_metric_cols,[])
            score_df = score_df[ordered_cols]
        elif self.task == 'splice':
            score_df = score_df[get_ordered_df_headers(":".join([self.task,"base"]),score_df.columns.values.tolist(),self.opt_out_cols,manip_metric_cols,[])]

#        score_df.fillna('',inplace=True)
        return score_df

if __name__ == '__main__':
    print "Do the main."
    #TODO: add options just to generate the perprobe scores, but for one pair of masks.
    import argparse
    parser = argparse.ArgumentParser(description="Score the spatial localization regions of the video probes.")
    parser.add_argument('-t','--task',default="manipulation",help="The task to score. Default: 'manipulation'.")

    args = parser.parse_args()
    
