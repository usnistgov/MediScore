import h5py
import sys
import os
import numpy as np
import pandas as pd
import ast
import multiprocessing
#lib_path = os.path.join(os.path.abspath(__file__),'../../../lib')
this_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(this_dir,'../../../lib')
sys.path.append(lib_path)
from constants import query_exception
from video_masks import video_mask,video_ref_mask,gen_mask,shift_intervals
from score_video_masks import get_confusion_measures,compute_metrics,score_GWL1,score_temporal_metrics
import maskMetrics as mm
from score_maximum_metrics import max_metrics_scorer
img_path = os.path.join(this_dir,'../../MaskScorer/modules')
sys.path.append(img_path)
from perimage_report import localization_perimage_runner


def printerr(string,verbose=None,exitcode=1):
    if verbose is not 0:
        print(string)
    exit(exitcode)

base_metric_cols = ['OptimumThreshold','OptimumNMM','OptimumMCC','OptimumBWL1',
                    'GWL1','AUC','EER',
                    'PixelAverageAUC','MaskAverageAUC',
                    'OptimumPixelTP','OptimumPixelTN','OptimumPixelFP','OptimumPixelFN',
                    'PixelN','PixelBNS','PixelSNS','PixelPNS',
                    'TemporalMCC',
                    'TemporalFrameTP','TemporalFrameTN','TemporalFrameFP','TemporalFrameFN',
                    'MaximumThreshold','MaximumNMM','MaximumMCC','MaximumBWL1',
                    'MaximumPixelTP','MaximumPixelTN','MaximumPixelFP','MaximumPixelFN',
                    'ActualThreshold','ActualNMM','ActualMCC','ActualBWL1',
                    'ActualPixelTP','ActualPixelTN','ActualPixelFP','ActualPixelFN'
                   ]

blank_metrics_defaults = {"NMM":np.nan,"MCC":np.nan,"BWL1":np.nan,"TemporalMCC":np.nan,
                          "GWL1":np.nan,"AUC":np.nan,"EER":np.nan,
                          "TP":np.nan,"TN":np.nan,"FP":np.nan,"FN":np.nan,"Threshold":np.nan,
                          "N":np.nan,"BNS":np.nan,"SNS":np.nan,"PNS":np.nan,
                          "TemporalFrameTP":np.nan,"TemporalFrameTN":np.nan,"TemporalFrameFP":np.nan,"TemporalFrameFN":np.nan,
                          "ActualNMM":np.nan,"ActualMCC":np.nan,"ActualBWL1":np.nan,
                          "ActualPixelTP":np.nan,"ActualPixelTN":np.nan,"ActualPixelFP":np.nan,"ActualPixelFN":np.nan,"ActualThreshold":np.nan,
                          "Scored":"N"}

class perprobe_module(localization_perimage_runner):
    def __init__(self,
                 task,
                 ref_df,
                 pjj_df,
                 jm_df,
                 index_df,
                 sys_df,
                 ref_dir,
                 sys_dir,
                 ref_bin=-10,
                 sys_bin=-10
                 ):
        localization_perimage_runner.__init__(self,
                                              task,
                                              ref_df,
                                              pjj_df,
                                              jm_df,
                                              index_df,
                                              sys_df,
                                              ref_dir,
                                              sys_dir,
                                              ref_bin,
                                              sys_bin
                                              )
        self.media = 'video'

    # image localization scorer.
    def preprocess(self,query,opt_out,temporal_scoring_only=False,processors=1,verbose=False):
#        confidence_score_field = "ConfidenceScore"
#        self.sys_df.loc[pd.isnull(sys_df[confidence_score_field]),confidence_score_field] = sys_df[confidence_score_field].min()
#        self.sys_df[confidence_score_field] = sys_df[confidence_score_field].astype(np.float)

        self.set_parameters(opt_out,verbose,processors)
        syscols = self.sys_df.columns.values.tolist()
        refcols = self.ref_df.columns.values.tolist()
        if "VideoTaskDesignation" in refcols:
            self.ref_df = self.ref_df.query("(IsTarget == 'Y') and ({} != '') and (VideoTaskDesignation in ['spatial','spatial-temporal'])".format(self.probe_mask_field)) #TODO: prone to change depending on implementation. Check later.

        if temporal_scoring_only:
            self.central_met = "TemporalMCC"

        if (self.task == 'manipulation') and (self.opt_out_cols == ["ProbeStatus"]) and (opt_out):
            self.sys_df = self.sys_df.query("{} not in {}".format(self.opt_out_column[0],self.undesirables))
            if temporal_scoring_only:
                self.sys_df = self.sys_df.query("{} != 'OptOutTemporal'".format(self.opt_out_column[0]))

        self.merged_df = self.ref_df.merge(self.index_df).merge(self.sys_df)
        self.journal_join_df = self.pjj_df.merge(self.jm_df)
        self.journal_join_df[self.evalcol] = "N" if query != "" else "Y" #NOTE: for convenience

        if query != "":
            #process the same way as with images
            try:
                big_df = merged_df.merge(self.journal_join_df,how='left').query(query)
            except query_exception:
                print("Error: The query '{}' doesn't seem to refer to a valid key. Please correct the query and try again.".format(query))
                exit(1)
            
            #filter merged_df accordingly
            self.merged_df = self.merged_df.merge(big_df[self.primary_fields + ["StartNodeID"]],how="inner").dropna().drop("StartNodeID",1).drop_duplicates()

            #self.journal_join_df to be filtered according to queried manipulations
            journal_join_fields_all = ['JournalName','StartNodeID','EndNodeID']
            journal_bigdf_join_fields = journal_join_fields_all + self.primary_fields
            target_manips = self.journal_join_df.reset_index().merge(big_df[journal_bigdf_join_fields + [self.probe_mask_field]],how='left',on=journal_join_fields_all).set_index('index').dropna().drop(self.probe_mask_field,1).index
            self.journal_join_df.loc[target_manips,self.evalcol] = 'Y'

    def set_parameters(self,
                       opt_out,
                       verbose,
                       processors):
        self.verbose = verbose
        self.processors = processors
        if self.task == 'manipulation':
            self.probe_id_field = "ProbeFileID"
            self.probe_mask_field = "HDF5MaskFileName"
            self.sys_mask_field = "OutputProbeMaskFileName"
            self.evalcol = "Evaluated"
            self.sys_df["Scored"] = "Y"

        syscols = self.sys_df.columns.values.tolist()
        if "ProbeStatus" in syscols:
            self.opt_out_cols = ["ProbeStatus"]
            self.undesirables = ["OptOutAll","OptOutLocalization"]

        if self.task == 'manipulation':
            self.primary_fields = ["ProbeFileID"] 
        else:
            self.primary_fields = ["ProbeFileID","DonorFileID"] 
                
    def score_all_masks(self,
                        out_root,
                        query="",
                        query_mode="",
                        opt_out=False,
                        video_opt_out=False,
                        truncate=False,
                        collars=None,
                        temporal_gt_only=False,
                        temporal_scoring_only=False,
                        eks=15,
                        dks=11,
                        ntdks=15,
                        nspx=-1,
                        pppns=False,
                        kernel='box',
                        precision=16,
                        verbose=False,
                        processors=1):
        self.preprocess(query,opt_out,temporal_scoring_only=temporal_scoring_only,verbose=verbose,processors=processors)

        #scores optimum metrics
        score_df = self.score_probe_run(opt_out,
                                        video_opt_out,
                                        truncate,
                                        collars,
                                        temporal_gt_only,
                                        temporal_scoring_only,
                                        eks,
                                        dks,
                                        ntdks,
                                        nspx,
                                        pppns,
                                        kernel,
                                        processors,
                                        log_dir = out_root)
        t_metric_list = ["MCC","NMM","BWL1"]
        t_pixel_list = ["TP","TN","FP","FN"]
        metric_dict = { m:"Optimum{}".format(m) for m in t_metric_list }
        pixel_dict = {p:"OptimumPixel{}".format(p) for p in t_pixel_list}
        metric_dict.update(pixel_dict)
        score_df.rename(columns=metric_dict,inplace=True)

        #score maximum metrics
        score_df = self.score_max_metrics(score_df,out_root)
        
        #join with reference-system merged_df
        score_df = pd.merge(score_df,self.merged_df)

        score_df = self.postprocess(score_df,base_metric_cols,precision=precision,verbose=verbose)

        #score_df.apply(lambda r: self.thresscores[r[self.probe_id_field]].to_csv(os.path.join(os.path.join(out_root,r[self.probe_id_field]),"thresMets.csv"),sep="|",index=False),axis=1)

        return score_df

    #one run of the mask scoring
    def score_probe_run(self,
                        opt_out,
                        video_opt_out,
                        truncate,
                        collars,
                        temporal_gt_only=False,
                        temporal_scoring_only=False,
                        eks=15,
                        dks=11,
                        ntdks=15,
                        nspx=-1,
                        pppns=False,
                        kernel='box',
                        processors=1,
                        log_dir = '.'):
        #TODO: parallelize over processors
        journal_join_query = " and ".join([ "(%s == '{}')" % p for p in self.primary_fields ])
        score_df = self.merged_df.apply(lambda r: perprobe_module.score_one_mask(r[self.probe_id_field],
                                                                                 os.path.join(self.ref_dir,r[self.probe_mask_field]),
                                                                                 os.path.join(self.sys_dir,r[self.sys_mask_field]) if r[self.sys_mask_field] != '' else '',
                                                                                 journal_join_df=self.journal_join_df.query(journal_join_query.format(*[r[p] for p in self.primary_fields])),
                                                                                 probe_id_field=self.probe_id_field,
                                                                                 probe_status=r["ProbeStatus"] if opt_out else "Processed",
                                                                                 central_met=self.central_met,
                                                                                 truncate=truncate,
                                                                                 collars=collars,
                                                                                 opt_out_frames=ast.literal_eval(r["VideoFrameOptOutSegments"])if video_opt_out else [],
                                                                                 temporal_gt_only=temporal_gt_only,
                                                                                 temporal_scoring_only=temporal_scoring_only,
                                                                                 eks=eks,
                                                                                 dks=dks,
                                                                                 ntdks=ntdks,
                                                                                 sys_bin=self.sys_bin,
                                                                                 nspx=nspx if not pppns else -1, 
                                                                                 kernel=kernel,
                                                                                 log_dir=os.path.join(log_dir,r[self.probe_id_field])),
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

        return score_df

    @staticmethod
    def score_one_mask(probe_file_id,
                       ref_mask_name,
                       sys_mask_name,
                       journal_join_df,
                       probe_id_field="ProbeFileID",
                       probe_status="Processed",
                       central_met="MCC",
                       truncate=False,
                       collars=None,
                       opt_out_frames=[],
                       temporal_gt_only=False,
                       temporal_scoring_only=False,
                       eks=15,
                       dks=11,
                       ntdks=15,
                       sys_bin=-10,
                       nspx=-1,
                       kernel='box',
                       log_dir='.'):
        if not os.path.isdir(log_dir):
            os.system("mkdir {}".format(log_dir))

        #read reference and system videos
        ref_mask = video_ref_mask(ref_mask_name)
        ref_mask.insert_journal_data(journal_join_df)
        if sys_mask_name != "":
            sys_mask = video_mask(sys_mask_name)
        else:
            #make blank sys masks
            white_mask_name = os.path.join(log_dir,'whitemask.hdf5')
            gen_mask(white_mask_name,ref_mask.shape,ref_mask.framecount)
            sys_mask = video_mask(white_mask_name)

        #generate collars and opt out frame intervals. Shift them back by one.
        ref_intervals = ref_mask.compute_ref_intervals(eks=eks,dks=dks,ntdks=ntdks,sys=sys_mask if sys_mask_name != "" else None,nspx=nspx,kern=kernel)
#        ref_mask.collars = shift_intervals(ref_mask.compute_collars(shift_intervals(ref_intervals,shift=1),collars=collars),shift=-1)
#        ref_mask.opt_out_frames = shift_intervals(opt_out_frames,shift=-1)
        ref_mask.collars = ref_mask.compute_collars(ref_intervals,collars=collars)
        ref_mask.opt_out_frames = opt_out_frames

        #also score temporal localization metrics by passing them into the video temporal localization scorer
        temporal_scores = 0
        if probe_status not in ["OptOutTemporal","OptOutLocalization","OptOutAll"]:
            temporal_scores = score_temporal_metrics(ref_mask,sys_mask,collars,truncate=truncate,eks=eks,dks=dks,ntdks=ntdks,kern=kernel)
            
            temporal_scores.rename(columns={"MCC":"TemporalMCC",
                                            "TP":"TemporalFrameTP",
                                            "TN":"TemporalFrameTN",
                                            "FP":"TemporalFrameFP",
                                            "FN":"TemporalFrameFN"
                                            },inplace=True)

        #TODO: if temporal_scoring_only, skip the below and return just the temporal metrics, with other metrics empty?
        #TODO: test this
        if not (temporal_scoring_only or (probe_status in ['OptOutSpatial',"OptOutLocalization","OptOutAll"])):
            #pass them into the video scorer
            confusion_measures = get_confusion_measures(ref_mask,
                                                        sys_mask,
                                                        truncate=truncate,
                                                        temporal_gt_only=temporal_gt_only,
                                                        eks=eks,
                                                        dks=dks,
                                                        ntdks=ntdks,
                                                        pppns=nspx,
                                                        kern=kernel
                                                        )
    
            #get the metrics for each set of confusion measures
            spatial_scores = compute_metrics(confusion_measures)
    
            #separate procedure to score GWL1 (not threshold dependent), AUC, and EER using det metrics
            spatial_scores["GWL1"] = score_GWL1(ref_mask,
                                                sys_mask,
                                                truncate=truncate,
                                                temporal_gt_only=temporal_gt_only,
                                                eks=eks,
                                                dks=dks,
                                                ntdks=ntdks,
                                                pppns=nspx,
                                                kern=kernel
                                                )
            if temporal_scores is 0:
                scores = spatial_scores
            else: 
                scores = pd.concat([spatial_scores,temporal_scores],axis=1)
        else:
            scores = temporal_scores
            if scores is 0:
                ref_mask.close()
                sys_mask.close()
                blank_row = pd.Series(blank_metrics_defualts)
                blank_row["ProbeFileID"] = probe_file_id
                return blank_row

        scores.to_csv(os.path.join(log_dir,"thresMets.csv"),sep="|",index=False)

        maxrow = localization_perimage_runner._localization_perimage_runner__get_row_metrics(scores,central_met,pd.Series(blank_metrics_defaults),sys_bin=sys_bin)
        #return the row with the max central metric
        #case no ground truth
        if (maxrow['TP'] + maxrow['FN'] == 0) or (maxrow['TN'] + maxrow['FP'] == 0):
            for i,x in maxrow.iteritems():
                maxrow.loc[i] = np.nan

        maxrow.loc['ProbeFileID'] = probe_file_id
        ref_mask.close()
        sys_mask.close()
        if sys_mask_name == "":
            os.system('rm {}'.format(white_mask_name))
        return maxrow


if __name__ == '__main__':
    #TODO: add options just to generate the perprobe scores, but for one pair of masks.
    import argparse
    parser = argparse.ArgumentParser(description="Score the spatial localization regions of the video probes.")
    parser.add_argument('-t','--task',default="manipulation",help="The task to score. So far, only 'manipulation' is available. Default: 'manipulation'.")
    

    args = parser.parse_args()
    print("Running perprobe_module.py...")
    
