import sys
import os
import h5py
import numpy as np
import pandas as pd
import cv2

hdf5_lib_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"lib")
sys.path.append(hdf5_lib_dir)
from hdf5_lib import open_video, play_mat_as_video, save_mat_as_video_file
from video_masks import video,video_mask,video_ref_mask,shift_intervals,contained_in_intervals
from masks import erode,dilate
#mask_metrics_dir="/Users/dfz/Desktop/Medifor/MediScoreV2-dfztemp/lib"
#sys.path.append(mask_metrics_dir)
import maskMetrics as mm
from detMetrics import Metrics as dmets

vtl_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"../tools/VideoTemporalLocalizationScorer")
sys.path.append(vtl_dir)
from VideoTemporalLocalizationScoring import VTLScorer
from TemporalVideoScoring import VideoScoring
from intervalcompute import IntervalCompute as IC

def frame_shape_is_valid(ref,sys):
    frameshape_ref = ref.shape[1:3]
    frameshape_sys = sys.shape[1:3]
    return frameshape_ref == frameshape_sys

def get_next_greatest_indices(ix1,ix2):
    ix1n2 = ix1.difference(ix2)
    last_ix_1n2 = []
    for i in ix1n2:
        last_ix = -1
        added_new_ix = False
        for j in ix2:
            if j > i:
                last_ix_1n2.append(last_ix)
                added_new_ix = True
                break
            else:
                last_ix = j

        if not added_new_ix:
            last_ix_1n2.append(ix2[-1])

    return last_ix_1n2

def update_dataframe_scores(df_to_update,df2):
    all_wo_this_ix = df_to_update.index.difference(df2.index)
    update_all_df = len(all_wo_this_ix) > 0
    if update_all_df:
        all_wo_this = df_to_update.loc[all_wo_this_ix]
        all_wo_this_update = df2.loc[get_next_greatest_indices(all_wo_this_ix,df2.index)]
        all_wo_this_update.index = all_wo_this_ix
        all_wo_this = all_wo_this + all_wo_this_update
    
    this_wo_all_ix = df2.index.difference(df_to_update.index)
    update_this_df = len(this_wo_all_ix) > 0
    if update_this_df:
        this_wo_all = df2.loc[this_wo_all_ix]
        this_wo_all_update = df_to_update.loc[get_next_greatest_indices(this_wo_all_ix,df_to_update.index)]
        this_wo_all_update.index = this_wo_all_ix
        this_wo_all = this_wo_all + this_wo_all_update

    if update_all_df:
        df_to_update.loc[all_wo_this_ix] = all_wo_this
    
    if update_this_df:
        df2.loc[this_wo_all_ix] = this_wo_all

    #pandas add everything on both sides.
    df_to_update = df_to_update.add(df2,fill_value = 0)
    
    return df_to_update

def gen_frame_no_score(ref_frame,sys_frame,eks,dks,ntdks,kern,pppns):
    met_counts = {}
    #TODO: get consistent counts per image localization scorer's treatment. The order doesn't affect scores, but it's convenient for display.
    ref_ns = ref_frame.get_boundary_no_score(eks=eks,dks=dks,kern=kern)
    met_counts["BNS"] = np.sum(ref_ns == 0)
    nt_ns = ref_frame.get_selective_no_score(eks=eks,ntdks=ntdks,kern=kern)
    met_counts["SNS"] = np.sum(nt_ns == 0)
    sys_ns = gen_sys_frame_no_score(sys_frame,pppns)
    met_counts["PNS"] = np.sum(sys_ns == 0)
    return ref_ns & sys_ns & nt_ns,met_counts

def check_framecounts(framecount_ref,framecount_sys,truncate,pad):
    if framecount_ref != framecount_sys:
        if not (truncate or pad):
            print("Error: Expected {} frames in system mask. Got {} frames.".format(framecount_ref,framecount_sys))
            exit(1)
        elif (framecount_ref < framecount_sys):
            print("Warning: Expected {} frames in system mask. Got {} frames. The system will be truncated to match the reference.".format(framecount_ref,framecount_sys))
    return 0

def get_confusion_measures(ref,sys,truncate=False,pad=True,temporal_gt_only=False,collars=None,eks=15,dks=11,ntdks=15,kern='box',pppns=-1,confusion_measures_list = ["TP","TN","FP","FN","N","BNS","SNS","PNS"]):
    """
    Description: Gets the confusion measures for the reference and system hdf5 "video" masks.
    Input params:
      - ref: the reference hdf5 mask object
      - sys: the system hdf5 mask object
      - ns: the no-score zone. Defaults to none. Implementation prone to change.
      - truncate: whether to truncate the system video if it is too long.
                         The system video with the same frame dimensions but different number of frames.
                         This behaves similarly to video temporal localization scoring.
      - pad: whether to pad the system video with white frames if it is too short. Default: True.
      - temporal_gt_only: Scores only on temporal ground-truth intervals.
    """
    confusion_mets_all = 0
    #compute the confusion measures dataframe
#    framecount_ref = ref.shape[0]
    framecount_ref = ref.framecount
    framecount_sys = sys.framecount
    if not frame_shape_is_valid(ref,sys):
        print("Error: Expected the shape of system mask to have dimensions {}. Got {}.".format(frameshape_ref,frameshape_sys))
        exit(1)

    check_framecounts(framecount_ref,framecount_sys,truncate,pad)
    ref_intervals = ref.compute_ref_intervals(eks=eks,dks=dks,ntdks=ntdks,sys=sys,nspx=pppns,kern=kern)

    #TODO: make amenable to parallel processing later?
    for f in range(framecount_ref):
        if pad and (framecount_ref > framecount_sys) and (f >= framecount_sys):
            if f == framecount_sys:
                #NOTE: this message only needs to print once.
                print("Warning: Expected {} frames in system mask. Got {} frames. Padding empty frames to the end of the system to match reference.".format(framecount_ref,framecount_sys))
#            print("Frame: {}".format(f))
            sysframe = sys.whiteframe.copy()
        else:
            sysframe = sys.get_frame(f)

        #add all pixels to BNS in dataframe and proceed. Get a blank dataframe.
        if (contained_in_intervals(f,ref.collars)):
            confusion_mets_this_frame = pd.DataFrame({"TP":0,"TN":0,"FP":0,"FN":0,"N":0,"BNS":sysframe.shape[0]*sysframe.shape[1],"SNS":0,"PNS":0},index=[-1])
            confusion_mets_all = update_dataframe_scores(confusion_mets_all,confusion_mets_this_frame)
            continue
        if (contained_in_intervals(f,ref.opt_out_frames)):
            confusion_mets_this_frame = pd.DataFrame({"TP":0,"TN":0,"FP":0,"FN":0,"N":0,"BNS":0,"SNS":0,"PNS":sysframe.shape[0]*sysframe.shape[1]},index=[-1])
            confusion_mets_all = update_dataframe_scores(confusion_mets_all,confusion_mets_this_frame)
            continue

        refframe = ref.get_frame(f)
        #TODO: a workaround, for now. Get from the frame intervals in the journal mask file instead of by frame.
        if temporal_gt_only and (np.sum(refframe) == 0):
            continue
        if ref.has_journal_data():
            refframe.insert_journal_data(ref.journal_data)

        #generate no score zone per frame here
        no_score,ns_counts = gen_frame_no_score(refframe,sysframe,eks,dks,ntdks,kern,pppns)

        confusion_mets_this_frame = mm.maskMetrics.confusion_mets_all_thresholds(refframe.get_binary(),sysframe,no_score)
        confusion_mets_this_frame.index = confusion_mets_this_frame['Threshold']
        confusion_mets_this_frame.drop(['Threshold'],1,inplace=True)
        for nsc in ns_counts:
            confusion_mets_this_frame[nsc] = ns_counts[nsc]

        confusion_mets_this_frame = confusion_mets_this_frame[confusion_measures_list]
        if confusion_mets_all is 0:
            confusion_mets_all = confusion_mets_this_frame
        else:
            #if threshold is not in frame df, add last threshold in frame df. Likewise for aggregate df. This update must be simultaneous. This is a preprocessing step.
            confusion_mets_all = update_dataframe_scores(confusion_mets_all,confusion_mets_this_frame)
            
    if confusion_mets_all is 0:
        print("No data obtained for reference.")
    else:
        confusion_mets_all['Threshold'] = confusion_mets_all.index
    return confusion_mets_all

#iterate over frames, sum the totals, and then divide to get the score
def score_GWL1(ref,sys,truncate=False,pad=True,temporal_gt_only=False,eks=15,dks=11,ntdks=15,kern='box',pppns=-1):
    framecount_ref = ref.framecount
    framecount_sys = sys.framecount
    check_framecounts(framecount_ref,framecount_sys,truncate,pad)

    difference_total = 0
    pixel_total = 0

    for f in range(framecount_ref):
        sysframe = sys.get_frame(f)
        refframe = ref.get_frame(f)
        #TODO: a workaround, for now. Get from the frame intervals in the journal mask file instead of by frame. The intent is to skip frames without GT.
        if temporal_gt_only and (np.sum(refframe) == 0):
            continue
        if ref.has_journal_data():
            refframe.insert_journal_data(ref.journal_data)
        
        no_score,ns_counts = gen_frame_no_score(refframe,sysframe,eks,dks,ntdks,kern,pppns)
        
        #this is more fundamental. Account for no-score zones more elegantly
        difference_total += np.sum(abs(refframe.get_binary().astype(int) - sysframe).astype(np.uint8) & no_score)
        pixel_total += np.sum(no_score)

    return float(difference_total)/pixel_total

def compute_metrics(df):
    df['MCC'] = df.apply(mm.maskMetrics.matthews,axis=1)
    df['NMM'] = df.apply(mm.maskMetrics.NimbleMaskMetric,axis=1)
    df['BWL1'] = df.apply(mm.maskMetrics.binaryWeightedL1,axis=1)
    df_cols = df.columns.values.tolist()
    if not ({"TPR","FPR"} < set(df_cols)):
        df["TPR"] = df["TP"].astype(float).divide(df["TP"] + df["FN"],fill_value=np.nan)
        df["FPR"] = df["FP"].astype(float).divide(df["FP"] + df["TN"],fill_value=np.nan)
    df_copy = df.copy().reset_index(drop=True)
    df["AUC"] = dmets.compute_auc(df_copy["FPR"],df_copy["TPR"])
    df["EER"] = dmets.compute_eer(df_copy["FPR"],1 - df_copy["TPR"])

    return df

#call Timothee's Temporal Localization scorer here
def score_temporal_metrics(ref_mask,sys_mask,collars=None,truncate=False,eks=15,dks=11,ntdks=15,nspx=-1,kern='box'):
    temporal_scorer = VideoScoring()
#    ref_intervals = shift_intervals(ref_mask.compute_ref_intervals(eks=eks,dks=dks,ntdks=ntdks,sys=sys_mask,nspx=nspx,kern=kern),shift=1)
    ref_intervals = ref_mask.compute_ref_intervals(eks=eks,dks=dks,ntdks=ntdks,sys=sys_mask,nspx=nspx,kern=kern)
    ref_intervals = np.array(ref_intervals)
    frame_count = ref_mask.framecount

    sys_thresholds = sys_mask.get_all_thresholds()
    SNS = sys_mask.get_selective_no_score_frames()

    df = pd.DataFrame(index=sys_thresholds,columns=["MCC","TP","TN","FP","FN"])
    confusion_mets = ["TP","TN","FP","FN"]

    global_range = np.array([1,ref_mask.framecount])

    #compute here and store in a dataframe
    for t in sys_thresholds:
#        sys_intervals = shift_intervals(sys_mask.compute_sys_intervals(t),shift=1)
        sys_intervals = sys_mask.compute_sys_intervals(t)
        if not sys_intervals:
            sys_intervals = [[]]
        sys_intervals = np.array(sys_intervals)
        
        collar_intervals = IC.compute_collars(ref_intervals, collars, crop_to_range = global_range) if collars is not None else None
        if truncate:
            sys_intervals = IC.truncate(sys_intervals,frame_count)
        
        if collars is not None or SNS is not None:
            (confusion_vector,_), all_intervals, all_interval_in_seq_array = temporal_scorer.compute_confusion_map(ref_intervals,
                                                                                                               sys_intervals,
                                                                                                               global_range,
                                                                                                               collars = collar_intervals,
                                                                                                               SNS = SNS
                                                                                                               )
        else:
            confusion_vector, all_intervals, all_interval_in_seq_array = temporal_scorer.compute_confusion_map(ref_intervals,
                                                                                                               sys_intervals,
                                                                                                               global_range,
                                                                                                               collars = collar_intervals,
                                                                                                               SNS = SNS
                                                                                                               )
        counts = temporal_scorer.count_confusion_value(all_intervals,confusion_vector)
        mcc = temporal_scorer.compute_MCC(*[counts[v] for v in ["TP", "TN", "FP", "FN"]])
        df.loc[t]['MCC'] = mcc
        for c in confusion_mets:
            df.loc[t][c] = counts[c]

    return df


#TODO: moved section ends here

#TODO: move these to a unit test. Alternatively since this is the unit test, keep them here?
hdf5_lib_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")
sys.path.append(hdf5_lib_dir)
from gen_hdf5 import gen_mask,gen_mask_from_ref,rm_mask,stamp_vid,gen_color_results,gen_sys_frame_no_score

vidprobe_sample_dir = "../data/test_suite/videoSpatialLocalizationScorerTests/0c5fc5f508229a3f47e589b18acefc25"

#TODO: test for correctness? Print the GT pos, GT neg, and total pixels.
#NOTE: test cases below
def perfect_test():
    #test with one reference mask and then the other
    ref_mask_path = os.path.join(vidprobe_sample_dir,"0c5fc5f508229a3f47e589b18acefc25/duplicated_orange_dude_Final_added_colored_mask_0.0_bp.hdf5")
    sys_mask_path = os.path.join(vidprobe_sample_dir,"0c5fc5f508229a3f47e589b18acefc25/duplicated_orange_dude_Final_added_colored_mask_0.0.hdf5")
    ref_mask = video_ref_mask(ref_mask_path)
    sys_mask = video_mask(sys_mask_path)
#    ref_pointer,ref_dset = open_video(ref_mask_path)
    print "Video scoring results perfect:"
    mets = compute_metrics(get_confusion_measures(ref_mask,sys_mask,truncate=True))
    mets['GWL1'] = score_GWL1(ref_mask,sys_mask,truncate=True)
    print mets
    assert len(mets[["N","BNS","SNS","PNS"]].drop_duplicates()) == 1, "Error: N, BNS, SNS, PNS seem to vary across thresholds."
    print score_temporal_metrics(ref_mask,sys_mask,collars=None,truncate=True)
    ref_mask.close()
    sys_mask.close()

def imperfect_truncation_test():
    ref_mask_path = os.path.join(vidprobe_sample_dir,"0c5fc5f508229a3f47e589b18acefc25/duplicated_orange_dude_Final_added_colored_mask_0.0_bp.hdf5")
    sys_mask_path = os.path.join(vidprobe_sample_dir,"0c5fc5f508229a3f47e589b18acefc25/stabilized_duplicated_orange_dude_mask_0.0.hdf5")
    ref_mask = video_ref_mask(ref_mask_path)
    sys_mask = video_mask(sys_mask_path)
    print "Video scoring results imperfect. Involves truncation of the system output:"
    mets = compute_metrics(get_confusion_measures(ref_mask,sys_mask,truncate=True))
    mets['GWL1'] = score_GWL1(ref_mask,sys_mask,truncate=True)
    print mets
    assert len(mets[["N","BNS","SNS","PNS"]].drop_duplicates()) == 1, "Error: N, BNS, SNS, PNS seem to vary across thresholds."
    print score_temporal_metrics(ref_mask,sys_mask,collars=None,truncate=True)
    ref_mask.close()
    sys_mask.close()

def white_mask_test():
    ref_mask_path = os.path.join(vidprobe_sample_dir,"0c5fc5f508229a3f47e589b18acefc25/duplicated_orange_dude_Final_added_colored_mask_0.0_bp.hdf5")
    ref_mask = video_ref_mask(ref_mask_path)
    sys_mask_path = os.path.join(vidprobe_sample_dir,'white_mask.hdf5')
    gen_mask_from_ref(sys_mask_path,ref_mask_name=ref_mask_path)
    sys_mask = video_mask(sys_mask_path)
    print "Video scoring results white mask:"
    mets = compute_metrics(get_confusion_measures(ref_mask,sys_mask,truncate=True))
    print mets
    assert len(mets[["N","BNS","SNS","PNS"]].drop_duplicates()) == 1, "Error: N, BNS, SNS, PNS seem to vary across thresholds."
    print score_temporal_metrics(ref_mask,sys_mask,collars=None,truncate=True)
    ref_mask.close()
    sys_mask.close()
    rm_mask(sys_mask_path)

def padded_mask_test():
    ref_mask_path = os.path.join(vidprobe_sample_dir,"0c5fc5f508229a3f47e589b18acefc25/duplicated_orange_dude_Final_added_colored_mask_0.0_bp.hdf5")
    #test case. One mask that requires padding. (See half of one mask.)
    ref_mask = video_ref_mask(ref_mask_path)
    sys_mask_path = os.path.join(vidprobe_sample_dir,'black_part_mask.hdf5')
    gen_mask_from_ref(sys_mask_path,frame_scale=0.7,value=0,ref_mask_name=ref_mask_path)
    sys_mask = video_mask(sys_mask_path)
    print "Video scoring results black mask with 0.7 length:"
    mets = compute_metrics(get_confusion_measures(ref_mask,sys_mask,truncate=True))
    print mets
    assert len(mets[["N","BNS","SNS","PNS"]].drop_duplicates()) == 1, "Error: N, BNS, SNS, PNS seem to vary across thresholds."
    print score_temporal_metrics(ref_mask,sys_mask,collars=None,truncate=True)
    ref_mask.close()
    sys_mask.close()
    rm_mask(sys_mask_path)

def gen_moving_animated_mask(ref_mask_path,start_frame = 90,end_frame=210,n_layer=1):
    #the manipulation in these frames should be more or less changing. Think disc moving in a square about the video.
    path_to_mask="{}/masks".format(start_frame)
    gen_mask(ref_mask_path,[480,600],end_frame - start_frame + 1,path_to_mask=path_to_mask,value=0,n_layer=n_layer)
    stamp_vid(ref_mask_path,0,end_frame - start_frame,path_to_mask=path_to_mask,value=1)

def middle_manipulation_test():
    ref_mask_path = os.path.join(vidprobe_sample_dir,"sample_ref_mask.hdf5")
    sys_mask_path = os.path.join(vidprobe_sample_dir,"my_sys_mask.hdf5")

    #play mat as video
    fps = 30
    t = 10
    total_frames = fps*t
    manip_time = [3,7]
    manip_frames = [ t*fps for t in manip_time ]
    gen_moving_animated_mask(ref_mask_path,start_frame = manip_frames[0],end_frame=manip_frames[1])
    ref_mask = video_ref_mask(ref_mask_path)
#    ref_pointer,ref_dset = open_video(ref_mask_path)
#    save_mat_as_video_file('myvid.mov',ref_dset)
#    for f in range(ref_dset.shape[0]):
#        print np.unique(ref_dset[f,:,:])

    #generate system output for the above video
    sys_time = [2,5]
    sys_frames = [ t*fps for t in sys_time]
    gen_mask(sys_mask_path,[480,600],sys_frames[1])
    sys_pointer,sys_dset = open_video(sys_mask_path,mode='r+')
    for f in range(sys_frames[0],sys_frames[1]):
        sys_dset[f,:,:] = 0*sys_dset[f,:,:]
    sys_pointer.close()

    color_mask_path = os.path.join(vidprobe_sample_dir,'color_vid_results.hdf5')
    sys_mask = video_mask(sys_mask_path)
    gen_color_results(color_mask_path,ref_mask,sys_mask)
    color_vid = video(color_mask_path)
    #generate color mask to visualize scoring for the above video.
    color_vid.save_as_video(os.path.join(vidprobe_sample_dir,'color_vid_results.mov'))
    color_vid.close()

    print "Middle manipulation scoring results:"
    mets = compute_metrics(get_confusion_measures(ref_mask,sys_mask,truncate=True))
    assert len(mets[["N","BNS","SNS","PNS"]].drop_duplicates()) == 1, "Error: N, BNS, SNS, PNS seem to vary across thresholds."
    print mets
    print compute_metrics(get_confusion_measures(ref_mask,sys_mask,truncate=True,temporal_gt_only=True))
    print score_temporal_metrics(ref_mask,sys_mask,collars=None,truncate=True)
    ref_mask.close()
    sys_mask.close()
    rm_mask(ref_mask_path)
    rm_mask(sys_mask_path)

def gray_manipulation_test():
    #generation stage
    ref_mask_path = os.path.join(vidprobe_sample_dir,"sample_ref_mask.hdf5")
    gen_moving_animated_mask(ref_mask_path)
    #grayscale system test case    
    sys_mask_path = os.path.join(vidprobe_sample_dir,"my_sys_mask.hdf5")
    fps = 30
    t = 10
    sys_time = [2,5]
    sys_frames = [ t*fps for t in sys_time]
    gen_mask(sys_mask_path,[480,600],sys_frames[1])
    sys_pointer,sys_dset = open_video(sys_mask_path,mode='r+')

    delumination_increment = 255/(sys_frames[1] - sys_frames[0])
    i = 0
    frame_dims = [sys_dset.shape[1],sys_dset.shape[2]]
    for f in range(sys_frames[0],sys_frames[1]):
        i = i + 1
        sys_dset[f,:,:] = (255 - i*delumination_increment)*np.ones(frame_dims,dtype=np.uint8)

    save_mat_as_video_file(os.path.join(vidprobe_sample_dir,'gs_mask.mov'),sys_dset)
    sys_pointer.close()

    #scoring stage
    ref_mask = video_ref_mask(ref_mask_path)
#    ref_pointer,ref_dset = open_video(ref_mask_path)
    sys_mask = video_mask(sys_mask_path)
    
    #Erosion and dilation frame by frame with default samples. Make a videomask object.
    eks = 15
    dks = 11
    ntdks = 15
    #test case. Pixel optout.
    pppns = 255 - 20

    print "Grayscale system scoring results:"
    mets = compute_metrics(get_confusion_measures(ref_mask,sys_mask,truncate=True,eks=eks,dks=dks,ntdks=ntdks,pppns=pppns))
    print mets
    assert len(mets[["N","BNS","SNS","PNS"]].drop_duplicates()) == 1, "Error: N, BNS, SNS, PNS seem to vary across thresholds."
    th_max = int(mets['MCC'].idxmax())
    print th_max
    print mets.loc[th_max]
    print score_temporal_metrics(ref_mask,sys_mask,collars=None,truncate=True,eks=eks,dks=dks,ntdks=ntdks,nspx=pppns)

#    bin_mask_path = 'my_sys_mask_bin.hdf5'
#    print th_max
#    sys_mask.gen_threshold_mask(bin_mask_path,th=th_max)

#    sys_t_mask = video_mask(bin_mask_path)
#    sys_t_mask.save_as_video('my_sys_mask_bin.mov')
#    sys_t_mask.close()

    color_mask_path = os.path.join(vidprobe_sample_dir,'color_vid_results.hdf5')
    gen_color_results(color_mask_path,ref_mask,sys_mask,th=th_max,eks=eks,dks=dks,ntdks=ntdks,pppns=pppns)
    color_pointer,color_dset = open_video(color_mask_path)
    #generate color mask to visualize scoring for the above video.
    save_mat_as_video_file(os.path.join(vidprobe_sample_dir,'color_vid_results.mov'),color_dset)
    color_pointer.close()

    sys_mask.close()
    rm_mask(ref_mask_path)
    rm_mask(sys_mask_path)
#    rm_mask(bin_mask_path)

#score on two BitPlanes, whole and selective.
def multi_bitplane_tests():
    #generate ref mask
    orig_ref_mask_path = os.path.join(vidprobe_sample_dir,"0c5fc5f508229a3f47e589b18acefc25/duplicated_orange_dude_Final_added_colored_mask_0.0_bp.hdf5")
    ref_mask_path = os.path.join(vidprobe_sample_dir,'sample_ref_mask.hdf5')

    #get a copy of Eric's mask. Add the animation. And let it roll.
    path_to_mask = '0/masks'
    os.system('cp {} {}'.format(orig_ref_mask_path,ref_mask_path))
    ref_tmp = video(ref_mask_path,mode='r+')
    for f in range(100,121):
        ref_tmp.fpointer[path_to_mask][f,:,:] = np.zeros(ref_tmp.shape,dtype=np.uint8)
    ref_tmp.close()
    stamp_vid(ref_mask_path,30,120,path_to_mask=path_to_mask,value=2,opaque=False)
    
    #generate sys mask
    sys_mask_path = os.path.join(vidprobe_sample_dir,"sys_mask.hdf5")
    gen_mask_from_ref(sys_mask_path,path_to_mask=path_to_mask,value=255,frame_scale=.9,ref_mask_name = ref_mask_path)
    sys_pointer,sys_dset = open_video(sys_mask_path,path_to_mask=path_to_mask,mode='r+')
    delumination_increment = 2
    i = 0
    frame_dims = [sys_dset.shape[1],sys_dset.shape[2]]
    for f in range(100):
        i = i + 1
        sys_dset[f,:,:] = (255 - i*delumination_increment)*np.ones(frame_dims,dtype=np.uint8)
    sys_pointer.close()

    journal_data = pd.DataFrame({"JournalName":"multi_bitplane_tests","StartNodeID":['bp_1','bp_2'],"EndNodeID":['bp_2','bp_3'],"Operation":["PasteSplice","Blur"],"BitPlane":[1,2],"Sequence":[2,1],"Evaluated":["Y","Y"]})
    #begin tests
    ref_mask = video_ref_mask(ref_mask_path)
    ref_mask.insert_journal_data(journal_data,"Evaluated")
    sys_mask = video_mask(sys_mask_path)

    eks = 11
    dks=15
    ntdks=15   
    #score on all
    print "Score on all"
    mets = compute_metrics(get_confusion_measures(ref_mask,sys_mask,truncate=True,eks=eks,dks=dks,ntdks=ntdks))
    print mets
    assert len(mets[["N","BNS","SNS","PNS"]].drop_duplicates()) == 1, "Error: N, BNS, SNS, PNS seem to vary across thresholds."
    th_max = int(mets['MCC'].idxmax())
    print th_max
    print mets.loc[th_max]
    print score_temporal_metrics(ref_mask,sys_mask,collars=None,truncate=True,eks=eks,dks=dks,ntdks=ntdks)
    #gen color mask
    color_mask_path = os.path.join(vidprobe_sample_dir,'color_vid_results_allbit.hdf5')
    gen_color_results(color_mask_path,ref_mask,sys_mask,th=th_max,eks=eks,dks=dks,ntdks=ntdks)
    color_pointer,color_dset = open_video(color_mask_path)
    #generate color mask to visualize scoring for the above video.
    save_mat_as_video_file(os.path.join(vidprobe_sample_dir,'color_vid_results_allbit.mov'),color_dset)
    color_pointer.close()
    rm_mask(color_mask_path)

    #score on 2 only
    journal_data2 = journal_data.copy()
    journal_data2.loc[0,"Evaluated"] = "N"
    ref_mask.insert_journal_data(journal_data2,"Evaluated")

    print "Score on BitPlane 2 only."
    mets = compute_metrics(get_confusion_measures(ref_mask,sys_mask,truncate=True,eks=eks,dks=dks,ntdks=ntdks))
    print mets
    assert len(mets[["N","BNS","SNS","PNS"]].drop_duplicates()) == 1, "Error: N, BNS, SNS, PNS seem to vary across thresholds."
    th_max = int(mets['MCC'].idxmax())
    print th_max
    print mets.loc[th_max]
    print score_temporal_metrics(ref_mask,sys_mask,collars=None,truncate=True,eks=eks,dks=dks,ntdks=ntdks)
    #gen color mask
    color_mask_path = os.path.join(vidprobe_sample_dir,'color_vid_results_bit2.hdf5')
    gen_color_results(color_mask_path,ref_mask,sys_mask,th=th_max,eks=eks,dks=dks,ntdks=ntdks)
    color_pointer,color_dset = open_video(color_mask_path)
    #generate color mask to visualize scoring for the above video.
    save_mat_as_video_file(os.path.join(vidprobe_sample_dir,'color_vid_results_bit2.mov'),color_dset)
    color_pointer.close()
    rm_mask(color_mask_path)
    
    ref_mask.close()
    sys_mask.close()
    rm_mask(ref_mask_path)
    rm_mask(sys_mask_path)

#TODO: multi-layer bitplane: 1, 2, and 9
def multi_layer_bp_video_test():
    orig_ref_mask_path = os.path.join(vidprobe_sample_dir,"0c5fc5f508229a3f47e589b18acefc25/duplicated_orange_dude_Final_added_colored_mask_0.0_bp.hdf5")
    ref_mask = video_ref_mask(orig_ref_mask_pathi)

    frame_dims = ref_mask.shape
    framecount = ref_mask.framecount

    ref_mask.close()

    ref_mask_path = os.path.join(vidprobe_sample_dir,'sample_ref_mask.hdf5')
    
    eks = 11
    dks=15
    ntdks=15

    #TODO: get the dims of the original and create a new 2-layer reference mask.

    #TODO: stamp it accordingly with different objects at various times. Even a complete overlay is okay.

    #TODO: generate grayscale system mask and score



if __name__ == '__main__':
    perfect_test()
    imperfect_truncation_test()
    white_mask_test()
    padded_mask_test()
    middle_manipulation_test()
    gray_manipulation_test()
    multi_bitplane_tests()

    #TODO: test case. Masks that require alignment along frames. Makes scoring and implementation more complicated. Not available yet,
    # but the middle_manipulation_test seems to be working for this test given the current object's implementations and my own design.
    


