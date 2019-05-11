import sys
import os
import numpy as np
import pandas as pd
import cv2
import ast

from count_colors_f import bp_sets_from_mask
from video_masks import video,video_mask,video_ref_mask

try:
    pos_avi_ratio_const = cv2.cv.CV_CAP_PROP_POS_AVI_RATIO
except AttributeError:
    pos_avi_ratio_const = cv2.CAP_PROP_POS_AVI_RATIO

try:
    pos_msec_const = cv2.cv.CV_CAP_PROP_POS_MSEC
except AttributeError:
    pos_msec_const = cv2.CAP_PROP_POS_MSEC

try:
    width_const = cv2.cv.CV_CAP_PROP_FRAME_WIDTH
    height_const = cv2.cv.CV_CAP_PROP_FRAME_HEIGHT
except AttributeError:
    width_const = cv2.CAP_PROP_FRAME_WIDTH
    height_const = cv2.CAP_PROP_FRAME_HEIGHT

def read_csv(csv_name,sep="|"):
    return pd.read_csv(csv_name,sep="|",header=0,na_filter=False,index_col=False)

def typecheck_interval(interval,mode):
    if mode == 'Time':
        intvl_type=float
    elif mode == 'Frame':
        intvl_type=int
    try:
        interval_str = [str(x) for x in interval]
        intvl_type(interval_str[0])
        intvl_type(interval_str[1])
        return 0
    except:
        return 1

def append_str(s1,s2,delim="\n"):
    return delim.join([s1,s2])

#interval evaluation
def interval_check(vrow,max_frame_number,vfield,precision=0.000001):
    """
    * Checks if the list of intervals passed is valid.
    """
    interval_list = 0
    errmsg = ''
    return_string = '%s#%s'
    if 'Time' in vfield:
        mode = 'Time'
        dtype = float
    elif 'Frame' in vfield:
        mode = 'Frame'
        dtype = int

    intervals = vrow[vfield]
    probe_ID = vrow['ProbeFileID']
    operation = vrow['Operation']
    try:
        interval_list = ast.literal_eval(intervals)
    except:
        errmsg = "Error: ProbeFileID {} encountered issue with parsing interval {} in {}. Input must be a string.".format(probe_ID,intervals,vfield)
        vrow['errflag'] = 1
        vrow['errmsg'] = errmsg
        return vrow
#        return return_string % (1,errmsg)

    #if empty list, passes
    if interval_list == []:
        return vrow
#        return return_string % (0,'')

    #ensure that each of the intervals is a legitimate interval (i.e. a_n <= b_n)
    intervalflag = 0
    typeflag = 0
    cleared_interval_list = []
    for interval in interval_list:
        if len(interval) != 2:
            errmsg = "Error: ProbeFileID {}, Operation {}: List {} in {} is not a valid interval.".format(probe_ID,operation,interval,vfield)
            intervalflag = 1
            continue
        this_typeflag = typecheck_interval(interval,mode)
        typeflag = typeflag | this_typeflag
        if this_typeflag == 1:
            errmsg = '\n'.join([errmsg,"Error: ProbeFileID {}, Operation {}; Interval {} in {} does not have {} fields.".format(probe_ID,operation,interval,vfield,dtype)])

        if interval[0] > interval[1]:
            errmsg = '\n'.join([errmsg,"Error: ProbeFileID {}, Operation {}: Interval {} in {} must be formatted as a valid interval [an,bn], where an <= bn".format(probe_ID,operation,interval,vfield)])
            intervalflag = 1
        #different start frame for time and frame numbers
        if mode=='Frame':
            min_frame_number=1
        elif mode=='Time':
            min_frame_number=0
        if (interval[0] < min_frame_number) or (interval[1] > max_frame_number):
            #deviation accounted for
            if (interval[0] >= min_frame_number) and (interval[1] < max_frame_number + precision):
                errmsg = "\n".join([errmsg,"Warning: ProbeFileID {}, Operiation {}: Interval {} in {} is slightly out of bounds with precision {}. The max interval for this video is {}. The deviation is: {}.".format(probe_ID,operation,interval,vfield,precision,[min_frame_number,max_frame_number],interval[1] - max_frame_number)])
            else:
                errmsg = '\n'.join([errmsg,"Error: ProbeFileID {}, Operation {}: Interval {} in {} is out of bounds. The max interval for this video is {}.".format(probe_ID,operation,interval,vfield,[min_frame_number,max_frame_number])])
                intervalflag = 1

        if intervalflag == 0:
            #each of the intervals in each list of intervals must be disjoint (except for endpoints)
            if len(cleared_interval_list) == 0:
                cleared_interval_list.append(interval)
                continue

            for intvl in cleared_interval_list:
                #ensure if a singularity that it only coincides with an endpoint
                if interval[0] == interval[1]:
                    if (interval[0] > intvl[0]) and (interval[0] < intvl[1]):
                        errmsg = '\n'.join([errmsg,"Error: ProbeFileID {}, Operation {}: Singular frame {} is contained in interval {} in field {} and is not one of its endpoints. Please revise the dataset.".format(probe_ID,operation,interval,intvl,vfield)])
                        intervalflag = 1
                else:
                    if ((interval[0] >= intvl[0]) and (interval[0] <= intvl[1])) or ((interval[1] <= intvl[1]) and (interval[1] >= intvl[0])):
                        #coincides with at least an endpoint
                        errmsg = '\n'.join([errmsg,"Error: ProbeFileID {}, Operation {}: Interval {} intersects with interval {} for field {}. Please revise the dataset.".format(probe_ID,operation,interval,intvl,vfield)])
                        intervalflag = 1
            cleared_interval_list.append(interval)

    vrow['errflag'] = intervalflag | typeflag
    vrow['errmsg'] = errmsg
    return vrow
#    return return_string % (intervalflag,errmsg)

def get_max_time(vfile):
    vfile.set(pos_avi_ratio_const,1)
    return vfile.get(pos_msec_const)

def video_length_check(row,ref_dir,journal_info,precision=0.000001):
    """
    * Description: wrapper for video and interval checking.
    """
    vfilename = os.path.join(ref_dir,row['ProbeFileName'])
    vfile = cv2.VideoCapture(vfilename)
    if not vfile.isOpened():
        row['errmsg'] = "Error: {} is unreadable with OpenCV VideoCapture. Additionally check if the file is present.".format(vfilename)
        row['errflag'] = 1
        return row

#    max_frame_number = int(vfile.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    try:
        max_frame_number = row['FrameCount']
    except:
        framecount = 0
        while True:
            (grabbed,frame) = vfile.read()
            if not grabbed:
                (grabbed2,frame2) = vfile.read()
                if not grabbed2:
                    break
            framecount += 1
#            print "File: {}. Frame: {}".format(vfilename,framecount)
        max_frame_number = framecount
#    max_time = float(vfile.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))/vfile.get(cv2.cv.CV_CAP_PROP_FPS) 
    max_time = get_max_time(vfile)
    vfile.release()

    #iterate over the journal files for a sub-journal
    sub_journal = journal_info.query("ProbeFileID == '{}'".format(row['ProbeFileID']))
    if sub_journal.shape[0] == 0:
        return row

    interval_columns = ['VideoFrame','VideoTime','AudioFrame','AudioTime']

    #interval_check above interval_columns
    for vfield in interval_columns:
        if 'Frame' in vfield:
            mode='Frame'
            max_number = max_frame_number
        elif 'Time' in vfield:
            mode='Time'
            max_number = max_time

        #iterate over sub_journal
        sub_journal.loc[:,"errflag"] = 0
        sub_journal.loc[:,"errmsg"] = ""
        sub_journal = sub_journal.apply(interval_check,axis=1,max_frame_number=max_number,vfield=vfield,precision=precision,reduce=False)

#        sub_journal['errflag'],sub_journal['errmsg'] = errflagmsg.str.split('#',1).str
        row['errflag'] = row['errflag'] + sub_journal['errflag'].astype(int).sum()
        row['errmsg'] = '\n'.join([row['errmsg']] + sub_journal['errmsg'].tolist())
    return row

#checks the video's spatial mask content.

def video_framecount_check(vfilename,framecount):
    flag = 0
    msg = []
    vmask = video(vfilename)
    vfcount = vmask.framecount
    ext = vfilename.split('.')[-1]
    miss_list = []
    if ext == 'hdf5':
        attrs = vmask.fpointer.attrs.keys()
        if not (('end_frame' in attrs) and ('start_frame' in attrs)):
            for f in ['start_frame','end_frame']:
                if f not in attrs:
                     miss_list.append(f)
        if len(miss_list) > 0:
            flag = 1
            msg.append("Error: Mask {} expects attributes {}. These attributes are not found.".format(vfilename,", ".join(miss_list)))
            vmask.close()
            return flag,"\n".join(msg)

        start_frame = vmask.fpointer.attrs['start_frame']
        end_frame = vmask.fpointer.attrs['end_frame']
    
        start_frames = [int(k) for k in vmask.fpointer.keys() if k.isdigit()]
        max_start_frame = max(start_frames)
        max_frame = max_start_frame + vmask.fpointer["{}/masks".format(max_start_frame)].shape[0]
        if max_frame >= end_frame:
            flag = 1
            msg.append("Error: Mask {}'s maximum frame {} goes beyond the recorded end frame {}.".format(vfilename,max_frame,end_frame,end_frame))

        if end_frame - start_frame + 1 != framecount:
            flag = 1
            msg.append("Error: Mask {}'s frame metadata does not agree with the index file: Expected: {}. Got: {}.".format(vfilename,framecount,end_frame - start_frame + 1))
    vmask.close()

    if vfcount != framecount:
        flag = 1
        msg.append("Error: Video file {} has {} frames. Expected {} frames.".format(vfilename,vfcount,framecount))

    return flag,"\n".join(msg)

#check the video's spatial (frame) dimensions
def video_dim_check(vfilename,width,height):
    vfile = video(vfilename)
    if os.path.basename(vfilename).split('.')[-1] == 'hdf5':
        vid_w = vfile.shape[1]
        vid_h = vfile.shape[0]
    else:
        vid_w = vfile.shape[0]
        vid_h = vfile.shape[1]
    vfile.close()
    if not ((vid_w == width) and (vid_h == height)):
        return 1,"Error: Dimensional mismatch for video {}. Expected dimensions {}. Got {}.".format(vfile.name,(width,height),(vid_w,vid_h))
    return 0,""

#checks that the manipulations in the mask correspond to their temporal (spatial localizable) intervals.
def video_mask_interval_check(vfilename,journal_data):
    bp_list = journal_data["BitPlane"].unique().tolist()
    if "" in bp_list:
        bp_list.remove("")
    flag = 0
    msg = []
    vfile = video_ref_mask(vfilename)
    for bp in bp_list:
        vframes_ref = journal_data.query("BitPlane == '{}'".format(bp))["VideoFrame"]
        if len(vframes_ref) > 1:
            flag = 1
            msg.append("Error: File {} for BitPlane {} got multiple lists of frames: {}.".format(vfilename,bp,journal_data.query("BitPlane == '{}'".format(bp))))
#            msg.append("Error: File {} for BitPlane {} got multiple lists of frames: {}.".format(vfilename,bp,vframes_ref.tolist()))
#            continue
        vframes_ref = np.array(ast.literal_eval(vframes_ref.iloc[0]))
        vframes_ref.sort(axis=0)
        v_ivl = np.array(vfile.compute_ref_intervals(bitplane=int(bp)))
        v_ivl.sort(axis=0)
        #shift by 1 to match the frames in the reference.
        v_ivl_shift = v_ivl + 0
        #v_ivl == vframes_ref?
        if not np.array_equal(v_ivl_shift,vframes_ref):
            flag = 1
            msg.append("Error: File {} for BitPlane {} got differing lists of frames from reference. Expected: {}. Got: {}.".format(vfilename,bp,str(vframes_ref).replace('\n',','),str(v_ivl_shift)).replace('\n',','))
    vfile.close()

    return flag,"\n".join(msg)

#checks if the mask is grayscale
def video_mask_check(vmaskname):
    vfile = video_mask(vmaskname)
    status = 0
    colorlist = set()
    msgs = []
    for i,f in enumerate(vfile):
        if len(f.shape) > 2:
            if f.shape[2] > 1:
                msgs.append("Error: {}'s frame at frame {} needs to be 2-dimensional. The number of layers is {}.".format(vmaskname,i+1,f.shape[2]))
                status = 1
        pxlist = np.unique(f)
        colorlist = colorlist.union(set(pxlist))
        
    for c in colorlist:
        if not ((c >= 0) and (c <= 255)):
            msgs.append("Error: the pixel value {} was found in mask {}. It should be in the interval [0,255].".format(c,vmaskname))
            return 1,"\n".join(msgs)

    return status,"\n".join(msgs)

#Does the BitPlane check.
def video_mask_bitplane_check(vfilename,journal_data):
    bp_vid = video_ref_mask(vfilename)
    flag = 0
    msg = []
    probe_file_id = journal_data["ProbeFileID"].unique()[0]
    bp_set = set()

    #iterate over video and get all BitPlanes
    for f in bp_vid:
        bp_list = bp_sets_from_mask(f,bitPlaces=True)
        for bp_g in bp_list:
            bp_set = bp_set.union(bp_g.split(','))
    bp_vid.close()
    if '' in bp_set:
        bp_set.remove('')
    msg.append("Video spatial mask {} contains BitPlanes: {}.".format(vfilename,bp_set))
    bp_vid.close()

    #check reference files for consonance with the journal file
    j_bp_set = set(journal_data["BitPlane"].tolist())
    msg.append("Journal for ProbeFileID {} contains BitPlanes: {}.".format(probe_file_id,j_bp_set))
    if '' in j_bp_set:
        j_bp_set.remove('')
    bp_jumc = j_bp_set - bp_set
    bp_mujc = bp_set - j_bp_set

    if bp_jumc > set():
        msg.append("Error: Journal for ProbeFileID {} contains BitPlanes not present in video mask: {}".format(probe_file_id,bp_jumc))
        flag = 1
    if bp_mujc > set():
        msg.append("Error: Video mask contains BitPlanes not present in journal for ProbeFileID {}: {}".format(probe_file_id,bp_mujc))
        flag = 1
    
    return flag,"\n".join(msg)

stringency_dict = {0:"Skip and run detection and consistency checks only, skipping temporal interval checks and all spatial (mask) checks.",
                   1:"Forces all temporal interval checks and detection and consistency checks. This will also skip all video spatial checks.",
                   2:"Forces all temporal, detection, and consistency checks, and skips spatial only if VideoTaskDesignation does not exist in the index file. (Default.)",
                   3:"Forces all temporal, detection, and consistency checks. This will do spatial checks if VideoTaskDesignation exists anywhere in the relevant reference or index files: not just in the index file.",
                   4:"Forces all temporal, detection, and consistency checks. Forces spatial checks if an HDF5 mask exists, regardless of VideoTaskDesignation."}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Validate the video reference files and ensure that they conform to data specifications.",formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--refDir',type=str,
        help="Reference data directory path.",metavar='valid path to directory')
    parser.add_argument('-r','--inRef',type=str,
        help="Reference csv file name, relative to --refDir.",metavar='valid path to file')
    parser.add_argument('-rjj','--refJournalJoin',type=str,
        help="Reference journal join csv file name, relative to --refDir.",metavar='valid path to file')
    parser.add_argument('-rjm','--refJournalMask',type=str,
        help="Reference journal mask csv file name, relative to --refDir.",metavar='valid path to file')
    parser.add_argument('-x','--inIndex',type=str,
        help="Optional reference index file, relative to --refDir",metavar='valid path to file')
    parser.add_argument('--precision',type=float,default=0.000001,
        help="Precision to check for when doing interval checks. Deviations that do not exceed this percentage (e.g. end_of_interval < total_time + precision) will not trigger an error.")
    parser.add_argument('--stringency',type=int,default=2,
        help="Pass one of numbers 0 to 4 to force video checks with some stringency. Default: 2.\r\
\t 0: Skip and run detection and consistency checks only, skipping temporal interval checks and all spatial (mask) checks.\r\
\t 1: Forces all temporal interval checks and detection and consistency checks. This will also skip all video spatial checks.\r\
\t 2: Forces all temporal, detection, and consistency checks, and skips spatial only if VideoTaskDesignation does not exist in the index file. (Default.)\r\
\t 3: Forces all temporal, detection, and consistency checks. This will do spatial checks if VideoTaskDesignation exists anywhere in the relevant reference or index files: not just in the index file.\r\
\t 4: Forces all temporal, detection, and consistency checks. Forces spatial checks if an HDF5 mask exists, regardless of VideoTaskDesignation.")
    
    args = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        exit(0)
    
    print("Beginning video dataset validation for {} with Time precision {} and stringency {}: {}.".format(args.refDir,args.precision,args.stringency,stringency_dict[args.stringency]))
    ref_pfx = os.path.join(os.path.dirname(args.inRef),os.path.basename(args.inRef).split('.')[0])
    if args.refJournalJoin is None:
        args.refJournalJoin = '-'.join([ref_pfx,'probejournaljoin.csv'])
    if args.refJournalMask is None:
        args.refJournalMask = '-'.join([ref_pfx,'journalmask.csv'])
    
    refMain = read_csv(os.path.join(args.refDir,args.inRef))
    journaljoin = read_csv(os.path.join(args.refDir,args.refJournalJoin))
    journalmask = read_csv(os.path.join(args.refDir,args.refJournalMask))
    
    journaljoinfields = ['JournalName','StartNodeID','EndNodeID']
    journalmaskjoin = journaljoin.merge(journalmask,how='left',on=journaljoinfields)
    
    fullrefjoin = journalmaskjoin.merge(refMain,how='left',on=['ProbeFileID','JournalName'])
    
#    if args.inIndex:
    index = read_csv(os.path.join(args.refDir,args.inIndex))
    refMain = refMain.merge(index)
    journalmaskjoin = journalmaskjoin.merge(index)
    fullrefjoin = fullrefjoin.merge(index)
    
    refMain['errmsg'] = ""
    refMain['errflag'] = 0
    
    #pandas apply that to each row of the full join.
    #TODO: separate the length check from the interval check in the future
    if args.stringency > 0:
        refMain = refMain.apply(video_length_check,ref_dir=args.refDir,journal_info=fullrefjoin,precision=args.precision,axis=1,reduce=False)
    #fullrefjoin = fullrefjoin.apply(video_length_check,axis=1)

    refFlags = refMain[["ProbeFileID",'errmsg','errflag']].copy()
#    refFlags[['errflag','errmsg']] = refMain.apply(lambda r: video_dim_check(os.path.join(args.refDir,r['HDF5MaskFileName']),r["ProbeWidth"],r["ProbeHeight"]))

    #use a bitwise or to update the errflag and errmsg of refMain
    refcols = refMain.columns.values.tolist()
    pjjcols = journaljoin.columns.values.tolist()
    jmcols = journalmask.columns.values.tolist()
    idxcols = index.columns.values.tolist()
    skip_framecheck = "FrameCount" not in idxcols
    has_taskdesignation=False
    if args.stringency == 2:
        has_taskdesignation = "VideoTaskDesignation" in idxcols
    elif args.stringency == 3:
        has_taskdesignation = "VideoTaskDesignation" in refcols

    for i,row in refMain.iterrows():
        eflag = 0
        emsg = []
        if row["ProbeFileName"] == "":
            continue
        filename = os.path.join(args.refDir,row["ProbeFileName"])
        print("Checking dimensions for {}...".format(filename))
        eflag_dim,emsg_dim = video_dim_check(filename,row["ProbeWidth"],row["ProbeHeight"])
        eflag = eflag | eflag_dim
        emsg.append(emsg_dim)
        if not skip_framecheck:
            print("Checking framecount for {}...".format(filename))
            eflag_fc,emsg_fc = video_framecount_check(filename,row["FrameCount"])
            eflag = eflag | eflag_fc
            emsg.append(emsg_fc)

        emsg = "\n".join([ m for m in emsg if m != ""])
        
        refMain.at[i,"errflag"] = refMain.at[i,"errflag"] | eflag
        refMain.at[i,"errmsg"] = "\n".join([refMain.at[i,"errmsg"],emsg])

    #HDF5 mask tests start here.
    if args.stringency > 1:
        if has_taskdesignation:
            h5refcols = ["ProbeFileID","HDF5MaskFileName","JournalName","ProbeWidth","ProbeHeight","FrameCount","VideoTaskDesignation"]
        else:
            h5refcols = ["ProbeFileID","HDF5MaskFileName","JournalName","ProbeWidth","ProbeHeight","FrameCount"]
        h5ref = refMain[h5refcols].drop_duplicates()
        h5ref["errflag"] = 0
        h5ref["errmsg"] = ""
        for i,row in h5ref.iterrows():
            eflag = 0
            emsg = []
            if row["HDF5MaskFileName"] == "":
                continue

            filename = os.path.join(args.refDir,row["HDF5MaskFileName"])
            print("Checking dimensions for {}...".format(filename))
            eflag_dim,emsg_dim = video_dim_check(filename,row["ProbeWidth"],row["ProbeHeight"])
            eflag = eflag | eflag_dim
            emsg.append(emsg_dim)

            if not skip_framecheck:
                print("Checking framecount for {}...".format(filename))
                eflag_fc,emsg_fc = video_framecount_check(filename,row["FrameCount"])
                eflag = eflag | eflag_fc
                emsg.append(emsg_fc)

            journal_data = journalmaskjoin.query("(ProbeFileID == '{}') and (BitPlane != '')".format(row["ProbeFileID"]))
            task_designation = ""
            if has_taskdesignation:
                task_designation = row["VideoTaskDesignation"]

            if (task_designation in ["temporal","spatial-temporal"]) or (args.stringency >= 1):
                print("Checking temporal information for {}...".format(filename))
                eflag_j,emsg_j = video_mask_interval_check(filename,journal_data)
                eflag = eflag | eflag_j
                emsg.append(emsg_j)
            
            if task_designation in ["spatial","spatial-temporal"] or (args.stringency >= 4):
                print("Checking spatial information for {}...".format(filename))
                eflag_bp,emsg_bp = video_mask_bitplane_check(filename,journal_data)
                eflag = eflag | eflag_bp
                emsg.append(emsg_bp)

            emsg = "\n".join([ m for m in emsg if m != ""]).replace("\n\n","\n")
            h5ref.at[i,"errflag"] = h5ref.at[i,"errflag"] | eflag
            h5ref.at[i,"errmsg"] = "\n".join([h5ref.at[i,"errmsg"],emsg])
    #if the errmsg sum doesn't sum to 0, print all error messages and exit 1
    #TODO: modify this report
    all_errmsgs = "\n".join([m for m in refMain['errmsg'].tolist() if m != ""])
    print(all_errmsgs)
    if args.stringency > 1:
        print("\n".join([m for m in h5ref['errmsg'].tolist() if m != ""]))
    print("Dataset validation has finished for the video task.")
#    print refMain['errflag'].sum() #TODO: debug
    exitsum = refMain['errflag'].sum()
    if args.stringency > 1:
        exitsum += h5ref['errflag'].sum()
    exit(int(exitsum > 0))
    
