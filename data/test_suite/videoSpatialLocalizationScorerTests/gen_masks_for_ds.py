import pandas as pd
import h5py
import os
import sys
import ast
from hashlib import md5
this_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(this_dir,"../../../lib")
sys.path.append(lib_dir)
from gen_hdf5 import gen_mask,stamp_vid
from video_masks import video_ref_mask

# taken from: https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file
def get_md5(fname):
    hash_md5 = md5()
    with open(fname,"rb") as f:
        for chunk in iter(lambda: f.read(4096),b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def read_csv(fname,sep="|"):
    return pd.read_csv(fname,sep=sep,index_col=False,header=0,na_filter=False)

def write_csv(fname,df,sep="|"):
    df.to_csv(fname,index=False,sep=sep)

def add_attributes(h5name,probe_file_id):
    h_ptr = h5py.File(h5name,'r+')
    max_block = max([int(n) for n in h_ptr.keys()])
    h_ptr.attrs['start_frame'] = 1
    h_ptr.attrs["end_frame"] = max_block + h_ptr["{}/masks".format(max_block)].shape[0] - 1
    fix_frame_rate = 30
#    total_time = float(h_ptr.attrs['end_frame'] - h_ptr.attrs['start_frame'] + 1)/fix_frame_rate
    h_ptr.attrs['time_stamps'] = [round(x*(1./fix_frame_rate),3) for x in range(h_ptr.attrs['end_frame'] + 1)]
    h_ptr.attrs['error'] = 0
    h_ptr.attrs['BitPlanes'] = [1]

    h_ptr.create_group("probe_ids")
    probe_bp_dict = {probe_file_id:1}
    for pid in probe_bp_dict:
        h_ptr["probe_ids"].create_dataset(pid,data=[probe_bp_dict[pid]])
    h_ptr.close()

def add_block(h5name,interval,frame_shape):
    dataset_name = "{}/masks".format(interval[0])
    h_ptr = h5py.File(h5name,'r+')
    group = str(interval[0])
    h_ptr.create_group(group)
    h_ptr[group].create_dataset('masks',data=np.zeros((interval[1] - interval[0] + 1,frame_shape[0],frame_shape[1]),dtype=np.uint8))
    h_ptr.close()
    stamp_vid(h5name,0,interval[1] - interval[0],path_to_mask=dataset_name,value=1)

#TODO: merge existing block and relevant meta parameters
def merge_blocks(h5name,interval):
    dataset_name = "{}/masks".format(interval[0])
    ref = video_ref_mask(h5name)
    ref_intervals = ref.compute_ref_intervals()
    ref.close()
    h_ptr = h5py.File(h5name,'w')
    #TODO: determine whether interval intersects with any intervals in the mask
    #TODO: if not, add new block
    #TODO: if so, generate new block with all information from all intervals, remove old (intersecting) intervals, and add a new block
    #TODO: 
    h_ptr.close()
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate spatial mask data for the video spatial localization scorer.")
    parser.add_argument("-ds","--dataset",type=str,help="The name of the dataset to generate masks for. Must have the index and all reference files.")
    parser.add_argument("--shift_mask",type=int,default=0,help="The amount to shift the generated mask's dimensions by. Use this to generate bad datasets with slightly skewed masks. Default: 0.")
    parser.add_argument("--shift_frames",type=int,default=0,help="The amount to shift the generated mask's frames by. Use this to generate bad datasets with masks that are slightly temporally shifted. Default: 0.")
    parser.add_argument("--add_layer",action='store_true',help="Add a layer to the generated mask.")
    args = parser.parse_args()

    dataset_dir = args.dataset
    print dataset_dir #TODO: check
    idx = read_csv(os.path.join(dataset_dir,"indexes/MFC18_Dev2-manipulation-video-index.csv"))
    ref_name = os.path.join(dataset_dir,"reference/manipulation-video/MFC18_Dev2-manipulation-video-ref.csv")
    ref = read_csv(ref_name)

    pjj = read_csv(os.path.join(dataset_dir,"reference/manipulation-video/MFC18_Dev2-manipulation-video-ref-probejournaljoin.csv"))
    jm = read_csv(os.path.join(dataset_dir,"reference/manipulation-video/MFC18_Dev2-manipulation-video-ref-journalmask.csv"))
    jj = pjj.merge(jm)

    ref_targs = ref.query("(IsTarget == 'Y') and (VideoTaskDesignation in ['spatial','spatial-temporal'])")
    refidx = ref_targs.merge(idx)

    mask_out_dir = os.path.join(dataset_dir,'reference/manipulation-video/mask')
    for i,row in refidx.iterrows():
        framecount = row["FrameCount"]
        frame_shape = (row["ProbeWidth"] - args.shift_mask,row["ProbeHeight"] + args.shift_mask)
        interval_list = jj.query("(ProbeFileID == '{}') and (BitPlane != '')".format(row["ProbeFileID"]))["VideoFrame"].tolist()
        print row["ProbeFileName"],frame_shape

        tmp_name = os.path.join(mask_out_dir,'tmp.hdf5')
        #stamp the vid for every VideoFrame interval after mask is generated
        ivls_in_mask = 0
        for ivl in interval_list:
            ll = ast.literal_eval(ivl)
            if len(ll) == 0:
                continue
            print ll
            for l in ll:
                start_frame = l[0] + args.shift_frames
                end_frame = l[1] + args.shift_frames
                if os.path.isfile(tmp_name):
                    print("{} exists.".format(tmp_name))
                    add_block(tmp_name,l,frame_shape) #TODO: merge blocks instead later
                else:
                    print("{} does not exist. Generating now.".format(tmp_name))
                    #generate a new mask for this probe in masks directory in this_dir
                    gen_mask(tmp_name,frame_shape,end_frame - start_frame + 1,path_to_mask="{}/masks".format(start_frame),value=0)
                    #stamp the vid with a mobile disc
                    stamp_vid(tmp_name,0,end_frame - start_frame,path_to_mask="{}/masks".format(start_frame),value=1)
                    if args.add_layer:
                        stamp_vid(tmp_name,(end_frame - start_frame)//2,end_frame - start_frame,path_to_mask="{}/masks".format(start_frame),value=2)
                    add_attributes(tmp_name,row["ProbeFileID"])
                ivls_in_mask += 1

        if ivls_in_mask == 0:
            continue

        #overrated, generate own mask and record the "manipulation" in jm and pjj on your own
        new_mask_name = os.path.join(mask_out_dir,"{}.hdf5".format(get_md5(tmp_name)))
        print "Mask name: ",new_mask_name
        os.system("mv {} {}".format(tmp_name,new_mask_name))
        #update the ref
        ref.loc[ref["ProbeFileID"] == row["ProbeFileID"],"ProbeBitPlaneMaskFileName"] = "reference/manipulation-video/mask/{}".format(os.path.basename(new_mask_name)) 

    write_csv(ref_name,ref)
