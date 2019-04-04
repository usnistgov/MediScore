import numpy as np
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
#    total_time = float(h_ptr.attrs['end_frame'] - h_ptr.attrs['start_frame'] + 1)/fix_frame_rate

    h_ptr.create_group("probe_ids")
    h_ptr.close()

def add_block(h5name,interval,frame_shape):
    dataset_name = "{}/masks".format(interval[0])
    h_ptr = h5py.File(h5name,'r+')
    group = str(interval[0])
    h_ptr.create_group(group)
    h_ptr[group].create_dataset('masks',data=255*np.ones((interval[1] - interval[0] + 1,frame_shape[0],frame_shape[1]),dtype=np.uint8))
    h_ptr.close()
    stamp_vid(h5name,0,interval[1] - interval[0],path_to_mask=dataset_name,value=0)

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
    
def shift_back(interval_list):
    new_interval_list = []
    for ivl in interval_list:
        new_interval_list.append([ivl[0] - 1,ivl[1] - 1])
    return new_interval_list


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate spatial mask data for the video spatial localization scorer.")
    parser.add_argument("-s","--inSys",type=str,help="The name of the system file to generate masks for.")
    parser.add_argument("-x","--inIndex",type=str,help="The name of the index file to generate masks for.")
    parser.add_argument("--shift_frames",type=int,default=0,help="The amount to shift the generated mask's frames by. Use this to generate bad data with masks that are slightly shifted. Default: 0.")
    parser.add_argument("--n_layer",type=int,default=1,help="The number of layers per frame in the mask. Default: 1.")
    args = parser.parse_args()

    sys_name = args.inSys
    sys = read_csv(sys_name)
    sys_dir = os.path.dirname(sys_name)

    idx_name = args.inIndex
    idx = read_csv(idx_name)

    sys_idx = sys.merge(idx).query("ProbeStatus in ['Processed','OptOutTemporal','OptOutDetection']")

    mask_out_dir = os.path.join(sys_dir,'mask')
    for i,row in sys_idx.iterrows():
        framecount = row["FrameCount"]
        frame_shape = (row["ProbeWidth"] - args.shift_frames,row["ProbeHeight"] + args.shift_frames)
        probe_file_id = row["ProbeFileID"]
        interval_list = row["VideoFrameSegments"]
        vmask_file_name = row["OutputProbeMaskFileName"]
        if vmask_file_name == '':
            continue

        tmp_name = os.path.join(mask_out_dir,'tmp.hdf5')
        #stamp the vid for every VideoFrame interval after mask is generated
        ivls_in_mask = 0
        ll_0 = ast.literal_eval(interval_list)
#        ll = shift_back(ll_0)
        ll = ll_0
        print ll
        stamp_value = 0
        for l in ll:
            start_frame = l[0]
            end_frame = l[1]
            if os.path.isfile(tmp_name):
                print("{} exists.".format(tmp_name))
                add_block(tmp_name,l,frame_shape) #TODO: merge blocks instead later
            else:
                print("{} does not exist. Generating now.".format(tmp_name))
                #generate a new mask for this probe in masks directory in this_dir
                gen_mask(tmp_name,frame_shape,end_frame - start_frame + 1,path_to_mask="{}/masks".format(start_frame),value=255,n_layer=args.n_layer)
                #stamp the vid with a mobile disc
                stamp_vid(tmp_name,0,end_frame - start_frame,path_to_mask="{}/masks".format(start_frame),value=stamp_value)
                stamp_value = (stamp_value + 100) % 256
                add_attributes(tmp_name,probe_file_id)
            ivls_in_mask += 1

        if ivls_in_mask == 0:
            continue

        #overrated, generate own mask and record the "manipulation" in jm and pjj on your own
        new_mask_name = os.path.join(sys_dir,vmask_file_name)
        print "Mask name: ",new_mask_name
        os.system("mv {} {}".format(tmp_name,new_mask_name))

