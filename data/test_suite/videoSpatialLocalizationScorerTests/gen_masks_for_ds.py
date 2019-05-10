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

#TODO: add end_frame
def add_attributes(h5name,probe_file_id,end_frame=None):
    h_ptr = h5py.File(h5name,'r+')
    max_block = max([int(n) for n in h_ptr.keys()])
    h_ptr.attrs['start_frame'] = 1
    h_ptr.attrs["end_frame"] = max_block + h_ptr["{}/masks".format(max_block)].shape[0] - 1 if end_frame is None else end_frame
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

def add_block(h5name,interval,frame_shape,n_layer=1):
    dataset_name = "{}/masks".format(interval[0])
    h_ptr = h5py.File(h5name,'r+')
    group = str(interval[0])
    h_ptr.create_group(group)
    if n_layer == 1:
        h_ptr[group].create_dataset('masks',data=np.zeros((interval[1] - interval[0] + 1,frame_shape[0],frame_shape[1]),dtype=np.uint8))
    elif n_layer > 1:
        h_ptr[group].create_dataset('masks',data=np.zeros((interval[1] - interval[0] + 1,frame_shape[0],frame_shape[1],n_layer),dtype=np.uint8))
    else:
        print("Error: Number of layers {} mus be a positive integer.".format(n_layer))
        exit(1)
    h_ptr.close()
#    stamp_vid(h5name,0,interval[1] - interval[0],path_to_mask=dataset_name,value=1)

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
    
#merges sets of frames into one
def merge_intervals(interval_list):
    merged_interval_list = []
    a_ivls = np.sort(np.array(interval_list),axis=0)
    l = None
    r = None
    for i in a_ivls:
        if l is None:
            l = i[0]
            r = i[1]
            continue
        if r >= i[0] - 1:
            #case merge
            if i[1] > r:
                r = i[1]
        else:
            #case not merge
            merged_interval_list.append([l,r])
            l = i[0]
            r = i[1]
    merged_interval_list.append([l,r])
    
    return merged_interval_list

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate spatial mask data for the video spatial localization scorer.")
    parser.add_argument("-ds","--dataset",type=str,help="The name of the dataset to generate masks for. Must have the index and all reference files.")
    parser.add_argument("--shift_mask",type=int,default=0,help="The amount to shift the generated mask's dimensions by. Use this to generate bad datasets with slightly skewed masks. Default: 0.")
    parser.add_argument("--shift_frames",type=int,default=0,help="The amount to shift the generated mask's frames by. Use this to generate bad datasets with masks that are slightly temporally shifted. Default: 0.")
    parser.add_argument("--add_layer",action='store_true',help="Add a layer to the generated mask.")
    args = parser.parse_args()

    dataset_dir = args.dataset
    print(dataset_dir) #TODO: check
    idx = read_csv(os.path.join(dataset_dir,"indexes/MFC18_Dev2-manipulation-video-index.csv"))
    ref_name = os.path.join(dataset_dir,"reference/manipulation-video/MFC18_Dev2-manipulation-video-ref.csv")
    ref = read_csv(ref_name)

    pjj = read_csv(os.path.join(dataset_dir,"reference/manipulation-video/MFC18_Dev2-manipulation-video-ref-probejournaljoin.csv"))
    jm = read_csv(os.path.join(dataset_dir,"reference/manipulation-video/MFC18_Dev2-manipulation-video-ref-journalmask.csv"))
    jj = pjj.merge(jm)

    refidx = ref.merge(idx)
    refidx = refidx.query("(IsTarget == 'Y') and (VideoTaskDesignation in ['spatial','spatial-temporal'])")

    mask_out_dir = os.path.join(dataset_dir,'reference/manipulation-video/mask')

    for i,row in refidx.iterrows():
        framecount = row["FrameCount"]
        frame_shape = (row["ProbeHeight"] + args.shift_mask,row["ProbeWidth"] - args.shift_mask)
        journal_data = jj.query("(ProbeFileID == '{}') and (BitPlane != '')".format(row["ProbeFileID"]))
        interval_list = journal_data["VideoFrame"].tolist()
        print(", ".join([row["ProbeFileName"],str(frame_shape)]))

        tmp_name = os.path.join(mask_out_dir,'tmp.hdf5')
        #stamp the vid for every VideoFrame interval after mask is generated
        ivls_in_mask = 0
        #generate blocks depending on the merged frames, and then stamp vid
        merged_vframes = merge_intervals([i2 for i in interval_list for i2 in ast.literal_eval(i)])
        #gen mask
        if not os.path.isfile(tmp_name):
            sf = merged_vframes[0][0]
            ef = merged_vframes[0][1]
            
            n_layers = (max([int(bp) for bp in journal_data["BitPlane"]])//8) + 1
            gen_mask(tmp_name,frame_shape,ef - sf + 1,path_to_mask="{}/masks".format(sf),value=0,n_layer=n_layers)
            add_attributes(tmp_name,row["ProbeFileID"],framecount)
            if len(merged_vframes) > 1:
                for i in range(1,len(merged_vframes)):
                    add_block(tmp_name,i,frame_shape,n_layer=n_layers)

        #stamp blocks based on the interval_list
        for i,jd in journal_data.iterrows():
            ll = ast.literal_eval(jd["VideoFrame"])
            if len(ll) == 0:
                continue
            bpval = 1 << ((int(jd["BitPlane"]) - 1) % 8)
            for l in ll:
                start_frame = l[0]
                end_frame = l[1]
                block = max([i[0] for i in merged_vframes if i[0] <= start_frame])
                stamp_vid(tmp_name,start_frame - block,end_frame - block, path_to_mask = "{}/masks".format(block),value=bpval,layer=(int(jd["BitPlane"]) - 1)//8, opaque = False)

                ivls_in_mask += 1
            if ivls_in_mask == 0:
                continue

        #overrated, generate own mask and record the "manipulation" in jm and pjj on your own
#        new_mask_name = os.path.join(mask_out_dir,"{}.hdf5".format(get_md5(tmp_name)))
        new_mask_name = os.path.join(mask_out_dir,"{}-refmask.hdf5".format(row['ProbeFileID']))
        print("Mask name: {}".format(new_mask_name))
        os.system("mv {} {}".format(tmp_name,new_mask_name))
        #update the ref
        ref.loc[ref["ProbeFileID"] == row["ProbeFileID"],"HDF5MaskFileName"] = "reference/manipulation-video/mask/{}".format(os.path.basename(new_mask_name)) 

    write_csv(ref_name,ref)
