"""
* File name: gen_hdf5.py
* Description: A set of functions used to generate video spatial localization HDF5 masks.
* Author: Daniel Zhou
* Date: 2018-10-29
"""

import h5py
import numpy as np
import os
import sys
import cv2
from math import ceil
vidprobe_sample_dir=".."
ref_mask_fname = os.path.join(vidprobe_sample_dir,"0c5fc5f508229a3f47e589b18acefc25/duplicated_orange_dude_Final_added_colored_mask_0.0.hdf5")
#"0c5fc5f508229a3f47e589b18acefc25/stabilized_duplicated_orange_dude_mask_0.0.hdf5"
#hdf5_lib_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"lib")
#sys.path.append(hdf5_lib_dir)
from masks import getKern
from hdf5_lib import open_video
from video_masks import video_mask

colordict_rev={'red':[0,0,255],'blue':[255,51,51],'yellow':[0,255,255],'green':[0,207,0],'pink':[193,182,255],'purple':[211,0,148],'white':[255,255,255],'gray':[127,127,127]}
colordict={'red':[255,0,0],'blue':[51,51,255],'yellow':[255,255,0],'green':[0,207,0],'pink':[255,182,193],'purple':[148,0,211],'white':[255,255,255],'gray':[127,127,127]}

colordict_grayscale = {'black':0,
                       'white':255}

def gen_mask_from_ref(fname,path_to_mask="masks/masks",value=255,frame_scale=1,ref_mask_name = ref_mask_fname,n_layer=1):
    file_pointer,ref_mask = open_video(ref_mask_fname)
    mask_shape = ref_mask.shape
    file_pointer.close()
    new_mask_shape = mask_shape
    frame_count = mask_shape[0]

    if frame_scale != 1:
#        new_mask_shape = (int(conditions['frame_scale']*mask_shape[0]),mask_shape[1],mask_shape[2])
        frame_count = int(frame_scale*mask_shape[0])

    gen_mask(fname,[mask_shape[1],mask_shape[2]],frame_count,path_to_mask=path_to_mask,value=value,n_layer=n_layer)

def rm_mask(fname):
    os.system('rm {}'.format(fname))

def gen_mask(fname,frame_shape,frame_count,path_to_mask="masks/masks",value=255,channel='gray',n_layer=1):
    f = h5py.File(fname,'w')
    group_name,dataset_name = path_to_mask.split("/")
    f.create_group(group_name)
    if n_layer == 1:
        if channel == 'gray':
            new_mask_shape = [frame_count,frame_shape[0],frame_shape[1]]
        elif channel == 'RGB':
            print("Warning: The 'channel' operation is deprecated and is soon to be replaced with the n_layer option.")
            new_mask_shape = [frame_count,frame_shape[0],frame_shape[1],3]
        else:
            print("{} is not recognized as a valid channel ('gray','RGB').".format(channel))
    else:
        new_mask_shape = [frame_count,frame_shape[0],frame_shape[1],n_layer]

    data = value*np.ones(new_mask_shape,dtype=np.uint8)

    f[group_name].create_dataset(dataset_name,data=data)

    f.close()

def gen_sys_frame_no_score(sys_frame,pppns):
    return (sys_frame != pppns).astype(np.uint8)

def color_code_frame(ref_frame,sys_frame,bns_frame = 1,sns_frame=1,pns_frame=1):
    score_frame = (ref_frame == 0) + 2*(sys_frame == 0) + 4*(bns_frame == 0) + 8*(sns_frame == 0) + 16*(pns_frame == 0)
    color_frame = 255*np.ones((score_frame.shape[0],score_frame.shape[1],3),dtype = np.uint8)
    color_frame[score_frame == 1] = colordict['blue'] #false negative
    color_frame[score_frame == 2] = colordict['red'] #false positive
    color_frame[score_frame == 3] = colordict['green'] #true positive
    color_frame[(score_frame >= 4) & (score_frame <= 7)] = colordict['yellow']
    color_frame[(score_frame >= 8) & (score_frame <= 15)] = colordict['pink']
    color_frame[score_frame >= 16] = colordict['purple']

    return color_frame
    
def gen_color_results(color_mask_name,ref,sys,th=None,eks=15,dks=11,ntdks=15,kern='box',pppns=-1):
    #takes in video mask objects
    framecount_ref = ref.framecount
    framecount_sys = sys.framecount
    color_mask = gen_mask(color_mask_name,ref.shape,ref.framecount,channel='RGB')
    col_pointer,col_data = open_video(color_mask_name,mode='r+')
    for f in range(framecount_ref):
        ref_frame = ref.get_frame(f)
        if ref.has_journal_data():
            ref_frame.insert_journal_data(ref.journal_data)

        if f >= framecount_sys:
            sys_frame = sys.whiteframe
        else:
            sys_frame = sys.get_frame(f)

        #boundary and pixel no-scores
        bns_frame = ref_frame.get_boundary_no_score(eks=eks,dks=dks,kern=kern)
        sns_frame = ref_frame.get_selective_no_score(eks=eks,ntdks=ntdks,kern=kern)
        pns_frame = gen_sys_frame_no_score(sys_frame,pppns)

        if th is not None:
            _,sys_frame = cv2.threshold(sys_frame,th,255,cv2.THRESH_BINARY)

        if f >= framecount_sys:
            col_data[f,:,:] = color_code_frame(ref_frame.get_binary(),1,bns_frame=bns_frame,sns_frame = sns_frame,pns_frame=pns_frame)
        else:
            col_data[f,:,:] = color_code_frame(ref_frame.get_binary(),sys_frame,bns_frame=bns_frame,sns_frame=sns_frame,pns_frame=pns_frame)
    col_pointer.close()

# stamp select frames of a video with a certain shape of some size
def stamp_vid(fname,start_frame,end_frame,path_to_mask="masks/masks",value=0,opaque=True,layer=0,stamp_relative_size=0.3,relative_padding=0.01,stamp_shape='disc',path_shape='box',start_position='lower_left'):
    f = h5py.File(fname,'r+')
    mask_shape = f[path_to_mask].shape
    dims = mask_shape[1:3]
    frame_count = mask_shape[0]
    is_multi_layer = len(mask_shape) > 3

    assert start_frame <= end_frame, "Start frame {} needs to be upper-bounded by end frame {}.".format(start_frame,end_frame)
    assert start_frame >= 0, "Start frame {} needs to be nonnegative.".format(start_frame)
    assert end_frame < frame_count, "End frame {} needs to be bounded by the video frame count {}.".format(end_frame,frame_count)
    assert min(stamp_relative_size,relative_padding) >= 0, \
        "Either the stamp relative size {} or the relative padding {} of the video needs to be nonnegative.".format(stamp_relative_size,relative_padding)
    assert stamp_relative_size + 2*relative_padding < 1, \
        "The stamp relative size {} should fit in the whole video with twice the relative padding {} (2*relative_padding).\
 Together they should be less than 1.".format(stamp_relative_size,2*relative_padding)
    if is_multi_layer:
        assert layer < mask_shape[3], "Layer {} as not accessible. There are only {} layers.".format(layer,mask_shape[3])

    padding = int(relative_padding*min(dims))
    stamp_size = int(stamp_relative_size*min(dims))
    stamp_size = stamp_size if stamp_size % 2 == 1 else stamp_size + 1
    if path_shape not in ['box']:
        print("Error: Path shape {} is not available yet.".format(path_shape))
        exit(1)

    if start_position == 'lower_left':
        start_coordinate = [dims[0]-padding-stamp_size,padding]
    elif start_position == 'lower_right':
        start_coordinate = [dims[0]-padding-stamp_size,dims[1]-padding-stamp_size]
    elif start_position == 'upper_left':
        start_coordinate = [padding,padding]
    elif start_position == 'upper_right':
        start_coordinate = [padding,dims[1]-padding-stamp_size]

    total_pixels_traversed = 2*(dims[0]+dims[1]) - 8*padding
    pixels_per_frame = int(total_pixels_traversed/(end_frame - start_frame + 1))
    if pixels_per_frame == 0:
        pixels_per_frame = 1
    
    kernmat = getKern(stamp_shape,stamp_size)
    coordinate = start_coordinate[:]
    side_traversal = start_position
    for fnum in range(start_frame,end_frame+1):
        #side_traversal dictates clockwise movement about the image
        if side_traversal == 'lower_left':
            #move up, right if the y_coordinate would otherwise be less than padding
            if coordinate[0] - pixels_per_frame < padding:
                coordinate = [padding,padding]
                side_traversal = 'upper_left'
            else:
                coordinate = [coordinate[0] - pixels_per_frame,padding]
        elif side_traversal == 'lower_right':
            #move left, up if the x_coordinate would otherwise be less than padding
            if coordinate[1] - pixels_per_frame < padding:
                coordinate = [dims[0] - padding - pixels_per_frame - stamp_size,padding]
                side_traversal = 'lower_left'
            else:
                coordinate = [coordinate[0],coordinate[1] - pixels_per_frame]

        elif side_traversal == 'upper_left':
            #move right, down if the x_coordinate plus kernmat shape would otherwise be greater than padding
            if coordinate[1] + pixels_per_frame + stamp_size > dims[1] - padding:
                coordinate = [padding,dims[1] - padding - pixels_per_frame - stamp_size]
                side_traversal = 'upper_right'
            else:
                coordinate = [coordinate[0], coordinate[1] + pixels_per_frame]

        elif side_traversal == 'upper_right':
            #move down, left if the y_coordinate plus kernmat shape would otherwise be greater than padding
            if coordinate[0] + pixels_per_frame + stamp_size > dims[0] - padding:
                coordinate = [dims[0] - padding - pixels_per_frame - stamp_size,dims[1] - padding - pixels_per_frame - stamp_size]
                side_traversal = 'lower_right'
            else:
                coordinate = [coordinate[0] + pixels_per_frame ,coordinate[1]]
        if is_multi_layer:
            frame = f[path_to_mask][fnum,:,:,layer]
        else:
            frame = f[path_to_mask][fnum,:,:]
        stamp = frame[coordinate[0]:coordinate[0] + stamp_size,coordinate[1]:coordinate[1] + stamp_size]
        if opaque:
            stamp[kernmat == 1] = value
        else:
            stamp += kernmat*value
        frame[coordinate[0]:coordinate[0] + stamp_size,coordinate[1]:coordinate[1] + stamp_size] = stamp
        if is_multi_layer:
            f[path_to_mask][fnum,:,:,layer] = frame
        else:
            f[path_to_mask][fnum,:,:] = frame

    f.close()

def staircase_vid(fname,start_frame,end_frame,path_to_mask="masks/0",value=0,layer=0,axis=0):
    f = h5py.File(fname,'r+')
    mask_shape = f[path_to_mask].shape
    dims = mask_shape[1:3]
    frame_count = mask_shape[0]
    assert start_frame <= end_frame, "Start frame {} needs to be upper-bounded by end frame {}.".format(start_frame,end_frame)
    assert start_frame >= 0, "Start frame {} needs to be nonnegative.".format(start_frame)
    assert end_frame < frame_count, "End frame {} needs to be bounded by the video frame count {}.".format(end_frame,frame_count)
    
    #four shades of gray, progress down to black
    start_colors = [255,191,127,63]
    time_interval_descent = 1
    chroma_interval = 1

    if min(start_colors) >= end_frame - start_frame + 1:
        time_interval_descent = int(ceil(min(start_colors)/(end_frame - start_frame + 1)))
    else:
        chroma_interval = int(frame_count/min(start_colors))

    last_frame = 255*np.ones(dims,dtype=np.uint8)
    if axis == 0:
        band_width = int(dims[0]/4)
        last_frame[0:band_width,:] = start_colors[0]
        last_frame[band_width:2*band_width,:] = start_colors[1]
        last_frame[2*band_width:3*band_width,:] = start_colors[2]
        last_frame[3*band_width:,:] = start_colors[3]
    elif axis == 1:
        band_width = int(dims[1]/4)
        last_frame[:,0:band_width] = start_colors[0]
        last_frame[:,band_width:2*band_width] = start_colors[1]
        last_frame[:,2*band_width:3*band_width] = start_colors[2]
        last_frame[:,3*band_width:] = start_colors[3]

    for fnum in range(start_frame,end_frame+1):
        if fnum % time_interval_descent == 0:
            last_frame = last_frame - chroma_interval
        f[path_to_mask][fnum,:,:] = last_frame
     
    f.close()
