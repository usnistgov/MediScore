import numpy as np
import h5py
import cv2
from skvideo.io import FFmpegWriter
import matplotlib.pyplot as plt

def open_video(fname,path_to_mask='masks/masks',mode='r'):
    myfile = h5py.File(fname,mode)
    dset = myfile[path_to_mask]
    return myfile,dset

def open_video_dsets(fname,path_to_mask='masks',mode='r'):
    myfile = h5py.File(fname,mode)
    dsets = myfile[path_to_mask]
    return myfile,dsets

def play_mat_as_video(vmat,frate=1./30):
    vmat_slice = vmat[0,:,:]
    vmat_slice = vmat_slice.astype(np.uint8)
    im = plt.imshow(vmat_slice,cmap='gray')#,interpolation='nearest',vmax=threshold,cmap = cm)
    plt.title('Frame 1')
    framecount = vmat.shape[0]
    
    for f in range(1,framecount):
        vmat_slice = vmat[f,:,:].astype(np.uint8)
        im.set_data(vmat_slice)
    #    plt.set_title('Frame: {}'.format(f)) 
        plt.title('Frame {}'.format(f+1))
        plt.pause(frate)
    plt.show()

def save_mat_as_video_file_1(vfname,vmat):
    framecount = vmat.shape[0]
    
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    fps = 30
    out = cv2.VideoWriter(vfname,fourcc,fps,(vmat.shape[2],vmat.shape[1]))
    for f in range(framecount):
        frame = vmat[f,:,:]
        if len(frame.shape) == 2:
            frame = np.stack((frame,)*3,-1)
        out.write(frame)
        
    out.release()

def save_mat_as_video_file(vfname,vmat):
    framecount = vmat.shape[0]
    
    fps = 30
#    fourcc = cv2.cv.CV_FOURCC(*'XVID')
#    out = cv2.VideoWriter(vfname,fourcc,fps,(vmat.shape[2],vmat.shape[1]))
    out = FFmpegWriter(vfname)
#    out.open()
    for f in range(framecount):
        frame = vmat[f,:,:]
        if len(frame.shape) == 2:
            frame = np.stack((frame,)*3,-1)
        out.writeFrame(frame)
        
    out.close()
