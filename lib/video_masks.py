import h5py
from hdf5_lib import open_video,open_video_dsets
from skvideo.io import FFmpegWriter
from math import log

import numpy as np
import cv2
from masks import erode,dilate,mask

try:
    frame_pos_const = cv2.cv.CV_CAP_PROP_POS_FRAMES
    width_const = cv2.cv.CV_CAP_PROP_FRAME_WIDTH
    height_const = cv2.cv.CV_CAP_PROP_FRAME_HEIGHT
    framecount_const = cv2.cv.CV_CAP_PROP_FRAME_COUNT
except AttributeError:
    frame_pos_const = cv2.CAP_PROP_POS_FRAMES
    width_const = cv2.CAP_PROP_FRAME_WIDTH
    height_const = cv2.CAP_PROP_FRAME_HEIGHT
    framecount_const = cv2.CAP_PROP_FRAME_COUNT

def gen_mask(fname,frame_shape,frame_count,path_to_mask="masks/masks",value=255,levels=1):
    f = h5py.File(fname,'w')
    group_name,dataset_name = path_to_mask.split("/")
    f.create_group(group_name)
    new_mask_shape = [frame_count,frame_shape[0],frame_shape[1]] if levels == 1 else [frame_count,frame_shape[0],frame_shape[1],levels]

    data = value*np.ones(new_mask_shape,dtype=np.uint8)
    f[group_name].create_dataset(dataset_name,data=data)
    f.close()

def union_frames(framelist):
    """
    Description: Takes in a list of separate integer frames and returns a list of contiguous intervals.
    """
    intervals = []
    ivl = []
    last_frame = -2
    for f in framelist:
        if (f == last_frame + 1) or (last_frame == -2):
            ivl.append(f)
            last_frame = f
        elif (f > last_frame + 1) and (last_frame > -2):
            if len(ivl) == 1:
                intervals.append([ivl[0],ivl[0]])
            else:
                intervals.append([ivl[0],ivl[-1]])
            ivl = [f]
            last_frame = f
    if len(ivl) == 1:
        intervals.append([ivl[0],ivl[0]])
    else:
        intervals.append([ivl[0],ivl[-1]])
    return intervals

class frame(np.ndarray):
    """
    Description: Wrapper for each video frame. Behaves similarly to image localization masks.
    * Inputs:
        - data: initializes with an existing np.ndarray
    """
    def __new__(cls,*args):
        return super(frame,cls).__new__(cls,*args)

    def __array_finalize__(self,obj):
        self.is_multi_layer = len(self.shape) > 2

    def has_journal_data(self):
        if 'journal_data' in vars(self):
            return True
        return False

    def insert_journal_data(self,jData,evalcol="Evaluated"):
        self.journal_data = jData.sort_values("Sequence",ascending=False)
        self.evalcol=evalcol

    def as_1_layer(self):
        if self.is_multi_layer:
            n_layers = self.shape[2]
            mat = np.zeros((self.shape[0],self.shape[1]))
            for l in range(n_layers):
                mat += self[:,:,l]*(1 << (8*l))
            return mat
        else:
            return self
           
    #TODO: change all get bitplanes to get bitvals. Too confusing to deal with bitplanes in the code unless subdividing bitplanes
    def get_bitplanes(self):
        if self.is_multi_layer:
            n_layers = self.shape[2]
            bitlist = []
            for l in range(n_layers):
                offset = 8*l
                bitlist.extend([ b + offset for b in self.get_bitplanes_layer(self[:,:,l])])
        else:
            bitlist = self.get_bitplanes_layer(self)
        if 0 in bitlist:
            bitlist.remove(0)
        return bitlist

    def get_bitplanes_layer(self,mat):
        #sum along a particular axis, np.unique that, and then disassemble
        pix_vals = list(np.unique(mat))
        pix_vals.remove(0)
        if len(pix_vals) == 0: return pix_vals

        max_bp = int(log(max(pix_vals),2)) + 1
        all_bps = [ 1 << (b - 1) for b in range(1,max_bp + 1) ]
        bp_list = [ int(log(b,2)) + 1 for p in pix_vals for b in all_bps if b & p > 0 ]
        bp_list = np.unique(bp_list).tolist()
        return bp_list

    def subdivide_bitplanes(self,bp_list,dividing_value = 8):
        """
        * Description: subdivide into a list of list of BitPlane's. Each list is also shifted back appropriately.
        """
        n_layer = (max(bp_list) - 1)//dividing_value + 1
        bp_l_list = []
        for l in range(n_layer):
            offset = dividing_value*l
            bp_l = [ b - offset for b in bp_list if ((b >= offset) and (b < offset + dividing_value)) ]
            bp_l_list.append(bp_l)
        return bp_l_list

    def get_binary(self):
        bmat = np.ones((self.shape[0],self.shape[1]),dtype=np.uint8)
        if self.is_multi_layer:
            n_layers = self.shape[2]
            for l in range(n_layers):
                bmat = bmat & (self[:,:,l] == 0)
            return 255*bmat.astype(np.uint8)
        else:
            return 255*(self == 0).astype(np.uint8)

    def bp_from_journal(self):
        if self.has_journal_data():
            bp = list(set(self.journal_data.query("Evaluated == 'Y'")['BitPlane'].tolist()))
            if '' in bp: 
                bp.remove('')
            bp = [int(b) for b in bp]
        else:
            bp = self.get_bitplanes()
        return bp

    def get_boundary_no_score(self,eks=0,dks=0,kern='box'):
        bp = self.bp_from_journal()
        if self.is_multi_layer:
            n_layers = self.shape[2]
            bp_list = self.subdivide_bitplanes(bp)
            bns = np.zeros((self.shape[0],self.shape[1]),dtype=np.uint8)
            for l in range(n_layers):
                bns = bns | self.get_boundary_no_score_1L(self[:,:,l],bp_list[l],eks=eks,dks=dks,kern=kern)
            return bns
        else:
            return self.get_boundary_no_score_1L(self,bp,eks=eks,dks=dks,kern=kern)

    def get_boundary_no_score_1L(self,mat,bitplanes,eks=0,dks=0,kern='box'):
        if self.has_journal_data():
            bp_final = sum([ 1 << (int(b) - 1) for b in bitplanes ])
            bit_mat = ((mat & bp_final) > 0).astype(np.uint8)
        else:
            bit_mat = (mat > 0).astype(np.uint8)
        emat = erode(bit_mat,kernel=kern,kernsize=eks)
        dmat = dilate(bit_mat,kernel=kern,kernsize=dks)
        return ((dmat - emat) == 0).astype(np.uint8)

    def get_selective_no_score(self,eks=0,ntdks=0,kern='box'):
        if not self.has_journal_data():
            return np.ones(self.shape,dtype=np.uint8)
        bp = self.bp_from_journal()
        if self.is_multi_layer:
            n_layers = self.shape[2]
            bp_list = self.subdivide_bitplanes(bp)
            sns = np.ones((self.shape[0],self.shape[1]),dtype=np.uint8)
            emat = np.zeros((self.shape[0],self.shape[1]),dtype=np.uint8)
            for l in range(n_layers):
                bv = video_ref_mask.bitplanes_to_bitvals(bp_list[l])
                sns = sns & self.get_selective_no_score_pure(self[:,:,l].view(frame)[:],bp_list[l],ntdks=ntdks,kern=kern)
                #get the eroded reference
                emat = emat | erode(((self[:,:,l] & sum(bv)) > 0).astype(np.uint8),kernel=kern,kernsize=eks)
        else:
            sns = self.get_selective_no_score_pure(self,bp,ntdks=ntdks,kern=kern)
            bv = video_ref_mask.bitplanes_to_bitvals(bp)
            emat = erode(((self & sum(bv)) > 0).astype(np.uint8),kernel=kern,kernsize=eks)
        #make the regional cut at the very end, after getting all the no-score zones
        sns = sns | emat
        return sns

    def get_selective_no_score_pure(self,mat,bp_to_eval,ntdks=0,kern='box'):
        frame_bp = mat.get_bitplanes()
        bp_not_to_eval = [ b for b in frame_bp if b not in bp_to_eval ]
        if len(bp_not_to_eval) == 0:
            return np.ones((mat.shape[0],mat.shape[1]),dtype=np.uint8)
        bitvals_not_to_eval = video_ref_mask.bitplanes_to_bitvals(bp_not_to_eval)
        bitsum_not = sum(bitvals_not_to_eval)

        #get no-scored zone according to ntdks
        sns = erode(((mat & bitsum_not) == 0).astype(np.uint8),kernel=kern,kernsize=ntdks)
        return sns

    #TODO: add function to treat SNS as white instead of SNS

    #TODO: add this to reference masks in image loc scorer. This is probe-dependent.
    def has_ground_truth_positive(self,no_score_zone,ftype='BitPlane'):
        """
        * Description: determines whether the frame has a ground-truth positive
        """
        rmat = (self.as_1_layer() > 0) if ftype == "BitPlane" else (self == 0)
        rmat = rmat.astype(np.uint8)
        return np.sum(rmat & no_score_zone) == 0

class video(object):
    #TODO: change to generalize to video. Most methods here should be in video_mask instead 
    def __init__(self,n,mode='r+'):
        self.name = n
        self.ext = n.split('.')[-1].lower()
        if self.ext == 'hdf5':
#            self.fpointer,self.data = open_video_dsets(n,mode=mode)
            self.fpointer = h5py.File(n,mode)
            self.start_frames = [int(k) for k in self.fpointer.keys() if k.isdigit()]
            #abstract to the segmented masks. 
#            self.datasets = self.data.keys()
#            if 'masks' in self.datasets:
            if 'masks' in self.start_frames:
                fullshape = self.fpointer['masks/masks'].shape
                self.shape = [fullshape[1],fullshape[2]]
                self.framecount = fullshape[0]
#                self.datasets = 'masks'
                max_start_frame = self.framecount - 1
            else:
#                start_frame_list = [int(k) for k in self.data.keys()]
                max_start_frame = max(self.start_frames)
                fullshape = self.fpointer["{}/masks".format(max_start_frame)].shape
                self.shape = [fullshape[1],fullshape[2]]
                if 'end_frame' in self.fpointer.attrs.items():
                    self.framecount = self.fpointer.attrs.items()['end_frame'] + 1
                else:
                    #infer the end frame if none given in the group attributes
                    self.framecount = int(max_start_frame) + fullshape[0]
            self.is_multi_layer = len(self.get_frame(max_start_frame).shape) > 2
            self.whiteframe = 255*np.ones(self.shape,dtype = np.uint8)
        else:
            try:
                self.fpointer = cv2.VideoCapture(n)
                self.shape = [self.fpointer.get(width_const),self.fpointer.get(height_const)]
                framecount = 0
                while True:
                    (grabbed,frame) = vfile.read()
                    if not grabbed:
                        (grabbed2,frame2) = vfile.read()
                        if not grabbed2:
                            break
                    framecount += 1
                self.framecount = framecount
            except:
                print("Error: File {} is not in a supported format. Expected an hdf5, or a video that can be read by OpenCV {}.".format(n,cv2.__version__))
                exit(1)

    def close(self):
        if self.ext == 'hdf5':
            self.fpointer.close()
        else:
            self.fpointer.release()

    def __iter__(self):
        self.frame_index = 0
        return self

    def __next__(self):
        if self.frame_index < self.framecount:
            result = self.get_frame(self.frame_index)
            self.frame_index += 1
            return result
        else:
            raise StopIteration

    def next(self):
        return self.__next__()

    def get_dims(self):
        return self.shape

    def get_framecount(self):
        return self.framecount

    def get_dataset_number(self,frame_number):
        ds_num_set = [ int(n) for n in self.start_frames if frame_number >= int(n) ]
        ds_num = None if len(ds_num_set) == 0 else str(max(ds_num_set))
        return ds_num

    def get_frame_vc(self,frame_number):
        self.fpointer.set(frame_pos_const,frame_number)
        ret,frame = self.fpointer.read()
        return frame

    def get_frame(self,frame_number):
        if self.ext != 'hdf5':
            return self.get_frame_vc(frame_number)

        if 'masks' in self.start_frames:
            ds = self.fpointer['masks/masks']
            if frame_number >= self.framecount:
                frame_mat = self.whiteframe.copy()
            else:
                frame_mat = self.fpointer['masks/masks'][frame_number,:,:]
        else:
            ds_num = self.get_dataset_number(frame_number)
            if ds_num is None:
                frame_mat = self.whiteframe.copy()
            else:
                ds = self.fpointer["{}/masks".format(ds_num)]
                frame_number_relative = int(frame_number) - int(ds_num)
                if frame_number_relative < ds.shape[0]:
                    frame_mat = ds[frame_number_relative]
                else:
                    frame_mat = self.whiteframe.copy()

        return frame_mat.view(frame)[:]

    def as_one_block(self,verbose=False):
        if verbose:
            print("Warning: Getting an array as_one_block will use up a lot of disk space. This method should not be used en masse (i.e. for evaluations).")

        if self.is_multi_layer:
            new_array = np.zeros((self.framecount,self.shape[0],self.shape[1],3),dtype = np.uint8)
        else:
            new_array = np.zeros((self.framecount,self.shape[0],self.shape[1]),dtype = np.uint8)

        for f in range(self.framecount):
            if self.is_multi_layer:
                new_array[f,:,:,:] = self.get_frame(f)
            else:
                new_array[f,:,:] = self.get_frame(f)
        return new_array

    def getColors(self):
        colorlist = set()
        for f in self:
            frame1L = f
            if self.is_multi_layer:
                frame1L = f[:,:,2] + 256*f[:,:,1] + 65536*f[:,:,0]
            colorlist = colorlist.union(set(np.unique(frame1L)))

        if self.is_multi_layer:
            colorlist = [ [c//65536,(c % 65536)//256,c % 256] for c in colorlist ]
        
        return colorlist

    def save_as_video(self,filename,fps=30):
        out = FFmpegWriter(filename)

        for f in self:
            frame = f
            if not self.is_multi_layer:
                frame = np.stack((f,)*3,-1)
            out.writeFrame(frame)

        out.close()

class video_mask(video):
    def __init__(self,n,SNS_frames=None):
        super(video_mask,self).__init__(n,mode='r')
        self.SNS_frames = SNS_frames

    def compute_intervals(self):
        frame_list = []
        for i,f in enumerate(self):
            for p in np.unique(f).tolist():
                if p > 0:
                    frame_list.append(i)
                    continue
        return union_frames(frame_list)

    def compute_sys_intervals(self,threshold):
        frame_list = []
        for i,f in enumerate(self):
            _,f_new = cv2.threshold(f,threshold,1,cv2.THRESH_BINARY_INV)
            for p in np.unique(f_new).tolist():
                if p > 0:
                    frame_list.append(i)
                    continue
        return union_frames(frame_list)

    def get_all_thresholds(self):
        th_set = set()
        for f in self:
            pixels = set(np.unique(f))
            th_set = th_set.union(pixels)

        th_set.union({-1,255})
        return list(th_set)

    def gen_threshold_mask(self,fname,th=-1):
        path_to_mask = "0/masks"
        gen_mask(fname,self.shape,self.framecount,path_to_mask=path_to_mask)
        t_pointer,t_data = open_video(fname,path_to_mask=path_to_mask,mode='r+')
        for i,f in enumerate(self):
            _,t_data[i,:,:] = cv2.threshold(f,th,255,cv2.THRESH_BINARY)
        t_pointer.close()

    def get_selective_no_score_frames(self):
        return self.SNS_frames

class video_ref_mask(video_mask):
    def __init__(self,n):
        super(video_ref_mask,self).__init__(n)
        self.whiteframe = np.zeros(self.shape,dtype = np.uint8)
    
    def insert_journal_data(self,jData,evalcol="Evaluated"):
        self.journal_data = jData.sort_values("Sequence",ascending=False)
        self.evalcol=evalcol
        
    def has_journal_data(self):
        if 'journal_data' in vars(self):
            return True
        return False

    def journal_data_to_csv(self,filename,sep="|"):
        if self.has_journal_data():
            self.journal_data.to_csv(filename,sep=sep,index=False)
        else:
            print("Warning: No journal data to output for mask {}. journal_data_to_csv() does nothing.".format(self.name))

    def compute_ref_intervals(self,eks=0,dks=0,ntdks=0,sys=None,nspx=-1,kern='box'):
        """
        * Description: Gets temporal intervals for reference. Accounts for no-score zones.
        """
        frame_list = []
        has_journal = self.has_journal_data()
        for i,f in enumerate(self):
            if has_journal:
                f.insert_journal_data(self.journal_data,self.evalcol)
            bns = f.get_boundary_no_score(eks,dks,kern=kern)
            sns = f.get_selective_no_score(eks,ntdks,kern=kern)
            if sys is not None:
                pns = (sys.get_frame(i) != nspx).astype(np.uint8)
            
            for p in np.unique((f > 0).astype(np.uint8) & bns & sns & pns).tolist():
                if p > 0:
                    frame_list.append(i)
                    continue
        return union_frames(frame_list)
    
    def get_bw_frame(self,frame_number):
        return self.get_frame(frame_number).get_binary()

    @staticmethod
    def bitplanes_to_bitvals(bplist):
        return [1 << (int(b) - 1) for b in bplist]

    def get_selective_no_score(self,frame_number,eks=0,ntdks=0,kern='box'):
        if not self.has_journal_data():
            return np.ones(self.shape,dtype=np.uint8)
        
        ref_frame = self.get_frame(frame_number)
        ref_frame.insert_journal_data(self.journal_data,self.evalcol)
        return ref_frame.get_selective_no_score(eks=eks,ntdks=ntdks,kern=kern)

    #TODO: determines whether or not the frame in question is a collared frame. I assume it's a temporal no-score zone? See Timothee.
    def is_collared_frame(self,frame_number,collar_value):
        #TODO: compute manipulated frame intervals, or otherwise retrieve them froum journal_data
        return False

#    #TODO: generate no score zanes for each frame and save that as an hdf5? This has the potential to be extremely data intensive.
#    def gen_boundary_no_score(self,erode_kern_size,dilate_kern_size,kern):
#        if (erode_kern_size == 0) and (dilate_kern_size == 0):
#            dims = self.shape
#            weight = np.ones(dims,dtype=np.uint8)
#            return {'wimg':weight}
#
#        kern = kern.lower()

