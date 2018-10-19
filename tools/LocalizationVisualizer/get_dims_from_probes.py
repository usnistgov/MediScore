import numpy as np
import pandas as pd
import cv2
import glymur
import rawpy
import sys
import os
import argparse
libdir=os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../lib')
sys.path.append(libdir)
import masks

def get_image_dims(imgname):
    mymask = masks.mask(imgname)
    try:
        dims = mymask.get_dims()
    except:
        print("{} could not be read successfully.".format(imgname))
        dims = [np.nan,np.nan]
    return dims

def get_image_dims_wrapper(maskrow,dsdir,task='manipulation'):
    #set for each of the relevant probes
    if task == 'manipulation':
        modes = ['Probe','ProbeBitPlane']
    elif task == 'splice':
        modes = ['Probe','Donor','BinaryProbe']

    for m in modes:
        mask_field = "{}MaskFileName".format(m)
        mask_w_field = "{}MaskWidth".format(m)
        mask_h_field = "{}MaskHeight".format(m)
        maskname = maskrow[mask_field]
        if maskname != '':
            dims = get_image_dims(os.path.join(dsdir,maskrow[mask_field]))
        else:
            dims = [np.nan,np.nan]
        maskrow[mask_w_field] = dims[1]
        maskrow[mask_h_field] = dims[0]

    return maskrow

if __name__ == '__main__':
    #apply get_image_dims over a set of file names and return their width and height
    parser = argparse.ArgumentParser(description='Get the dimensions of the images involved with the task.')
    parser.add_argument('-t','--task',type=str,default='manipulation',
        help='Two different types of tasks: [manipulation] and [splice]',metavar='character')
    parser.add_argument('--refDir',type=str,
        help='Dataset directory path',metavar='character')
    parser.add_argument('-r','--inRef',type=str,
        help='Reference csv file name relative to --refDir',metavar='character')
    parser.add_argument('-x','--inIndex',type=str,
        help='Task Index csv file name relative to --refDir. The index file can be customized by the user depending on the probes to display.',metavar='character')

    parser.add_argument('-oR','--outRoot',type=str,
        help="Directory root plus prefix to save outputs. Will save as [directory]/[prefix]_refdims.csv separated by pipes '|'.",metavar='character')
    parser.add_argument('--filter',action='store_true',
        help="Filter out rows with empty masks (since they would not apply to dimensionality discrepancy anyway)")
    parser.add_argument('-p','--processors',type=int,default=1,
        help="Number of processors to parallelize the dataframe's dimension comprehension.",metavar="integer")

    args = parser.parse_args()

    refname = os.path.join(args.refDir,args.inRef)
    ref = pd.read_csv(refname,sep="|",header=0,na_filter=False)
    #preset the dims for each
    if args.task == 'manipulation':
        modes = ['Probe','ProbeBitPlane']
        wh = ["ProbeWidth","ProbeHeight"]
    elif args.task == 'splice':
        modes = ['Probe','Donor','BinaryProbe']
        wh = ["ProbeWidth","ProbeHeight","DonorWidth","DonorHeight"]

    if args.filter:
        queries = ["%sMaskFileName!=''" % m for m in modes]
        fullquery = " | ".join(queries)
        ref = ref.query(fullquery)

    idxname = os.path.join(args.refDir,args.inIndex)
    idx = pd.read_csv(idxname,sep="|",header=0,na_filter=False)
    ref = ref.merge(idx,how='inner')

    for m in modes:
        mask_w_field = "{}MaskWidth".format(m)
        mask_h_field = "{}MaskHeight".format(m)
        ref[mask_w_field] = np.nan
        ref[mask_h_field] = np.nan

    #set the dims for each
    #TODO: parallelize this
    ref = ref.apply(get_image_dims_wrapper,axis=1,dsdir = args.refDir,task = args.task)
    ref["ProbeValid"] = ''
    if args.task == 'splice':
        ref['DonorValid'] = ''
    ref['Comments'] = ''
    ref['Dataset'] = os.path.basename(args.refDir)
    ref.to_csv("_".join([args.outRoot,"refdims.csv"]),sep="|",index=False)

