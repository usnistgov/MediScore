import sys
import os
import cv2
import numpy as np
import math
import pandas as pd
import cv2
import glymur

#split the bits into a list of its consituent bits
def expand_bits(n,bitPlaces=False):
    if n == 0:
        return []
    top_bit = 1
    if n > 0:
        top_bit = int(math.log(n,2)) + 1
    else:
        print("The integer {} put into expand_bits is not unsigned.".format(n))
        return -1
    all_bits = range(0,top_bit)
    bitlist = []
#    try:
    for b in all_bits:
        myb = 1 << b
        bt = type(myb)
        n = bt(n)
        if (myb & n) != 0:
            mybit = str(myb)
            if bitPlaces:
                mybit = str(b + 1)
            bitlist.append(mybit)
#    except:
#        exc_type,exc_obj,exc_tb = sys.exc_info()
#        print("Exception {} at line {}.".format(exc_type,exc_tb.tb_lineno))
#        raise
    bitlist = bitlist[::-1]
    return bitlist

#function takes in a matrix and outputs distinct BitPlane sets. Use a separate function to turn the BitPlane sets into BitPlanes
def bp_sets_from_mask(mask,bitPlaces=False):
    is_multi_layer = len(mask.shape) > 2
    if is_multi_layer:
        singlematrix = np.zeros((mask.shape[0],mask.shape[1]),dtype=np.uint8)
        n_layers = mask.shape[2]
        for l in range(n_layers):
            const_factor = 1 << 8*l
            singlematrix = singlematrix + const_factor*mask[:,:,l]
        distincts = np.unique(singlematrix)
    else:
        distincts = np.unique(mask)

    bitplanelist = []
    for n in distincts:
        blist = expand_bits(n,bitPlaces)
        if blist is -1:
            print("Error: Encountered negative bit {}.".format(n))
            exit(1)
        bits = ','.join(blist)
        bitplanelist.append(bits)
    return bitplanelist
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Take a png or JPEG2000 and dump the list of distinct colors or values into a csv file.')
    
    #get an image or list of images
    #parser.add_argument('-imgs','--imageNames',nargs='*',default=None,\
    #                    help='The name of the image or a list of images for recording colors.',metavar='name or names of png or JPEG2000 file')
    parser.add_argument('-imgf','--imageNames',default=None,\
                        help='The name of the image or a list of images for recording colors.',metavar='name or names of png or JPEG2000 file')
    parser.add_argument('-csv','--csvName',type=str,default='colors.csv',
                        help='The name of the file to output the color data.',metavar='valid csv file name')
    parser.add_argument('-xw','--excludeWhite',action='store_true',\
                        help='Exclude white from the table.')
    parser.add_argument('-bp','--bitPlaces',action='store_true',\
                        help='Record the colors in the JPEG2000 files as their bit places.')
    parser.add_argument('-agg','--aggregate',action='store_true',
                        help='Aggregate the bits recorded into a single table. The table will be produced in addition to the table produced. This option is only applicable to JPEG2000 color counting.')
    parser.add_argument('-p','--processors',type=int,default=1,\
                        help='The number of processors to use to compute the colors in each image.')
    
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        parser.print_help()
        exit(0)
    
    fileName = args.imageNames
    colordfs = []
    aggcolordfs = []
    
    #pass in filename instead
    with open(fileName,'r') as f:
        line = f.readline()
        if line == '':
            print("No images to filter for colors.")
            os.system("echo \"FileName|Ch1Value|Ch2Value|Ch3Value|Ch1BitPlanes\" > {}".format(args.csvName))
            exit(0)
    
    #if len(fileNames) == 0:
    #    print("No images to filter for colors.")
    #    os.system("echo \"FileName|Ch1Value|Ch2Value|Ch3Value|Ch1BitPlanes\" > {}".format(args.csvName))
    #    exit(0)
    
    #read in the masks and list the colors in the mask
    #TODO: parallelize this if at all possible
    colordf_cols = ['FileName','Ch1Value','Ch2Value','Ch3Value','Ch1BitPlanes'] 
    #TODO: debug
    os.system("echo DEBUG > /tmp/tmp.log")
    #for myFileName in fileNames:
    with open(fileName,'r') as f:
        for i,myFileName in enumerate(f):
            myFileName = myFileName.strip()
            if not (os.path.isfile(os.path.realpath(myFileName))):
                print("Error: Expected {}. The file does not exist. Please check to see if it was named correctly.".format(myFileName))
                #TODO: debug
                os.system("echo 'File: {} not found' >> /tmp/tmp.log".format(myFileName))
                continue
            #TODO: debug
            os.system("echo 'File: {}' >> /tmp/tmp.log".format(myFileName))
            
            myFileExt = myFileName.split('.')[-1]
            mymask = 0
            if myFileExt == 'png':
                mymask = cv2.imread(myFileName)
                mysinglechannel = 256*256*mymask[:,:,0] + 256*mymask[:,:,1] + mymask[:,:,2]
                distincts = np.unique(mysinglechannel)
                mask_color_list = [[],[],[]]
                for c in distincts:
                    mask_color_list[0].append(c % 256)
                    mask_color_list[1].append(c % (256*256) // 256)
                    mask_color_list[2].append(c // (256*256))
            
                colordf = pd.DataFrame({'FileName':os.path.basename(myFileName),
                                        'Ch1Value':mask_color_list[0],
                                        'Ch2Value':mask_color_list[1],
                                        'Ch3Value':mask_color_list[2],
                                        'Ch1BitPlanes':''})
        
                if args.excludeWhite:
                    colordf = colordf.query("(Ch1Value != 255) or (Ch2Value != 255) or (Ch3Value != 255)")
        
                colordf = colordf[colordf_cols]
                colordfs.append(colordf)
            
            elif myFileExt == 'jp2':
                mymask = glymur.Jp2k(myFileName)[:]
                #move the below into its own function. 
                bitplanelist = bp_sets_from_mask(mymask,bitPlaces=args.bitPlaces)
            
                aggset = set()
                for blist in bitplanelist:
                    aggset = aggset.union(blist.split(','))
                
                colordf = pd.DataFrame({'FileName':os.path.basename(myFileName),
                                        'Ch1Value':'',
                                        'Ch2Value':'',
                                        'Ch3Value':'',
                                        'Ch1BitPlanes':bitplanelist})
        
                if args.excludeWhite:
                    colordf = colordf.query("Ch1BitPlanes != ''")
                    if '' in aggset:
                        aggset.remove('')
                colordf = colordf[colordf_cols]
                colordfs.append(colordf)
                if args.aggregate:
                    aggcolordf = pd.DataFrame({'FileName':os.path.basename(myFileName),
                                               'Ch1Value':'',
                                               'Ch2Value':'',
                                               'Ch3Value':'',
                                               'Ch1BitPlanes':sorted(aggset)})
                    aggcolordf = aggcolordf[colordf_cols]
                    aggcolordfs.append(aggcolordf)
            
            else:
                print("Error: the image file is neither a png nor a JPEG2000.")
                exit(1)
    
    if len(colordfs) == 0:
        os.system("echo \"FileName|Ch1Value|Ch2Value|Ch3Value|Ch1BitPlanes\" > {}".format(args.csvName))
        exit(0)
    
    colordf_all = pd.concat(colordfs)
    colordf_all.to_csv(path_or_buf=args.csvName,sep="|",index=False)
    
    if args.aggregate and len(aggcolordfs) > 0:
        aggdf_all = pd.concat(aggcolordfs)
        mypath = os.path.dirname(args.csvName)
        mybase = os.path.basename(args.csvName).split('.')[0]
        aggcsvname = '_'.join([os.path.join(mypath,mybase),'agg.csv'])
        aggdf_all.to_csv(path_or_buf=aggcsvname,sep="|",index=False)
