import sys
import cv2
import argparse

parser = argparse.ArgumentParser(description='Validate the dimensions of an image.') 
parser.add_argument('-i','--image',type=str,default=None,\
help='required image file',metavar='image file')
parser.add_argument('-wd','--width',type=int,default=None,\
help='required image width',metavar='integer')
parser.add_argument('-ht','--height',type=int,default=None,\
help='required image height',metavar='integer')
parser.add_argument('-v','--verbose',type=int,default=None,\
help='Control print output. Select 1 to print all non-error print output and 0 to suppress all printed output (bar argument-parsing errors).',metavar='0 or 1')

if len(sys.argv) > 1:
    args = parser.parse_args()
    dims = cv2.imread(args.image,cv2.IMREAD_UNCHANGED).shape
    
    if (args.height != dims[0]) | (args.width != dims[1]):
        print("Dimensions of image: {} x {}".format(dims[0],dims[1]))
        print("Queried dimensions: {} x {}".format(args.height,args.width))
        print("Image file {} is not valid!".format(args.image))
        exit(1)
    else:
        print("Image file {} is valid.".format(args.image))
        exit(0)
else:
    parser.print_help()
