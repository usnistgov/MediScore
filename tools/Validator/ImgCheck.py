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

    height = args.height
    width = args.width
    imgName = args.image 

    if (height is None) or (width is None):
        print("ERROR: No width and height specified.")
        exit(1)
    
    if (height != dims[0]) | (width != dims[1]):
        print("Dimensions of image: {} x {}".format(dims[0],dims[1]))
        print("Queried dimensions: {} x {}".format(height,width))
        print("Image file {} is not valid!".format(imgName))
        exit(1)
    else:
        print("Image file {} is valid.".format(imgName))
        exit(0)
else:
    parser.print_help()
