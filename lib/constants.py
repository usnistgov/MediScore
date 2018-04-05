import cv2
import pandas as pd

try:
    png_compress_const=cv2.cv.CV_IMWRITE_PNG_COMPRESSION
except:
    try:
        png_compress_const=cv2.IMWRITE_PNG_COMPRESSION
    except:
        png_compress_const=16

try:
    query_exception = pd.computation.ops.UndefinedVariableError
except:
    query_exception = pd.core.computation.ops.UndefinedVariableError

colordict={'red':[0,0,255],
           'blue':[255,51,51],
           'yellow':[0,255,255],
           'green':[0,207,0],
           'pink':[193,182,255],
           'purple':[211,0,148],
           'white':[255,255,255],
           'gray':[127,127,127]}
