import cv2

try:
    png_compress_const=cv2.cv.CV_IMWRITE_PNG_COMPRESSION
except:
    try:
        png_compress_const=cv2.IMWRITE_PNG_COMPRESSION
    except:
        png_compress_const=16
