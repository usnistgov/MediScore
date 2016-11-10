import numpy as np
import pandas as py
import cv2

execfile('masks.py')

rmask = refmask('../data/test_suite/maskScorerTests/flower-splice-mask.png')
smask = mask('../data/test_suite/maskScorerTests/B_NC2016_Splice_ImgOnly_p-me_1/mask/flower-splice-mask.png')

thres = rmask.runningThresholds(smask,15,9)
#print thres.to_string()

print('Plot ready.')
rmask.getPlot(thres)
