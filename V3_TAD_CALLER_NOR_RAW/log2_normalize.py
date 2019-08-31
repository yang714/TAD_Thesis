import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os
import random
import   test  as  teat
import glob
import math


def findmaxmin(data):
    findseq=[]
    for i in  data:
        for eatch_in_i in i:
            eatch_in_i=eatch_in_i +1
            eatch_in_i= math.log2(eatch_in_i)
            findseq.append(eatch_in_i)
    print("max(findmax)---->",max(findseq))
    maxf=max(findseq)
    print("min(findmax)---->", min(findseq))
    minf= min(findseq)
    findseq=np.array(findseq)
    findseq=np.reshape(findseq,(data.shape[0],data.shape[1]))
    return maxf,minf,findseq

def N_HiCData(data):
    max_s, min_s, log2_hicdata = findmaxmin(data)
    new_hic_data = (log2_hicdata / max_s) * 255  # 0~255
    # new_hic_data = (log2_hicdata / max_s) # 0~1
    return   new_hic_data

