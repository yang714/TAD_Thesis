import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os
import random
import glob
import creat_TAD_image.log2_normalize as LN
# resolution = 1000
resolution = 40000
image_size =60
pick_size=6
nonTAD = 64
epochs = 3000
batch_size = 16
sample_interval = 200

def ReshapeToMatrix(path):

    hicdata = pd.read_table(path)
    hicdata[np.isnan(hicdata)] = 0  # -------------nan replace by 0
    hicdata = np.array(hicdata)
    if hicdata.shape[0] < hicdata.shape[1]:
        hicdata = hicdata[0:hicdata.shape[0], 0:hicdata.shape[0]]

    hic_matrix= hicdata
    hic_matrix= np.triu(hic_matrix)#########取對角
    return hic_matrix

#
# Char = {"chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13"
#     , "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20","chr21","chr22","chrX"}
Char = {"chr1"}

# Cell={"hES","hIMR","mCO","mES"}
# Cell={"hES","mCO","mES"}
Cell={"hIMR"}
Save_path = "D:/TAD_DATA/TAD_image"
for cellname in Cell:

    # PATH_TAD = "D:/TAD_DATA/TAD_image/"+cellname
    PATH_TAD = "D:\TAD_DATA/ch125000.txt"#----------------------------------->testing

    HiC_Train_Data = []  # svae_image as matrix array
    HiC_Train_Shape = []
    for j, i in enumerate(Char):
        # call_path = PATH_TAD + "/" + "nij." + i
        call_path= PATH_TAD
        # print("call_path", call_path)
        if os.path.isfile(call_path)==True:
         # print("call_path", call_path)
         HiC_Matrix = ReshapeToMatrix(call_path)
         HiC_Matrix=LN.N_HiCData(HiC_Matrix)
         # HiC_Matrix=HiC_Matrix/255
         # cv2.imwrite(SAVE_path + "im(" + str(count) + ").jpg", image_TAD)
         # cv2.imwrite(Save_path+"/Hi_C_Matrix_image/" +cellname+"_chr_"+i+ ".png", HiC_Matrix)
         cv2.imwrite(Save_path + "/Hi_C_Matrix_image/"  + "_test_" + i + ".png", HiC_Matrix)#----------------------------------->testing

