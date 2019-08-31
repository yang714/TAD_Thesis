import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os
import random
import glob

import  Data_Pre.log2_normalize as lg

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten,LeakyReLU,BatchNormalization,Reshape,Input
from keras.models import Sequential, Model
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist


resolution = 40000
image_size = 60
pick_size=0
nonTAD = 64
epochs = 3000
batch_size = 16
sample_interval = 200


# ReshapeToMatrix------------------------------------------

# def  Non_TADBin(chr_pad,matrix):
#     for  j in len( chr_pad):
#          for  k in


def ReshapeToMatrix(path):

    hicdata = pd.read_table(path)
    hicdata[np.isnan(hicdata)] = 0  # -------------nan replace by 0
    hicdata = np.array(hicdata)
    print( "hicdata.shape----------------------->",hicdata.shape)
    # need = max(hicdata.shape) % image_size  # -------------------完美分割大小
    # need = image_size - need  # ----------------------------------
    # # print("need", need)
    #
    # if hicdata.shape[0] != hicdata.shape[1]:
    #     result = np.zeros([max(hicdata.shape) + need, max(hicdata.shape) + need])
    #     # print("re", result.shape)
    #     result[:hicdata.shape[0], :hicdata.shape[1]] = hicdata
    # hic_matrix = result

    if hicdata.shape[0]<hicdata.shape[1]:
        # hicdata=hicdata[0:hicdata.shape[0],0:hicdata.shape[0]]
        hicdata = hicdata[-hicdata.shape[0]:-1, -hicdata.shape[0]:-1]
    # else:
    #     hicdata = hicdata[ start_point:hicdata.shape[1]+1,  start_point:hicdata.shape[1]+1]
    print("hicdata.shape=========================>", hicdata.shape)
    hic_matrix= hicdata
    # hic_matrix=np.triu(hic_matrix)
    return hic_matrix
#--------------------------------------------------------------------------------


#pickTAD-and-NONTAD??-------------------------------------------------------------
def pick_save_TAD(matrix,domain,ch_number,save_path,cellname):
    HiC_Size_list=[]
    matrix_allbin=[]
    Non_TAD_BIN=[]
    ch_pandas_data = domain.loc[domain["ch"] == ch_number].values
    matrix = np.triu(matrix)

    # print("ch_N",ch_number)
    # print(" matrix ",  matrix .shape)
    # print(" LEN ch_pandas_data",len(  ch_pandas_data))
    # print("ch_pandas_data[i][1]--->", "    ", ch_pandas_data[0][1], "ch_pandas_data[i][2]", ch_pandas_data[0][2])
   #---------------------------------------------------------------------------------NON-TAD BIN

    if ch_pandas_data[0][1]>0:#------------------first
        for fist in range(0,ch_pandas_data[0][1]):
            Non_TAD_BIN.append(fist)

    for i in range(len(ch_pandas_data)-1):#middle
       if  ch_pandas_data[i][2]!=ch_pandas_data[i+1][1]:
        for  k  in range(ch_pandas_data[i][2]+1,ch_pandas_data[i+1][1]):#+1!!!
            Non_TAD_BIN.append(k)

    if ch_pandas_data[len(ch_pandas_data)-1][2] > 0:  # -----------------last
        for last in range(ch_pandas_data[len(ch_pandas_data)-1][2]+1, matrix .shape[0]):#+1!!!
            Non_TAD_BIN.append(last)



#--------------------------------------------------------------------------------------
    #--------------------------------------------------------All TAD size
    for i in range(len(ch_pandas_data)):
     TAD_size= ch_pandas_data[i][2]-ch_pandas_data[i][1]
     HiC_Size_list.append( TAD_size)
     # print("ch_pandas_data[i][1]--->",i,"    ",ch_pandas_data[i][1],"ch_pandas_data[i][2]",ch_pandas_data[i][2])
    #---------------------------------------------------------------------------------------
     if abs(TAD_size)>pick_size:
      # print("ch_pandas_data[i][1]-1",ch_pandas_data[i][1]-1,"ch_pandas_data[i][2]",ch_pandas_data[i][2])
      matrix_svae=matrix[ch_pandas_data[i][1]:ch_pandas_data[i][2]+1,ch_pandas_data[i][1]:ch_pandas_data[i][2]+1]
      # print("TAD_size",TAD_size)
      save_p = save_path + "/" + cellname + "_" + ch_number + "_" + str(i) + ".png"
      cv2.imwrite(save_p, matrix_svae)
      image = cv2.imread(save_p)
      res = cv2.resize(image, (image_size, image_size))

      # data_matrix.append(res)
      # print("   data_matrix",  data_matrix)
      cv2.imwrite(save_p, res)
     # shape_data_list = np.unique( shape_data_list)
    # print("   shape_data_list", shape_data_list)
    #-----------------------------------------------

    # s1 = set(matrix_allbin)
    # s2 = set( HiC_Size_list)
    # NON_TAD_Bin=list(s1.symmetric_difference(s2))
    # print("NON_TAD_Bin---->",NON_TAD_Bin)

    #-----------------------------------------------
    return  HiC_Size_list,Non_TAD_BIN


def NON_TAD_bin(matrix,HiC_List_Size,Non_TAD_BIN,No_TAD_OK):
    while No_TAD_OK==False:
        Non_TAD_size_index = random.randint(0, len(HiC_List_Size)-1)
        Non_TAD_size = HiC_List_Size[Non_TAD_size_index]
        Non_TAD_point_index = random.randint(0, len(Non_TAD_BIN)-1)
        Non_TAD_point=Non_TAD_BIN[Non_TAD_point_index ]
        # OneverlapOrCross=random.randint(0, 1)
        # if  OneverlapOrCross==0:#overlap
        RangeOfNONTAD=random.randint(1,  Non_TAD_size)
        # matrix_svae = matrix[a:b, a:b]
        end=Non_TAD_point+ RangeOfNONTAD-1#包含Non_TAD_point
        start=Non_TAD_point-(Non_TAD_size-RangeOfNONTAD)
        if  end<=matrix.shape[0]and start>=0:
            No_TAD_OK=True
        # print("Non_TAD_size---->",Non_TAD_size,"  Non_TAD_point---->   ",Non_TAD_point ,"   start---> ",start,"  end--->",end)
    matrix_svae = matrix[start:end+1, start:end+1]
    return  matrix_svae,Non_TAD_size,Non_TAD_point,start,end

# ----------------------
def pick_save_NOTTAD(matrix,domain,ch_number,save_path,cellname,HiC_List_Size,Non_TAD_BIN ):
    ch_pandas_data = domain.loc[domain["ch"] == ch_number].values
    matrix = np.triu(matrix)
    # matrix_shap=matrix.shape
    # Save_path = "F:\Thsis_TAD\Human/"
    # save_path__NOTtad = Save_path + "/" + "Non-TAD"
    print("ch_N", ch_number)
    count_for_1=0
    count_for_1point5 = 0
    for i in range(int(len(ch_pandas_data)*2)):#<--------------------------TAD Non-TAD比例 比例1:2
        No_TAD_OK=False
        Not_TAD_matrix,Non_TAD_Size,Non_TAD_Point,Start,End=NON_TAD_bin(matrix, HiC_List_Size, Non_TAD_BIN, No_TAD_OK)
        save_p = save_path + "/2/" + cellname + "_" + ch_number + "_" + str(i)+"_2_time_s_"+str(Non_TAD_Size)+"_Start_"+str(Start)+"_End_"+str(End)+"_p_"+str(Non_TAD_Point)+ ".png"

        cv2.imwrite(save_p,  Not_TAD_matrix)
        image = cv2.imread(save_p)
        res = cv2.resize(image, (image_size, image_size))
        cv2.imwrite(save_p, res)
        if   count_for_1point5<int(len(ch_pandas_data)*1.5):#-------------------------------TAD Non-TAD比例 比例1:1.5
            save_p = save_path + "/1.5/" + cellname + "_" + ch_number + "_" + str(i) + "_2_time_s_" + str(
                Non_TAD_Size) + "_Start_" + str(Start) + "_End_" + str(End) + "_p_" + str(Non_TAD_Point) + ".png"
            cv2.imwrite(save_p, Not_TAD_matrix)
            image = cv2.imread(save_p)
            res = cv2.resize(image, (image_size, image_size))
            cv2.imwrite(save_p, res)
        if count_for_1 < int(len(ch_pandas_data) * 1):  # -------------------------------TAD Non-TAD比例 比例1:1
            save_p = save_path + "/1/" + cellname + "_" + ch_number + "_" + str(i) + "_2_time_s_" + str(
                Non_TAD_Size) + "_Start_" + str(Start) + "_End_" + str(End) + "_p_" + str(Non_TAD_Point) + ".png"
            cv2.imwrite(save_p, Not_TAD_matrix)
            image = cv2.imread(save_p)
            res = cv2.resize(image, (image_size, image_size))
            cv2.imwrite(save_p, res)
        count_for_1 =count_for_1+1
        count_for_1point5 = count_for_1point5+1



#-----------------------------------------------------------------------------------------

Char = {"chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13"
    , "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20","chr21","chr22","chrX"}
Cell={"mCO","mES"}
# Cell={"hES","hIMR","mCO","mES"}
# Cell={"hES","hIMR"}
# Cell={"hES"}
# Char = {"chr1", "chr2"}

# Save_path = "F:\Thsis_TAD\Human/"
Save_path = "F:\Thsis_TAD\Mouse/"

for cellname in Cell:
    print("cell_name",cellname)
    PATH="F:/TAD_DATA/total."+cellname+".combined.domain"
    domain = pd.read_table(PATH)
    domain_array = np.array(domain)
    add = []
    for i, j in enumerate(domain.columns):
        if i != 0:
            j = int(j)
        add.append(j)
    domain_array = np.insert(domain_array, 0, add)
    domain_array = domain_array.reshape(domain_array.shape[0] // 3, 3)  # reshape
    domain = pd.DataFrame(data=domain_array, columns=["ch", "star", "end"])
    # print("-------------------------------------------------------------------------------------------")
    domain["star"] = domain["star"] // resolution
    domain["end"] = domain["end"] // resolution
    # ch1_pandas_data = domain.loc[domain["ch"] == "chr1"].values
    PATH_TAD = "F:/TAD_DATA/TAD_image/"+cellname#<-------------------------------TAD Domain

    # HiC_Train_Data = []  # svae_image as matrix array
    # HiC_Train_Shape_list = []
    for j, i in enumerate(Char):
        HiC_Size_list = []
        call_path = PATH_TAD + "/" + "nij." + i
        # print("call_path", call_path)
        if os.path.isfile(call_path)==True:
         # print("call_path", call_path)
         save_path_tad = Save_path + "/" + "TAD"
         save_path__NOTtad = Save_path + "/" + "Non-TAD"
         HiC_Matrix = ReshapeToMatrix(call_path)
         HiC_Matrix = lg.N_HiCData(HiC_Matrix)  # ------------------------>log normalize
         # HiC_Matrix=HiC_Matrix*1.5
         # print("HI-c",HiC_Matrix)
         HiC_TAD_SIZE_list,Non_TAD_BIN = pick_save_TAD(HiC_Matrix, domain, i, save_path_tad, cellname)
         # UniqueHiC_List_Size=np.unique( HiC_TAD_SIZE_list)#-----------------------------------------unique TAD size
         UniqueHiC_List_Size=HiC_TAD_SIZE_list
         pick_save_NOTTAD(HiC_Matrix, domain, i, save_path__NOTtad, cellname, UniqueHiC_List_Size, Non_TAD_BIN)
#--------------------------------------------------------------------
         # pick_save_NOTTAD(HiC_Matrix, domain, i, save_path__NOTtad, cellname, HiC_TAD_SIZE_list,Non_TAD_BIN )
#------------------------------------------------------------------










         # HiC_Train_Data, HiC_Train_Shape = pick_save_NOTTAD(HiC_Matrix, domain, i, save_path__NOTtad, cellname, HiC_Train_Data,
         #                                                 HiC_Train_Shape)
         #pick_save_TAD(matrix,domain,ch_number,save_path,cellname,data_matrix,shape_data):



        # pick_save_NONTAD(HiC_Matrix, domain, i, save_path_nontad)
    # HiC_Train_Data = np.array(HiC_Train_Data)
    # HiC_Train_Shape = np.array(HiC_Train_Shape)
    # TEST = np.expand_dims(HiC_Train_Data, axis=3)
# a=pre_style_gandata()


