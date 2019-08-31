import numpy as np
import pandas as pd
import cv2



Char = {"chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13"
    , "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20","chr21","chr22","chrX"}
# ch_number="chr3"
# Cell="hES"
def  CTCF_Point(mESCotCO,chr_number):
    data = pd.read_csv("C:/Users\PC\Desktop\CTCF/"+mESCotCO+".ctcf.peak.txt", delimiter = "\t", header = None)
    # data=pd.read_table("C:/Users\PC\Desktop\CTCF/cortex.ctcf.peak.txt", header = None)
    # print(data[0])
    data_chr=data[data[0].values==chr_number][:]
    data_chr_ctcf=data_chr[1][:]//40000
    # print( data_chr_ctcf)
    CTCF_point=[]
    for i in data_chr_ctcf:
        CTCF_point.append(i)
    # print(len(  CTCF_point))
    CTCF_point_LIST=np.unique(  CTCF_point)
    # print("u",len(  CTCF_point_LIST))
    return  CTCF_point_LIST
# point=[]
# for i in range(0,len(data)):
#     if data[][]
# test=CTCF_Point("mESC","chr3")
# print(len(test))
# for i in test:
#     print(i)

# sum=0
# for  ctcf_in in  Char:
#     test=CTCF_Point("mCO", ctcf_in)
#     print("mES--->"+"chr_"+ctcf_in+"---------> ", len(test))
#     sum=len(test)+sum
# print("sum",sum)