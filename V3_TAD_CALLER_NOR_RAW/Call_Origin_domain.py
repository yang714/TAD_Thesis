import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os
import random
import glob






Cell={"hES","mCO"}

def ReSet(pandas_data):
    TEMP=[]
    for i in range(len(pandas_data)):
        set = []
        set.append(pandas_data[i][1])
        set.append(pandas_data[i][2])
        TEMP.append(set)
    # set=np.array(set)
    # uniqueItems = np.unique(set)

    # TEMP= np.array(TEMP)
    # uniqueTEMP = np.unique(TEMP)
    # print(uniqueItems)
    return   TEMP


def  TAD_domain(cellname,chr_number,resolution):
    # Char = {"chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13"
    #     , "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX"}
    # print("cell_name",cellname)
    PATH="D:\TAD_DATA\TAD_image/total."+cellname+".combined.domain"
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
    # print(domain.loc[domain["ch"] == "chr1"].iloc[[2]])  # call chr2 地01"2"個
    # print(domain.loc[domain["ch"] == "chr1"].iloc[[3]])
    ch1_pandas_data = domain.loc[domain["ch"] == "chr1"].values
    # print("ch1_pandas_data", (ch1_pandas_data[0][1]), "---", ch1_pandas_data[0][2])
    PATH_TAD = "F:/TAD_DATA/TAD_image/"+cellname
    ch_pandas_data = domain.loc[domain["ch"] == chr_number].values
    CHR_SET=ReSet(ch_pandas_data)
    # print( CHR_SET )
    return CHR_SET


# TAD_domain_set= TAD_domain("hIMR","chr1",40000)
# print( "----",TAD_domain_set[0] )

