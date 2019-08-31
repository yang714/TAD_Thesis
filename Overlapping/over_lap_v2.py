import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os
import random
import glob
import numpy as np
from keras.models import Model,load_model

resolution = 40000
image_size = 60
pick_size=0
nonTAD = 64
epochs = 3000
batch_size = 16
sample_interval = 200


Char = {"chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13"
    , "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20","chr21","chr22","chrX"}

# Cell={"hES","hIMR"}
ch_number="chr3"
# Cell="hIMR"
Cell="hES"
Filename="hES_chr3_167_2_time_s_12_Start_4793_End_4804_p_4793.png"
C_A=4793
C_B=4804
model = load_model('C:/Users\PC\Desktop\model/TAD_RESNet_Mouse--1-1_E400.h5')
im=cv2.imread("C:/Users\PC\Desktop\model\TAD_NONTAD\Human/Non-TAD/1/"+Filename)


PATH = "D:\TAD_DATA\TAD/total." + Cell + ".combined.domain"
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
list=[]
ch_pandas_data = domain.loc[domain["ch"] == ch_number].values
for i in range(int(len(ch_pandas_data))):
    temp = []
    # print(ch_pandas_data[i][1])
    if C_B>=ch_pandas_data[i][2]>=C_A or C_A<=ch_pandas_data[i][1]<=C_B :
        temp.append(ch_pandas_data[i][1])
        temp.append(ch_pandas_data[i][2])
        list.append(temp)

print(list)

save=[]
for i in  list:
    print("iii",i)
    A=i[0]
    B=i[1]
    if A >= C_A:
        Big_Start = A
    else:
        Big_Start = C_A
    if B >= C_B:
        Small_END = C_B
    else:
        Small_END = B
    Start = A
    END = B
    overlap=  abs(Big_Start-Small_END) /abs((END-Start))
    save.append(overlap)
    print("A",A)
    print("B",B)
    print("C_A",C_A)
    print("C_B",C_B)
    print("(Start)",(Start))
    print("(END)",(END))
    print("(Big_Start))",(Big_Start))
    print("(Small_END)",(Small_END))

    print("Big_Start-Small_END",(Big_Start-Small_END))
    print("(C_B-C_A)",(C_B-C_A))

    print("(Start-END)",(Start-END))

    print("OVERLAP: " ,overlap)
    print("------------------------------------------------------")
print("SAVE " ,save)
save=np.array(save)
print("SAVE _AVG" ,save.mean())


#--------------------------------------------------------------


im=im/255
# im=np.array(im)
# im = im.reshape(im.shape+(1,1,))
# print(im[0])
im  = im .reshape((1, 60, 60, 3))
# im=im/255
print("-----------------------------------------------------------------")


y_pred = model.predict(im)

print("y_pred:",y_pred)

a = np.argmax(y_pred, axis=1)
print("------->",a)
if a==1:
    print("is TAD")
else:
    print("NOT TAD")