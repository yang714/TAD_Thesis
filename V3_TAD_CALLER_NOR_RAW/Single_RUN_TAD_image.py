import numpy as np
import pandas as pd
import cv2
# from PIL import Image,ImageDraw
import os
import TADLASSIFICATION_TAD  as CT
import Call_Origin_domain as CO
import  Detail_of_TADs as DT
# path="D:\TAD_DATA\TAD_image\Hi_C_Matrix_image/"
path="D:\TAD_DATA\TAD\TAD_im/hES/"
# file_name="hIMR_chr_chr1.png"

#RAW-------------------------------------------------------------------------------------------------------------------

# file_name="mES_RAW_ CH2.png"#----------------------------------->testing
file_name="hES_chrX_Nor.png"#----------------------------------->testing
model_path="D:\TAD_DATA\model/TAD_RESNet_Mouse--1-1_E400.h5"
seq=CT.TAD_find(path,file_name,6,1,model_path)
a=seq
# print("SEQ",seq)
#
# # print("SEQ---->",seq[0:3])
# # cv2.line(img, (x, y), (x+w, y), (0, 255, 0), 3)
# image_basic=cv2.imread(path+"mCO_NOR_ CH2_print.png")#----->獨立
# for i in (seq):
#     # print(i[0],"----",i[1])
#     w=i[1]-i[0]
#     # img = cv2.rectangle(image_basic, (i[0], i[0]), (i[1] , i[1]), (0, 100, 255), 2)
#     cv2.line(image_basic, (i[0], i[0]), (i[1], i[0]),(0, 100, 255), 1)
#     cv2.line(image_basic, (i[1] , i[0]), (i[1] , i[1]), (0, 100, 255), 1)
# # img=img[0:600,0:600,:]
# # print(path+"test_1.png")
# cv2.imwrite(path+"mCO_NOR_ CH2_print.png",image_basic)#---->MODEL TAD 之後的
#NOR----------------------------------------------------------------------------------------------------------------------


# cv2.imwrite(path+"Finally_mES_NOR_ CH2_print.png",image_basic)
print("RAW----------------------")
print(len(a))
print("--------------->", a)
print(max(a))
print(min(a))



DT.AVG_and_number(a)