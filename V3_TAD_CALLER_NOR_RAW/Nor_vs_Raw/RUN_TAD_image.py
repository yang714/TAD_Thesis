import numpy as np
import pandas as pd
import cv2
# from PIL import Image,ImageDraw
import os
import with_CTCF.TADLASSIFICATION_TAD  as CT
import Call_Origin_domain as CO
import  Detail_of_TADs as DT

# path="D:\TAD_DATA\TAD_image\Hi_C_Matrix_image/"
path="C:/Users\PC\Desktop\RAW_NOR/nor/"
# file_name="hIMR_chr_chr1.png"
Chr="mES"
cell="chr7"
file_name=Chr+"_"+cell+".png"
#-------------------------------------------------------------------------------------------------------------------

# file_name="mES_RAW_ CH2.png"#----------------------------------->testing
file_name_nor="mES_Chr_"+cell+"_Nor.png"#----------------------------------->testing
model_path="C:/Users\PC\Desktop\model/TAD_RESNet_Human--1-1_E400.h5"
IM=cv2.imread(path+file_name_nor)
print(IM.shape)
seq=CT.TAD_find(path,file_name_nor,11,1,model_path,0.95)
Seq_model_TAD_NOR=seq
print("SEQ",seq)
path_paint="C:/Users\PC\Desktop\RAW_NOR\print/"
# print("SEQ---->",seq[0:3])
# cv2.line(img, (x, y), (x+w, y), (0, 255, 0), 3)
image_basic=cv2.imread(path_paint+file_name_nor)#----->獨立
for i in (Seq_model_TAD_NOR):
    # print(i[0],"----",i[1])
    w=i[1]-i[0]
    # img = cv2.rectangle(image_basic, (i[0], i[0]), (i[1] , i[1]), (0, 100, 255), 2)
    cv2.line(image_basic, (i[0], i[0]), (i[1], i[0]),(255, 0, 255), 1)
    cv2.line(image_basic, (i[1] , i[0]), (i[1] , i[1]), (255, 0, 255), 1)
# img=img[0:600,0:600,:]
# print(path+"test_1.png")
path_temp="C:/Users\PC\Desktop\RAW_NOR\Temp_Save/"
cv2.imwrite(path_temp+"print_temp"+file_name,image_basic)#---->MODEL TAD 之後的
#NOR----------------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------origin
file_name_raw="mES_Chr_"+cell+"_Raw.png"#----------------------------------->testing
path_raw="C:/Users\PC\Desktop\RAW_NOR/raw_with_pre/"
model_path="C:/Users\PC\Desktop\model/TAD_RESNet_Human--1-1_E400.h5"
IM=cv2.imread(path_raw+file_name_raw)
print(IM.shape)
seq=CT.TAD_find(path_raw,file_name_raw,11,1,model_path,0.95)
Seq_model_TAD_Raw=seq
print("SEQ",Seq_model_TAD_Raw)
# path_paint="C:/Users\PC\Desktop\RAW_NOR\print/"
path_temp="C:/Users\PC\Desktop\RAW_NOR\Temp_Save/"
# print("SEQ---->",seq[0:3])
# cv2.line(img, (x, y), (x+w, y), (0, 255, 0), 3)
image_basic=cv2.imread(path_temp+"print_temp"+file_name)#----->獨立
for i in (Seq_model_TAD_Raw):
    # print(i[0],"----",i[1])
    w=i[1]-i[0]
    # img = cv2.rectangle(image_basic, (i[0], i[0]), (i[1] , i[1]), (0, 100, 255), 2)
    cv2.line(image_basic, (i[0], i[0]), (i[0], i[1]), (0, 255, 0), 1)
    cv2.line(image_basic, (i[0], i[1]), (i[1], i[1]), (0, 255, 0), 1)
# img=img[0:600,0:600,:]
# print(path+"test_1.png")
path_temp="C:/Users\PC\Desktop\RAW_NOR\Temp_Save/"
cv2.imwrite(path_temp+"print_temp"+file_name,image_basic)#---->MODEL TAD 之後的
image_basic=cv2.imread(path_temp+"print_temp"+file_name)#----->獨立
#------------------------------------------------------------------------------------CTCF
# TAD_domain_seq= CO.TAD_domain(Chr,cell,40000)#cell chr resoluation
# image_basic=cv2.imread(path_temp+"print_temp"+file_name)#----->獨立
# # image_basic = cv2.cvtColor(image_basic,cv2.COLOR_BGR2RGB)
# for i in (TAD_domain_seq):
#     # print(i[0],"----",i[1])
#     # w=i[1]-i[0]
#     img = cv2.rectangle(image_basic, (i[0], i[0]), (i[1] , i[1]), (255, 0, 0), 1)
#     # cv2.line(image_basic, (i[0], i[0]), (i[0], i[1]), (0, 255, 0), 1)
#     # cv2.line(image_basic, (i[0], i[1]), (i[1], i[1]), (0, 255, 0), 1)
# cv2.imwrite(path_temp+"print_temp"+file_name,image_basic)#---->MODEL TAD 之後的
# image_basic=cv2.imread(path_temp+"print_temp"+file_name)#----->獨立
#------------------------------------------------------------------------------------






#------------------------------------------------------------------



path_result="C:/Users\PC\Desktop\RAW_NOR\Result_Save/"
cv2.imwrite(path_result+"NEW_rawpre_Finally_"+file_name,image_basic)
print("---------------------")
# print(len(Seq_model_TAD_NOR))
print("--------------->", Seq_model_TAD_NOR)
# print(max(Seq_model_TAD_NOR))
# print(min(Seq_model_TAD_NOR))

print("NOR----------------------")

# print(len(b))
# print("--------------->", b)
# print(max(b))
# print(min(b))
print("detailraw--------------------------------")
print("detailNOR--------------------------------")

DT.AVG_and_number(Seq_model_TAD_NOR)


print("detailNOR--------------------------------")
print("DI--------------------------------")
# DT.AVG_and_number(TAD_domain_seq)
#-------------------
print("detailRAW*****************************************")
DT.AVG_and_number(Seq_model_TAD_Raw)


print("detailRAW--------------------------------")
#


