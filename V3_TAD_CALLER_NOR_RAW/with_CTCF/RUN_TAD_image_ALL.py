import numpy as np
import pandas as pd
import cv2
# from PIL import Image,ImageDraw
import os
import with_CTCF.TADLASSIFICATION_TAD  as CT
import Call_Origin_domain as CO
import  Detail_of_TADs as DT
import with_CTCF.READ_CTCF as CTCF
import with_CTCF.Evaluate_Two_method as ETM
# path="D:\TAD_DATA\TAD_image\Hi_C_Matrix_image/"
# path="C:/Users\PC\Desktop\CTCF\Test_mco/"
path="C:/Users\PC\Desktop\CTCF\Test_Mes/"
# file_name="hIMR_chr_chr1.png"
Chr="mCO"
cell="chrX"
cell_LI= {"chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13"
    , "chr14", "chr15", "chr16", "chr17", "chr18", "chr19","chrX"}
Chr="mES"
Chr_2={"mES"}
TAD_average_size_totall=0
TAD_number_totall=0
model_count_totall=0


TAD_average_size_totall_DI=0
TAD_number_totall_DI=0
method_count_totall_DI=0

count_match_TOTALL=0
count_miss_match_TOTALL=0
count_match_TOTALL_D=0
count_miss_match_TOTALL_D=0


for cellname in Chr_2:
    for cell in(cell_LI):
#-------------------------------------------------------------------------------------------------------------------

# file_name="mES_RAW_ CH2.png"#----------------------------------->testing
        file_name="mES_Chr_"+cell+"_Nor.png"#----------------------------------->testing
        model_path="C:/Users\PC\Desktop/New_Thsis\Model_save/TAD_RESNet_Human--1-108_28_E300.h5"
        IM=cv2.imread(path+file_name)
        print(IM.shape)
        seq=CT.TAD_find(path,file_name,14,2,model_path,0.85)
        Seq_model_TAD_NOR=seq
        # print("SEQ",seq)
        path_paint="C:/Users\PC\Desktop\CTCF\Test_Mes\print/"
        # print("SEQ---->",seq[0:3])
        # cv2.line(img, (x, y), (x+w, y), (0, 255, 0), 3)
        image_basic=cv2.imread(path_paint+file_name)#----->獨立
        for i in (Seq_model_TAD_NOR):
            # print(i[0],"----",i[1])
            w=i[1]-i[0]
            # img = cv2.rectangle(image_basic, (i[0], i[0]), (i[1] , i[1]), (0, 100, 255), 2)
            cv2.line(image_basic, (i[0], i[0]), (i[1], i[0]),(255, 255, 0), 1)
            cv2.line(image_basic, (i[1] , i[0]), (i[1] , i[1]), (255, 255, 0), 1)
        # img=img[0:600,0:600,:]
        # print(path+"test_1.png")
        path_temp="C:/Users\PC\Desktop\CTCF\Test_Mes\Temp_Save/"
        cv2.imwrite(path_temp+"print_temp"+file_name,image_basic)#---->MODEL TAD 之後的
        #NOR----------------------------------------------------------------------------------------------------------------------


        #-------------------------------------------------------------------------------origin
        TAD_domain_seq= CO.TAD_domain(Chr,cell,40000)#cell chr resoluation
        image_basic=cv2.imread(path_temp+"print_temp"+file_name)#----->獨立
        # image_basic = cv2.cvtColor(image_basic,cv2.COLOR_BGR2RGB)
        for i in (TAD_domain_seq):
            # print(i[0],"----",i[1])
            # w=i[1]-i[0]
            # img = cv2.rectangle(image_basic, (i[0], i[0]), (i[1] , i[1]), (255, 0, 0), 1)
            cv2.line(image_basic, (i[0], i[0]), (i[0], i[1]), (255, 0, 0), 1)
            cv2.line(image_basic, (i[0], i[1]), (i[1], i[1]), (255, 0, 0), 1)

        #------------------------------------------------------------------------------------CTCF
        ctcf_point_list=CTCF.CTCF_Point(Chr+"C",cell)
        cv2.imwrite(path_temp+"print_temp"+file_name,image_basic)#---->MODEL TAD 之後的
        image_basic=cv2.imread(path_temp+"print_temp"+file_name)#----->獨立
        for i in (ctcf_point_list):
            # print(i[0],"----",i[1])
            # w=i[1]-i[0]
            # img = cv2.rectangle(image_basic, (i[0], i[0]), (i[1] , i[1]), (255, 0, 0), 1)
            cv2.line(image_basic, (i, i), (i, i), (0, 255, 0), 1)
            cv2.line(image_basic, (i, i), (i, i), (0, 255, 0), 1)







        #------------------------------------------------------------------

        # print("CTCF-------------------------------------------------------------")
        save_CTCF=[]
        for i in ctcf_point_list:
            save_CTCF.append(i)
        # print("--------------->", save_CTCF)
        # print("--------------->", len(save_CTCF))
        # print("CTCF--------------------------------")

        path_result="C:/Users\PC\Desktop\CTCF\Test_Mes\Result_Save/"
        cv2.imwrite(path_result+"Finally_"+file_name,image_basic)
        print(cellname + "   ------------------------------------" + cell + "------emd")
        # print("---------------------")
        # print(len(Seq_model_TAD_NOR))
        # print("--------------->", Seq_model_TAD_NOR)
        # print(max(Seq_model_TAD_NOR))
        # print(min(Seq_model_TAD_NOR))

        # print("NOR----------------------")

        # print(len(b))
        # print("--------------->", b)
        # print(max(b))
        # print(min(b))
        # print("detailraw--------------------------------")
        # print("detailNOR--------------------------------")

        TAD_average_size,TAD_number=DT.AVG_and_number(Seq_model_TAD_NOR)
        model_count,count_match,count_miss_match=ETM.comparefunction(Seq_model_TAD_NOR,save_CTCF,IM.shape[0]-1)
        # print("model_count",model_count)
        TAD_average_size_totall=TAD_average_size+TAD_average_size_totall
        TAD_number_totall=TAD_number_totall+TAD_number
        model_count_totall=model_count_totall+model_count
        # print("detailNOR--------------------------------")
        count_match_TOTALL = count_match + count_match_TOTALL
        count_miss_match_TOTALL = count_miss_match_TOTALL + count_miss_match

        # print("detailDI--------------------------------")
        # print("--------------->", TAD_domain_seq)
        TAD_average_sizeD, TAD_numberD=DT.AVG_and_number(TAD_domain_seq)
        method_countD,count_match_D,count_miss_match_D=ETM.comparefunction(TAD_domain_seq,save_CTCF,IM.shape[0]-1)
        TAD_average_size_totall_DI = TAD_average_sizeD+ TAD_average_size_totall_DI
        TAD_number_totall_DI  = TAD_number_totall_DI + TAD_numberD
        method_count_totall_DI  = method_count_totall_DI + method_countD
#------------------------------------------------------------------------------
        count_match_TOTALL_D=count_match_D+count_match_TOTALL_D
        count_miss_match_TOTALL_D=count_miss_match_TOTALL_D+count_miss_match_D
        # print("method_count",method_count)
        # print("detailDI--------------------------------")
        # DT.AVG_and_number(b)
        #-------------------

        #

# TAD_average_size_totall=TAD_average_size+TAD_average_size_totall
# TAD_number_totall=TAD_number_totall+TAD_number
# model_count_totall=model_count_totall+model_count
print("TAD_average/20:",TAD_average_size_totall/20)
print("TAD_number_totall:",TAD_number_totall)
print("model_count_totall:",model_count_totall)
print("precision:",model_count_totall/(TAD_number_totall))
print("count_match",count_match_TOTALL)
print("count_match_miss",count_miss_match_TOTALL)
print("DI---------------------------------------------------------------------")




print("TAD_average/20:",TAD_average_size_totall_DI/20)
print("TAD_number_totall:",TAD_number_totall_DI)
print("model_count_totall:",method_count_totall_DI )
print("precision:",method_count_totall_DI /(TAD_number_totall_DI))
print("count_match",count_match_TOTALL_D)
print("count_match_miss",count_miss_match_TOTALL_D)
print("---------------------------------------------------------------------")