from keras.models import load_model
import matplotlib.pyplot as  plt
from keras.utils import np_utils
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from collections import Counter
import os
import glob
import cv2
from keras.models import load_model
# model=load_model('C:/Users\PC\Desktop\K_FLOAD_MODEL\TMEH\RES/TAD_RRRM_model.h5')#CNN
# model=load_model('C:/Users\PC\Desktop\K_FLOAD_MODEL\TMEH\CNN/TAD_CNNM_model.h5')#CNN
# model = load_model('C:/Users\PC\Desktop\TAD_CNN_mouse_model.h5')
# model = load_model('C:/Users\PC\Desktop\TAD_DenseNet_mouse_model.h5')
# model = load_model('D:\TAD_DATA\model/TAD_RESNet_Human1-t1_E400.h5')
def Median(HiC_Matrix):
    seq=[]
    for  i in range(HiC_Matrix.shape[0]):
        seq.append(HiC_Matrix[i][i])
    media=np.median(seq)
    return   media
def Find_Start_POINT(media,HiC_Matrix):
    startpoint_seq=[]
    # print("  print(media)-->",media)
    for i in range(HiC_Matrix.shape[0]):
       # print("HiC_Matrix[i][o]--->",HiC_Matrix[i][i][0])
       if  HiC_Matrix[i][i][0]>=media:
           startpoint_seq.append(i)
    # media = np.median(seq)
    return  startpoint_seq

def find_thing_seq(seq):
    Seq=[]
    for i in range(0,len(seq)):
        Seq.append(seq[i][1]-seq[i][0])
    return Seq


def finmin_TADSIZE(cellname,chtname):
  resolution=40000
  HiC_Size_list=[]
  # print("chtname",chtname)
  #-----0-------------------
  # print("cell_name", cellname)
  PATH = "D:\TAD_DATA\TAD_image/total." + cellname + ".combined.domain"
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
  ch_pandas_data = domain.loc[domain["ch"] == chtname].values

  for i in range(len(ch_pandas_data)):
     TAD_size= ch_pandas_data[i][2]-ch_pandas_data[i][1]
     HiC_Size_list.append(TAD_size)
  # MIN_SIZE_TAD=min(HiC_Size_list)
  MIN_SIZE_TAD =(HiC_Size_list)
  return  MIN_SIZE_TAD
#--------------------------






def Read_image_Find_TAD(filepath,file_name,Save_Count_Max,move_range,model_path,PREDICET):
    # min_size_TAD=finmin_TADSIZE("hIMR","chr1")
    # print("~~~~~~",min_size_TAD)
    #---------------------------------------------------------------
    model = load_model(model_path)
    # print("filepath+file_name----->", filepath + file_name)
    img = cv2.imread(filepath+file_name)
    Hi_C_Shape=img.shape[0]
    # print("Hi_C_Shape--->",Hi_C_Shape)
    X_test=img


    # X_test= X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2])
    # print("X_test_shape", X_test.shape)
    # print("Cell_char_name",file_name+"______",Hi_C_Shape)
    SE_TAD=[]
    Test_Max=[]
    # media=Median(img)
    End_chr=False

    # defore_TAD=False#--------
    count = 0
    # star_point=Find_Start_POINT(media, img )
    # for start_point in  star_point:
    start_point=0
    # while start_point< Hi_C_Shape-2:
    while End_chr==False:#<--------------------------一張Hi-C map end
        # print("HERE---------------------------------")
        TAD_flag_stop = False
        end_point = 0
        Save_Count=0
        initial_size=5
        comput_size=5
        move_size=3
        is_TAD=0

        TEMP=0
            #TAD_flag_stop   單一TAD    End_chr ALL CHR
        while  TAD_flag_stop ==False  :#<---------------------------------------一個TAD END
         # print("STAR---------------TAD  TAD_flag_stop  ",TAD_flag_stop,"   SAVE",Save_Count)
         image_pre=X_test[start_point: start_point  +move_size, start_point: start_point + move_size]

         image_pre= cv2.resize(image_pre, (60, 60))
         x_image = image_pre.reshape((1, 60, 60, 3))

         x_image=  x_image/255
         # print("???????????????", x_image[0][0])
         y_pred = model.predict(x_image)
         IsTAD_or_NOT = np.argmax(y_pred, axis=1)
         #--------------------------------see 機率>0.75
         # print(y_pred[0][0])
         if  y_pred[0][1]>PREDICET:
             IsTAD_or_NOT=1
         else:
             IsTAD_or_NOT=0
         #----------------------------------------
         # print("IsTAD_or_NOT --->",IsTAD_or_NOT )
         # print("start_point----->",start_point,"  move_size  ",move_size)
         if IsTAD_or_NOT == 1:
             if 0.5 <= y_pred[0][1] <= 0.6:
                 move_range = 4
             elif 0.7 < y_pred[0][1] <= 0.9:
                 move_range= 2
             else:
                 move_range = 1
         if IsTAD_or_NOT  == 1:#1--------->TAD
            # print("start_point + move_size>Hi_C_Shape-1 ", start_point, "-----", move_size)
            #-------------------------------------------------------------------------------add8/20
            # is_TAD=Save_Count

                #-----------------------------------
            TAD_flag_stop=False
            TEMP=[start_point, start_point + move_size]
            if start_point + move_size > Hi_C_Shape - 1:
                # print(" tart_point+ move_size>Hi_C_Shape-1:")
                SE_TAD.append([start_point, Hi_C_Shape - 1])
                Test_Max.append(Hi_C_Shape - 1 - start_point)  # ------------
                TAD_flag_stop = True
                End_chr = True
            move_size = move_size + move_range # <---------------------------------------擴大2

            # move_size = 1 + move_size
            # print("STILL IS TAD")
         else:#1--------->NOT_TAD
            # print("start_point!!!!!!!----->", start_point, "  move_size !!!!! ", move_size)
            if Save_Count<Save_Count_Max:#--------------------------------一次搜TAD的機會
                # print("start_point!!!!!!!----->", start_point,"  SAVE",Save_Count)
                move_size=move_size+move_range
                TAD_flag_stop = False
                Save_Count=Save_Count+1

            else:
                if TEMP!=0:
                    # print("----->",TEMP)
                    SE_TAD.append(TEMP)
                    start_point = TEMP[1]
                    TAD_flag_stop = True
                else:#----------------all non-TAD
                    # print("222222222222----->", TEMP)
                    start_point = start_point + 1
                    TAD_flag_stop = True
                # End_chr = True
            if start_point + move_size > Hi_C_Shape - 1:
                # print("start_point!!!!!!!----->", start_point, "  move_size !!!!! ", move_size,"   TEMP--->",TEMP,"    SAVE",Save_Count)
                # print("!!!!!!!!!!!!!!!!!!!!!!!!----->", TEMP)
                if TAD_flag_stop==False :
                    if TEMP!=0:
                        SE_TAD.append(TEMP)
                        start_point = TEMP[1]
                    if start_point==Hi_C_Shape - 1:
                        TAD_flag_stop = True
                        End_chr = True


    print("SE",len(SE_TAD))
    return  SE_TAD, X_test,Test_Max
    #------------------------------------
    # y_pred = model.predict(X_test[start_point:end_point + 2, start_point:end_point + 2])
    # a = np.argmax(y_pred, axis=1)
    # if a == 0:
    #     TAD_flag = False
    #     print("hi")
    # else:
    #     end_point = end_point + 1
    #     TAD_flag = True
    #     print("no")
#-------------------------------------------------
    # while  start_point<Hi_C_Shape:
    #     TAD_flag=True
    #     while  TAD_flag:
    #         y_pred = model.predict(X_test[start_point:end_point+2,start_point:end_point+2])
    #         a = np.argmax(y_pred, axis=1)
    #         if a==0:
    #             TAD_flag=False
    #         else:
    #             end_point=end_point+1
    #             TAD_flag =True


def TAD_find(path,image_name,Save_Count_Max,move_range,model_path,PREDICET):
    seq, X_test,max_test=Read_image_Find_TAD(path,image_name,Save_Count_Max,move_range,model_path,PREDICET)
    test_seq=find_thing_seq(seq)
    # print(len(seq))
    # print("--------------->",seq)
    # print(max(test_seq))
    # print(min(test_seq))
    return  seq
# print(max_test)









#
# Read_path = "F:\TAD_DATA\TAD_image\Hi_C_Matrix_image/"
# data_path = os.path.join(Read_path , '*jpg')
#
# files = glob.glob(data_path)
# print(" data_path -->", data_path )
# data = []
# for filepath in files:
#     read_file_name = os.path.basename(filepath  )#取最後一個/之後的字串
#     # print("data_path_1",data_path_1)
#     Read_image_Find_TAD(filepath ,read_file_name)
#     # img = cv2.imread(filename)
#     # data.append(img)
# data= np.array(data)#-----算valod有幾個
# valid = np.ones((  data.shape[0], 1))