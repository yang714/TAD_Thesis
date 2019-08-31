import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os
import random
import   test  as  teat
import glob

# Cell={"hES","mCO","mES"}
# Save_path = "F:/TAD_DATA/TAD_image/TAD/"
# data_path = os.path.join(Save_path,'*jpg')
# files = glob.glob(data_path)
# data = []
# for filename in files :
#     img = cv2.imread(filename)
#     data.append(img)
# data=np.array(data)
#
# print(data.shape)

def read_image():
    Cell = {"hES", "mCO", "mES"}
    #---------------------------------------------------------------
    # ---------------------------------------------------------------
    # Save_path = "F:/TAD_DATA/TAD_image_human/TAD/"
    # Save_path_NOT="F:/TAD_DATA/TAD_image_human/NOTTAD/"
    # ------------------------------------------------------------------
    #
    # Save_path = "F:\Thsis_TAD\Human/TAD/"
    # Save_path_NOT = "F:\Thsis_TAD\Human/Non-TAD/2/"#要改1 1.5 2
    Save_path = "F:\Thsis_TAD\Mouse/TAD/"
    Save_path_NOT = "F:\Thsis_TAD\Mouse/Non-TAD/1/"  # 要改1 1.5 2
#------------------------CELL
    # Save_path = "F:\Thsis_TAD\Mouse_CO\TAD/"
    # Save_path_NOT = "F:\Thsis_TAD\Mouse_CO/Non-TAD/1/"  # 要改1 1.5 2
    #------------------------------

    data_path = os.path.join(Save_path, '*png')
    files = glob.glob(data_path)
    print(" Training_data_path -->", data_path )
    data = []
    for filename in files:
        img = cv2.imread(filename)
        # img=cv2.resize(img, (60, 60))
        # data.append(img[:,:,0])
        data.append(img)
    data= np.array(data)#-----算valod有幾個
    # valid = np.ones((  data.shape[0], 1))
    valid  = np.ones(( data.shape[0], 1))

    # ---------------------------------------------------------------
    # Save_path_NOT = "F:/TAD_DATA/TAD_image_mouse/NOTTAD/"
    # Save_path = "F:/TAD_DATA/TAD_image_human/NOTTAD/"

    data_path = os.path.join(Save_path_NOT, '*png')
    files = glob.glob(data_path)
    data_NOT = []
    for filename in files:
        img = cv2.imread(filename)
        # img = cv2.resize(img, (60, 60))
        # data_NOT .append(img[:,:,0])
        data_NOT.append(img)
    data_NOT = np.array(data_NOT )#-----算valod有幾個
    # fake = np.zeros((data_NOT.shape[0], 1))
    fake = np.zeros((data_NOT.shape[0], 1))
    return data ,valid ,data_NOT,fake
    # print(data.shape)
