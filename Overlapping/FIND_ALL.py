import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os
import random
import glob
import numpy as np
from keras.models import Model,load_model
from keras.models import Model,load_model
# Filename="mES_chr10_31_2_time_s_28_Start_2970_End_2997_p_2971.png"
model = load_model('C:/Users\PC\Desktop\model/TAD_RESNet_Human--1-1_E400.h5')
data_path=os.path.join("C:/Users\PC\Desktop\model\TAD_NONTAD\Mouse/TAD/","*png")
# data_path=os.path.join("C:/Users\PC\Desktop\model\TAD_NONTAD\Human\TAD/","*png")
file=glob.glob(data_path)
for i in file:
    im = cv2.imread(i)

    im = im / 255
    im = im.reshape((1, 60, 60, 3))
    # print("y_pred:", y_pred)
    y_pred = model.predict(im)
    a = np.argmax(y_pred, axis=1)
    # print("------->", a)
    if a == 0:
        print("is  TAD")
        print("iii--->",i)
    # else:
    #     print("NOT TAD")