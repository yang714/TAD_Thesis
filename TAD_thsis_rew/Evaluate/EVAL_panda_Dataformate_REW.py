import numpy as np
import  Evaluate.EVAL_Read_File_Image_REW as RFI
from sklearn.utils import shuffle

def Train_data():

 data ,valid ,data_NOT,fake=RFI.read_image()

 X_train=np.concatenate((data,data_NOT))
 Y_train=np.concatenate((valid,fake))
 print(" X_train",X_train.shape)
 print(" Y_train",Y_train.shape)

 X_train, Y_train = shuffle( X_train, Y_train)
 # print(" Y_train",Y_train[0:100])
 return  X_train,Y_train






