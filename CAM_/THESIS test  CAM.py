from __future__ import print_function
from keras.layers import Input, Dense, LSTM, MaxPooling1D, Conv1D, TimeDistributed, Flatten, AveragePooling1D, \
    BatchNormalization, Activation, GlobalAveragePooling1D, GlobalAveragePooling2D, Permute, multiply, Reshape,Conv2D,Dropout,activations
from keras.models import Model
import matplotlib.pyplot as  plt
from keras.models import Model,load_model
from keras.utils import np_utils
import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau,EarlyStopping,TensorBoard
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import plot_model
# from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
# from numba import cuda
import keras
import cv2
from vis.visualization import visualize_cam
from vis.utils import utils
# from numpy.lib.arraypad import validate_lengths
import skimage


# flist = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee', 'Cricket_X', 'Cricket_Y', 'Cricket_Z',
# 'DiatomSizeReduction', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR', '50words', 'FISH', 'Gun_Point', 'Haptics',
# 'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MALLAT', 'MedicalImages', 'MoteStrain', 'NonInvasiveFatalECG_Thorax1',
# 'NonInvasiveFatalECG_Thorax2', 'OliveOil', 'OSULeaf', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'SwedishLeaf', 'Symbols',
# 'synthetic_control', 'Trace', 'TwoLeadECG', 'Two_Patterns', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer', 'WordsSynonyms', 'yoga']


# mCO_chr1_86.png
# hIMR_chr10_12.png
# im=cv2.imread("C:/Users\PC\Desktop\K_FLOAD_MODEL\TAD_image_mouse/NOTTAD/mCO_chr4_39.png")
# Filename="hES_chr15_63_2_time_s_35_Start_2282_End_2316_p_2313.png"#---->PRE TAD

# Filename="hIMR_chr2_169_2_time_s_81_Start_5204_End_5284_p_5275.png"

# Filename="mES_chr17_57_2_time_s_48_Start_266_End_313_p_272.png"#---->save
Filename="mES_chr3_74.png"
im=cv2.imread("C:/Users\PC\Desktop\model\TAD_NONTAD\Mouse\TAD/"+Filename)#---->non_aTAD
# im=cv2.imread("C:/Users\PC\Desktop\model\TAD_NONTAD\Human\TAD/"+Filename)

im=im/255
# model = load_model('C:/Users\PC\Desktop\TAD_CNN_mouse_model.h5')
# model = load_model('C:/Users\PC\Desktop\TAD_DenseNet_mouse_model.h5')
# model = load_model('C:/Users\PC\Desktop\TAD_RES_mouse_model.h5')

# model = load_model('C:/Users\PC\Desktop\TAD_CNN_human_model.h5')
# model = load_model('C:/Users\PC\Desktop\TAD_DenseNet_human_model.h5')
# model = load_model('C:/Users\PC\Desktop\model/TAD_RESNet_Mouse--1-1_E400.h5')
model = load_model('C:/Users\PC\Desktop\model/TAD_RESNet_Human--1-1_E400.h5')
model.layers[-1].activation = activations.linear

#-----------------------------------------------------------------------------------------------------WAKE
heat_map = visualize_cam(model,layer_idx=-1,filter_indices=0,seed_input=im,backprop_modifier=None)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)
plt.imshow(heat_map)
plt.show()
main_path="C:/Users\PC\Desktop\model\CAM\Test_on_Mouse\Type_TAD_Pre_Non-TAD/"
# main_path="C:/Users\PC\Desktop\model\CAM\Test_on_Human\Type_TAD_Pre_Non-TAD/"
#----------------
path=main_path+"CAM_0/"
img = cv2.cvtColor(heat_map,cv2.COLOR_BGR2RGB)
cv2.imwrite(path+"CAM_NONTAD_"+Filename,img)
#---------------------------------------------------------------------------------------
heat_map_2 = visualize_cam(model,layer_idx=-1,filter_indices=1,seed_input=im,backprop_modifier=None)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)
plt.imshow(heat_map_2)
plt.show()
#----------------

path=main_path+"CAM_1/"
img_2 = cv2.cvtColor(heat_map_2,cv2.COLOR_BGR2RGB)
cv2.imwrite(path+"CAM_TAD_"+Filename,img_2)
#--------------------------------------------------------------------------------------------------------WAKE
# im=np.array(im)
# im = im.reshape(im.shape+(1,1,))
# print(im[0])
im  = im .reshape((1, 60, 60, 3))

print("-------------------------------------------")


y_pred = model.predict(im)
print("y",y_pred)
a = np.argmax(y_pred, axis=1)
print("------->",a)
if a==1:
    print("is TAD")
else:
    print("NOT TAD")


