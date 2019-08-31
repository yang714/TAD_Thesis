from keras.models import load_model
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os
import random
import   test  as  teat
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, LSTM, MaxPooling1D, Conv1D, TimeDistributed, Flatten, AveragePooling1D, \
    BatchNormalization, Activation, GlobalAveragePooling1D, GlobalAveragePooling2D, Permute, multiply, Reshape,Conv2D,Dropout
from keras.models import Model
import matplotlib.pyplot as  plt
from keras.models import Model
from keras.utils import np_utils
import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau,EarlyStopping,TensorBoard
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from numba import cuda
import keras
from sklearn.utils import class_weight
import  Evaluate.EVAL_panda_Dataformate_REW as PD
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from collections import Counter
X_test,Y_test=PD.Train_data()
print("X_test_shape",X_test.shape)

# model=load_model('C:/Users\PC\Desktop/New_Thsis\Model_save\TAD_RESNet_Human--1-1_E400.h5')
model=load_model('C:/Users\PC\Desktop/New_Thsis\Model_save\THEM/TAD_RESNet_Human--1-108_28_E300.h5')
X_test=X_test/255
Y_test = np_utils.to_categorical(Y_test,2)








a=[]



y_pred =model.predict(X_test, batch_size=12)

# practical and predicted labels
print(np.argmax(Y_test, axis=1))
print(np.argmax(y_pred, axis=1))
a=np.argmax(y_pred, axis=1)
b=np.argmax(Y_test, axis=1)
print("y_pred",a[0:3])
print("y_pred",y_pred[0:3])
print("YTEST_",Y_test[0:10])
print("YTESTargmax",b[0:10])

print("y_pred",Counter(a))
print("y_pred",a.shape)
print("Y_test",Counter(b))
print("Y_test",b.shape)
np.argmax(Y_test)
np.argmax(Y_test, axis=1)

cnf_matrix = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(y_pred, axis=1), labels=[0, 1])
cm = pd.crosstab( np.argmax(Y_test, axis=1),np.argmax(y_pred, axis=1), rownames=['prediction'], colnames=['label'])

tp, fp, fn, tn = cnf_matrix.ravel()
tpr = tp / (tp + fn)
tnr = tn / (fp + tn)
recall = tp / (tp + fn)
precision = tp / (tp + fp)
f1 = 2 / ((1 / precision) + (1 / recall))
print("Y_test!!!!!!!!!!!!!!!",np.argmax(Y_test, axis=1))
print("Y_p!!!!!!!!!!!!!", np.argmax(y_pred, axis=1))
roc_fpr, roc_tpr, threshold = metrics.roc_curve(np.argmax(Y_test, axis=1), np.argmax(y_pred, axis=1))
# roc_fpr, roc_tpr, threshold = metrics.roc_curve(np.argmax(Y_test, axis=1), np.argmax(y_pred, axis=1),pos_label=1)
# roc_fpr, roc_tpr, threshold = metrics.roc_curve(np.argmax(Y_test, axis=1), np.argmax(y_pred, axis=1),pos_label=0)
roc_auc = metrics.auc(roc_fpr, roc_tpr)
accuracy = (tp + tn) / (tp + tn + fp + fn)
print('----------------------------------------------------------------')
print('confusion matrix')
print("----------------------------------------------------------------")
print(cm)
print("----------------------------------------------------------------")
# print(cnf_matrix)
print('tp:', tp)
print('tn:', tn)
print('fp:', fp)
print('fn:', fn)
print('tpr:', tpr)
print('tnr:', tnr)
print('roc:', roc_auc)
print('precision', precision)
print('recall', recall)
print('f1 score', f1)
print('accuracy:', accuracy)
# plot the roc curve
plt.title('Receiver Operating Characteristic')
print("roc_fpr",roc_fpr)
print("roc_tpr",roc_tpr)
print("roc_auc",roc_auc)
plt.plot(roc_fpr, roc_tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

