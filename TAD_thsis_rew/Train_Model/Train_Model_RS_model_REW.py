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
    BatchNormalization, Activation, GlobalAveragePooling1D, GlobalAveragePooling2D, Permute, multiply, Reshape,Conv2D,Dropout,concatenate,add
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
import  Training_Testing_Data.panda_Dataformate_REW as PD
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from collections import  Counter
from sklearn.model_selection import KFold
nb_classes=2

def squeeze_excite_block(input, ratio=16):
    ''' Create a channel-wise squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
    Returns: a keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x
#K-FLOAD-----------------------------------------------
# def show_history_acc(train_history, train):
#     plt.plot(train_history.history[train])
#
#     plt.title('acc. train history')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'], loc='upper left')
#     plt.show()
#
#
# def show_history_loss(train_history, train):
#     plt.plot(train_history.history[train])
#
#     plt.title('loss train history')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'], loc='upper left')
#     plt.show()
#-------------------------------------------------

def show_history_acc(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('acc. train history')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def show_history_loss(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('loss train history')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

#--------------------------------------------------------------------------------
batch_size=48
nb_epochs=300
X,Y=PD.Train_data("Human","1")
print(Y.shape)



# skf = StratifiedKFold(n_splits=2, random_state=None, shuffle=True)#子樣本數
# skf.get_n_splits(X, Y)
#---------------------------------------------------------------

# kf= KFold(n_splits=5)
# temp=0
# cvscores = []
# score_acc=[]
# KFold(n=len(X), n_folds=2, shuffle=False,random_state=None)
# for train_index, test_index in kf.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     Y_train, Y_test = Y[train_index], Y[test_index]
#-------------------------------------------------------------------


# X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)
#
# stratified_folder = StratifiedKFold(n_splits=4, random_state=0, shuffle=False)
#----------------------------------------------------------------
#DensenNet!!!!!!!!!!!!!!!!!!
#----------------------------------------------------------------------------------
X_train=X
Y_train=Y
X_train=X_train/255
# X_test= X_test/255
# print(Y_train)
# print("print(Counter(Y_train))",Counter(Y_train))
# print("print(Counter(Y_test))",Counter(Y_test))
#------------------------------k-fload
# for train_index, test_index in  skf.split(X, Y):
#
#     #--------------
#
#     X_train, X_test = X[train_index], X[test_index]
#     Y_train, Y_test = Y[train_index], Y[test_index]
#----------------------------------------------------------
Y_train = np_utils.to_categorical( Y_train, nb_classes)
# Y_test = np_utils.to_categorical(Y_test, nb_classes)
x = Input(shape=X_train.shape[1:])
# input_layer = Input(shape=(x))
conv1 = Conv2D(64, (8, 8), strides=1, padding='same')(x)
conv1 = BatchNormalization()(conv1)
conv1 = Activation('relu')(conv1)
conv1 = Conv2D(64, (5, 5), strides=1, padding='same')(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = Activation('relu')(conv1)
conv1 = Conv2D(64, (3, 3), strides=1, padding='same')(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = Activation('relu')(conv1)
conv1 = squeeze_excite_block(conv1)
is_expand_channels = not (x.shape[-1] == 64)
if is_expand_channels:
    shortcut_y = keras.layers.Conv2D(64, 1, strides=1, padding='same')(x)
    shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)
else:
    shortcut_y = keras.layers.normalization.BatchNormalization()(x)
print('shape of x[-1]', x.shape[-1])
print('Merging skip connection')
# y = merge([shortcut_y, conv_z], mode='sum')
y = add([shortcut_y, conv1])
y = Activation('relu')(y)
# conv2 = squeeze_excite_block(y)
conv2 = Conv2D(128, (8, 8), strides=1, padding='same')(y)
conv2 = BatchNormalization()(conv2)
conv2 = Activation('relu')(conv2)
conv2 = Conv2D(128, (5, 5), strides=1, padding='same')(conv2)
conv2 = BatchNormalization()(conv2)
conv2 = Activation('relu')(conv2)
conv2 = Conv2D(128, (3, 3), strides=1, padding='same')(conv2)
conv2 = BatchNormalization()(conv2)
conv2 = Activation('relu')(conv2)
conv2 = squeeze_excite_block(conv2)
is_expand_channels = not (y.shape[-1] == 128)
if is_expand_channels:
    shortcut_y2 = keras.layers.Conv2D(128, 1, strides=1, padding='same')(y)
    shortcut_y2 = keras.layers.normalization.BatchNormalization()(shortcut_y2)
else:
    shortcut_y2 = keras.layers.normalization.BatchNormalization()(y)
print('shape of y[-1]', y.shape[-1])
print('Merging skip connection')
# y = merge([shortcut_y, conv_z], mode='sum')
y2 = add([shortcut_y2, conv2])
y2 = Activation('relu')(y2)
conv3 = Conv2D(128, (8, 8), strides=1, padding='same')(y2)
conv3 = BatchNormalization()(conv3)
conv3 = Activation('relu')(conv3)
conv3 = Conv2D(128, (5, 5), strides=1, padding='same')(conv3)
conv3 = BatchNormalization()(conv3)
conv3 = Activation('relu')(conv3)
conv3 = Conv2D(128, (3, 3), strides=1, padding='same')(conv3)
conv3 = BatchNormalization()(conv3)
conv3 = Activation('relu')(conv3)
conv3 = squeeze_excite_block(conv3)
is_expand_channels = not (y2.shape[-1] == 128)
if is_expand_channels:
    shortcut_y3 = keras.layers.Conv2D(128, 1, strides=1, padding='same')(y2)
    shortcut_y3 = keras.layers.normalization.BatchNormalization()(shortcut_y3)
else:
    shortcut_y3 = keras.layers.normalization.BatchNormalization()(y2)
print('shape of y2[-1]', y2.shape[-1])
print('Merging skip connection')
# y = merge([shortcut_y, conv_z], mode='sum')
y3 = add([shortcut_y3, conv3])
y3 = Activation('relu')(y3)
final = keras.layers.GlobalAveragePooling2D()(y3)
# add_0 = Dense(128, activation='relu')(conv3)
# add_0 = Dropout(0.2)(add_0)
# add_1 = Dense(128, activation='relu')(add_0)
# add_1 = Dropout(0.2)(add_1)
output_layer = Dense(nb_classes, activation='softmax')(final)
# add_0 = Dense(128, activation='relu')(conv3)
# add_0 = Dropout(0.2)(add_0)
# add_1 = Dense(128, activation='relu')(add_0)
# add_1 = Dropout(0.2)(add_1)

model = Model(inputs=x, outputs=output_layer)

try:
    # model.load_weights('F:/TAD_DATA/TAD_CNN.h5')
    # model.load_weights('F:/TAD_DATA/TAD_CNN_w.h5')
    print("have model------------------------------------------------------------------------------------")
except:
    print("no model-----------------------------------------------------------------------------------------")
# -----------------
model.summary()




optimizer = keras.optimizers.adam()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
# early_stopping = EarlyStopping(monitor='val_loss',patience=500,verbose=1,mode='auto',min_delta=0.0000001)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=200, min_lr=0.0001)

# hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
#                  verbose=0, callbacks=[reduce_lr], shuffle=True, validation_data=(X_test, Y_test))

hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
                 verbose=1, callbacks=[reduce_lr], shuffle=True)
# Print the testing results which has the lowest training loss.
# log = pd.DataFrame(hist.history)
# print("log.loc[log['loss'].idxmin]['loss']",log.loc[log['loss'].idxmin]['loss'], "log.loc[log['loss'].idxmin]['val_acc']",log.loc[log['loss'].idxmin]['val_acc'])
# show_history_acc(hist, 'acc', 'val_acc')
# show_history_loss(hist, 'loss', 'val_loss')
#*K-fload-----------------------------------------------------------------------------------------
# hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
#                  verbose=0, callbacks=[reduce_lr], shuffle=True)
#
# # Print the testing results which has the lowest training loss.
# log = pd.DataFrame(hist.history)
# print(log.loc[log['loss'].idxmin]['loss'])
# show_history_acc(hist, 'acc')
# show_history_loss(hist, 'loss')
# ------------------------------------------------------------------------------------------------------
# model.save_weights('F:/TAD_DATA/TAD_CNN_human_w.h5')
# model.save('F:/TAD_DATA/TAD_CNN_human_model.h5')
model.save_weights('C:/Users\PC\Desktop/New_Thsis\Model_save\THEM/TAD_RESNet_Human--1-1W0828_E300.h5')
model.save('C:/Users\PC\Desktop/New_Thsis\Model_save\THEM/TAD_RESNet_Human--1-108_28_E300.h5')

# scores = model.evaluate(X_test, Y_test, verbose=0)
# cvscores.append(scores[1] * 100)
# if  scores[1]>temp:
#     model.save_weights('F:/TAD_DATA/TAD_CNNH_w.h5')
#     model.save('F:/TAD_DATA/TAD_CNNH_model.h5')
#     temp=scores[1]
# else:
#     print("scores:--------> ",scores[1],"  temp:------>",temp)
#--------------------------------------------------------
# y_pred = model.predict(X_test, batch_size=12)

# practical and predicted labels
# print(np.argmax(Y_test, axis=1))
# print(np.argmax(y_pred, axis=1))

# cnf_matrix = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(y_pred, axis=1), labels=[0, 1])
# cm = pd.crosstab(np.argmax(y_pred, axis=1), np.argmax(Y_test, axis=1), rownames=['prediction'], colnames=['label'])

# tp, fp, fn, tn = cnf_matrix.ravel()
# tpr = tp / (tp + fn)
# tnr = tn / (fp + tn)
# recall = tp / (tp + fn)
# precision = tp / (tp + fp)
# f1 = 2 / ((1 / precision) + (1 / recall))
# roc_fpr, roc_tpr, threshold = metrics.roc_curve(np.argmax(Y_test, axis=1), np.argmax(y_pred, axis=1))
# roc_auc = metrics.auc(roc_fpr, roc_tpr)
# accuracy = (tp + tn) / (tp + tn + fp + fn)
# print('----------------------------------------------------------------')
# print('confusion matrix')
# print("----------------------------------------------------------------")
# print(cm)
# print("----------------------------------------------------------------")
# # print(cnf_matrix)
# print('tp:', tp)
# print('tn:', tn)
# print('fp:', fp)
# print('fn:', fn)
# print('tpr:', tpr)
# print('tnr:', tnr)
# print('roc:', roc_auc)
# print('precision', precision)
# print('recall', recall)
# print('f1 score', f1)
# print('accuracy:', accuracy)
# # plot the roc curve
# plt.title('Receiver Operating Characteristic')
# plt.plot(roc_fpr, roc_tpr, 'b', label='AUC = %0.2f' % roc_auc)
# plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1], 'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()

# for j,i in  enumerate(cvscores):
#     print(j,"----->",i)
#
# print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
