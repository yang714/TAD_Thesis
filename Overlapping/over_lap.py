import numpy as np
from keras.models import Model,load_model
import cv2
A=90840000
B=91560000
A=A//40000
B=B//40000
Start=A
END = B
# real_TAD=[A//40000,B//40000]

C_A=2282
C_B=2316

# creat_NonTAD=[]


if A>=C_A:
    Big_Start=A
else:
    Big_Start =C_A
if B>=C_B:
    Small_END=C_B
else:
    Small_END = B

overlap=  abs(Big_Start-Small_END) /abs((END-Start))
print("A",A)
print("B",B)
print("C_A",C_A)
print("C_B",C_B)
print("(Start)",(Start))
print("(END)",(END))
print("(Big_Start))",(Big_Start))
print("(Small_END)",(Small_END))

print("Big_Start-Small_END",(Big_Start-Small_END))
print("(C_B-C_A)",(C_B-C_A))

print("(Start-END)",(Start-END))

print("OVERLAP: " ,overlap)

#--------------------------------------------------------------
Filename="hES_chr15_63_2_time_s_35_Start_2282_End_2316_p_2313.png"
im=cv2.imread("C:/Users\PC\Desktop\model\TAD_NONTAD\Human/Non-TAD/1/"+Filename)
im=im/255
# im=np.array(im)
# im = im.reshape(im.shape+(1,1,))
# print(im[0])
im  = im .reshape((1, 60, 60, 3))
# im=im/255
print("-----------------------------------------------------------------")
model = load_model('C:/Users\PC\Desktop\model/TAD_RESNet_Mouse--1-1_E400.h5')

y_pred = model.predict(im)

print("y_pred:",y_pred)

a = np.argmax(y_pred, axis=1)
print("------->",a)
if a==1:
    print("is TAD")
else:
    print("NOT TAD")