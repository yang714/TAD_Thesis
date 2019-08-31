import cv2

path="C:/Users\PC\Desktop\RAW_NOR\MES/"
image_basic=cv2.imread(path+"DIVADL_rawpre_Finally_mES_chr15.png")#----->獨立

Cut_image=image_basic[0:800,0:800]
print(Cut_image.shape)
cv2.imwrite(path+"DI_vs_DLFinally_mCO_CUT_method_ CH2_print.png",Cut_image)
file_name="mCO_RAW_ CH2_origin.png"#----------------------------------->testing

