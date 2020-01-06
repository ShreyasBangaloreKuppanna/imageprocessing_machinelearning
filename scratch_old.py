
import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt

scale_factor = 4
threshold_canny = 0.95

################## Basic Image Processing Operations
img = cv2.imread(r'S:\Imageprocessing\Audi-task\Imageset\IMG_20180402_102947689.jpg',1)
img_copy = img.copy()
img_small = cv2.resize(img_copy, (
            img_copy.shape[1] // scale_factor, img_copy.shape[0] // scale_factor))
gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)




blur=cv2.GaussianBlur(gray,(17,17),0)
# blur = cv2.medianBlur(img,5)


###################Refining

# dft = cv2.dft(np.float32(morph),flags = cv2.DFT_COMPLEX_OUTPUT)
#
# dft = cv2.dft(np.float32(gray),flags = cv2.DFT_COMPLEX_OUTPUT)
#
# dft_shift = np.fft.fftshift(dft)
# cv2.imshow('dft',gray)
# magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

####################### Binary
edge = cv2.Canny(blur, 0,100)
# cv2.imshow('edgedft',edge)

##############################3Eorsion or dilation



kernel = np.ones((15,15), np.uint8)
close = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
kernel = np.ones((2,2), np.uint8)
opens=cv2.morphologyEx(close,cv2.MORPH_OPEN,kernel)

cv2.imshow('morph',close)

cnts = cv2.findContours(close, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

print(len(cnts))

cntrarea=[]

for i in cnts:
    cntrarea.append(cv2.contourArea(i))

print(cntrarea)
pos= cntrarea.index(max(cntrarea))
x, y, w, h = cv2.boundingRect(cnts[pos])
cv2.rectangle(img_small, (x, y), (x + w, y + h), (255, 255, 0), 2)

roi=img_small[y+60:y+h-60,x+60:x+w-60]
cv2.imshow('roi',roi)
# cv2.imshow('gray',img_small)



######################################functions on roi.


gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
blur=cv2.GaussianBlur(gray,(17,17),0)
edge = cv2.Canny(blur, 0,100)
kernel = np.ones((15,15), np.uint8)
close = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
kernel = np.ones((5,5), np.uint8)
opens=cv2.morphologyEx(close,cv2.MORPH_OPEN,kernel)

cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

print(len(cnts))
cntrarea=[]

for i in cnts:
    cntrarea.append(cv2.contourArea(i))
    a, b, wd, ht = cv2.boundingRect(i)
    cv2.rectangle(img_small, (x+60+a, y+60+b), (x + 60+a+wd, y +60+b+ht), (255, 0, 0), 2)
print(cntrarea)
# pos= cntrarea.index(max(cntrarea))
# x, y, w, h = cv2.boundingRect(cnts[pos])
# cv2.rectangle(img_small, (x, y), (x + w, y + h), (255, 255, 0), 2)

cv2.imshow('edited roi',roi)
cv2.imshow('image',img_small)



cv2.waitKey(0)
cv2.destroyAllWindows()