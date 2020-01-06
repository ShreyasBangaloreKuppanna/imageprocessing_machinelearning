import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt

scale_factor = 4
threshold_canny = 0.95

################## Basic Image Processing Operations
img = cv2.imread(r'S:\Imageprocessing\Audi-task\Imageset\IMG_20180402_103109183.jpg',1)
img_copy = img.copy()
img_small = cv2.resize(img_copy, (
            img_copy.shape[1] // scale_factor, img_copy.shape[0] // scale_factor))
cv2.imshow('actual image',img_small)
gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale',gray)


blur=cv2.GaussianBlur(gray,(17,17),0)

cv2.imshow('Gaussian blur',blur)

####################### Edge detection
edge = cv2.Canny(blur, 0,100)
cv2.imshow('Cannyedge',edge)

##############################Remove noises
kernel = np.ones((15,15), np.uint8)
close = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
kernel = np.ones((2,2), np.uint8)
opens=cv2.morphologyEx(close,cv2.MORPH_OPEN,kernel)

cv2.imshow('morphological image',opens)
#############################Extract the roi


cnts = cv2.findContours(close, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# print(len(cnts))

cntrarea=[]

for i in cnts:
    cntrarea.append(cv2.contourArea(i))

# print(cntrarea)
pos= cntrarea.index(max(cntrarea))
x, y, w, h = cv2.boundingRect(cnts[pos])
cv2.rectangle(img_small, (x, y), (x + w, y + h), (255, 255, 0), 2)

cv2.imshow('final',img_small)
roi=img_small[y+60:y+h-60,x+60:x+w-60]
cv2.imshow('roi',roi)
#cv2.imshow('gray',img_small)



######################################functions on roi to extract objects.


gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
blur=cv2.GaussianBlur(gray,(17,17),0)
edge = cv2.Canny(blur, 0,100)
kernel = np.ones((15,15), np.uint8)
close = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
kernel = np.ones((5,5), np.uint8)
opens=cv2.morphologyEx(close,cv2.MORPH_OPEN,kernel)

cv2.imshow('Morphed ROI',close)

cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# print(len(cnts))
cntrarea=[]

for i in cnts:
    cntrarea.append(cv2.contourArea(i))
    a, b, wd, ht = cv2.boundingRect(i)
    cv2.rectangle(roi, (a, b), (a + wd, b + ht), (255, 255, 0), 2)
    cv2.rectangle(img_small, (x+60+a, y+60+b), (x + 60+a+wd, y +60+b+ht), (255, 0, 0), 2)
# print(cntrarea)


cv2.imshow('edited roi',roi)
cv2.imshow('image',img_small)
print('Number of objects identified', len(cnts))


cv2.waitKey(0)
cv2.destroyAllWindows()