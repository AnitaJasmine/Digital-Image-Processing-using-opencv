
#a.	Histogram Equalization
#b.	Image Normalization
#c.	Convert to Binary Image
#d.	Inverted Image
#e.	Image Multiplication

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load an color image in the varible img
img = cv2.imread('D:/Retina.jpg')

# convert img to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# display the gray image
cv2.imshow('Original image', img)
#cv2.imshow('Gray image', gray)

# creating a Histograms Equalization
# of a image using cv2.equalizeHist()
equ = cv2.equalizeHist(gray)
#cv2.imshow('Histogram equalized  image',equ)

#normalize the image intensity
norm_img = np.zeros((800,800))
norm_img = cv2.normalize(gray,norm_img, 0, 40, cv2.NORM_MINMAX)

# display the normalized image
#cv2.imshow('normalized image',norm_img)

# display the threshold image
#cv2.imshow('Thresholded  image',thresh)

#inverted image
inverted = (255-equ)
#cv2.imshow('Inverted  image',inverted)
#threshold the image
ret,thresh=cv2.threshold(inverted,50,250,cv2.THRESH_BINARY)
# multiply images
#cv2.multiply(thresh,gray)
final_img=thresh*gray
#cv2.imshow('Final  image',final_img)
# wait for user key response
titles = ['Gray image','normalized image','Histogram equalized  image','Inverted  image','thresholded  image','Image Multiplicatio']
images = [gray, norm_img, equ, inverted, thresh,final_img]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()


cv2.waitKey(0)
# Close all the windows
cv2.destroyAllWindows()
