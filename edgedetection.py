import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load an color image in the varible img
imge = cv2.imread('D:/4.jpg')

# convert img to grayscale
gray = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)

#1.canny edge detection
edges = cv2.Canny(gray,100,200)


#2.prewitts edge detection
#noise reduction using Gaussian filter
imge_gaussian = cv2.GaussianBlur(gray,(3,3),0)
#prewitt
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
imge_prewittx = cv2.filter2D(imge_gaussian, -1, kernelx)
imge_prewitty = cv2.filter2D(imge_gaussian, -1, kernely)
prewitt = cv2.addWeighted(imge_prewittx, 0.5, imge_prewitty, 0.5, 0)

#3.Scharr filter edge Detection, ksize=-1
sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

#4.Laplacian Transformation
laplacian = cv2.Laplacian(gray,cv2.CV_64F)

#5.Robert Edge Detection

kernelx = np.array([[1, 0], [0, -1]])
kernely = np.array([[0, 1], [-1, 0]])
img_robertx = cv2.filter2D(imge_gaussian, -1, kernelx)
img_roberty = cv2.filter2D(imge_gaussian, -1, kernely)
robert = cv2.addWeighted(img_robertx, 0.5, img_roberty, 0.5, 0)


plt.subplot(2,3,1),plt.imshow(imge,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(edges,cmap = 'gray')
plt.title('CANNY'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(prewitt,cmap = 'gray')
plt.title('prewitt'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4),plt.imshow(sobel,cmap = 'gray')
plt.title('sobel'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(laplacian,cmap = 'gray')
plt.title('laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6),plt.imshow(robert,cmap = 'gray')
plt.title('robert'), plt.xticks([]), plt.yticks([])



plt.show()

cv2.waitKey(0)
# Close all the windows
cv2.destroyAllWindows()