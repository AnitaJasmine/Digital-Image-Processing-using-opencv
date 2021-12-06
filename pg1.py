import cv2

# Load an color image in the varible img
img = cv2.imread('D:/4.jpg')

# convert img to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# display the gray image
cv2.imshow('Original image', img)
cv2.imshow('Gray image', gray)
print(img.shape)
print(gray.shape)
gray1 = cv2.resize(gray, (512, 512))
print("Resized to:"+str(gray1.shape))
# Filename
file_name = 'D:/grayimg.jpg'

# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(file_name,gray)

# wait for user key response
cv2.waitKey(0)
# Close all the windows
cv2.destroyAllWindows()
