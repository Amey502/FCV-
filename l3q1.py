import cv2
import numpy as np

img = cv2.imread("street.jpg",0)

print(img.shape)

# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# blur = cv2.GaussianBlur(img,(7,7),0)

# details = img-blur

# sharpimg = details+img

# cv2.imshow("pig",sharpimg)
# cv2.waitKey()