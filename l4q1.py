import cv2
import numpy as np

img = cv2.imread("smile.jpeg",0)

print(img.shape)

img_new = [[0 for i in range(266)] for j in range(200)]

for i in range(200):
    for j in range(266):

        if img[i][j]>=128:
            img_new[i][j] = 255

        else:
            img_new[i][j] = 0


img_new = np.array(img_new,dtype=np.uint8)
cv2.imshow("img_n",img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()