import cv2
import numpy as np

img = cv2.imread("images.png")
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ix = cv2.Sobel(img_gray,cv2.CV_64F,1,0,3)
iy = cv2.Sobel(img_gray,cv2.CV_64F,0,1,3)

# img_grap = np.pad(img_gray,1,mode='constant')

ixx = ix*ix
iyy = iy*iy
ixy = ix*iy

Sx = cv2.GaussianBlur(ixx,(3,3),0)
Sy = cv2.GaussianBlur(iyy,(3,3),0)
Sxy = cv2.GaussianBlur(ixy,(3,3),0)

detM = Sx*Sy - Sxy**2
TrM = Sx+Sy

R = detM - 0.04 * TrM**2

thres = 0.01 * np.max(R)

# for i in range(1,R.shape[0]-1):
#     for j in range(1,R.shape[1]-1):

#         if(R[i][j]>=R[i][j+1]) and (R[i][j]>=R[i][j-1]) and (R[i][j]>=R[i-1][j]) and (R[i][j]>=R[i+1][j]) and (R[i][j]>=R[i+1][j-1]) and (R[i][j]>=R[i+1][j+1]) and (R[i][j]>=R[i-1][j+1]) and (R[i][j]>=R[i-1][j-1]):
#             continue

#         else:
#             R[i][j] = 0

img[R > thres] = [0, 0, 255]

# for i in range(0,R.shape[0]):
#     for j in range(0,R.shape[1]):

#         if(R[i][j]<=thres):
#             R[i][j] = 0


cv2.imshow("thres",img)
cv2.waitKey(0)
cv2.destroyAllWindows()