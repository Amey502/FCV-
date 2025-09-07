import cv2
import numpy as np

img = cv2.imread("usain.jpg",0)
gx = np.zeros_like(img,dtype=np.float32)
gy = np.zeros_like(img,dtype=np.float32)
box_r = np.zeros_like(img,dtype=np.float32)

img_pad = np.pad(img,1,mode = 'constant')

# print(img_pad)

Gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
Gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

box = np.array([[1,1,1],[1,1,1],[1,1,1]])

for i in range(img.shape[0]):
    for j in range(img.shape[1]):

        gx[i][j] = np.sum(Gx*img_pad[i:i+3,j:j+3])
        gy[i][j] = np.sum(Gy*img_pad[i:i+3,j:j+3])
        box_r[i,j] = (1/9)*np.sum(box*img_pad[i:i+3,j:j+3])

box_r = [[255 if i >=128 else 0 for i in row] for row in box_r]
box_r = np.array(box_r, dtype=np.float32)
grad = np.sqrt(gx**2+gy**2)
# box = np.uint8(np.clip(grad, 0, 255))
grad = np.uint8(np.clip(grad, 0, 255))

ang = np.arctan2(gy,gx)

print(ang)

cv2.imshow("gx",gx)
cv2.imshow("gy",gy)
cv2.imshow("gg",grad)
cv2.imshow("box",box_r)

cv2.waitKey(0)
cv2.destroyAllWindows()

