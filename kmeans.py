import cv2
import numpy as np

img = cv2.imread("usain.jpg")
img_new = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

k = 3
np.random.seed(42)

pixels = img_new.reshape((-1,3)).astype(np.float32)

rind = np.random.choice(len(pixels),replace=False,size = k)
centroids = pixels[rind]

# print(centroids)

for i in range(10):

    distances = np.linalg.norm(pixels[:,np.newaxis]-centroids, axis=2)
    labels = np.argmin(distances,axis = 1)

    new_c = np.array([pixels[labels==j].mean(axis=0) for j in range(k)])
    centroids = new_c

sp = centroids[labels].astype(np.uint8)
so = sp.reshape(img.shape)

cv2.imshow("seg",so)
cv2.waitKey(0)