import cv2
import numpy as np

img = cv2.imread("smile.jpeg", 0)

hist_bin = [0]*256

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        hist_bin[img[i][j]] += 1

hist_bin = np.array(hist_bin) / np.sum(hist_bin)

cums = [0]*256
s = 0
for i in range(256):
    s += hist_bin[i]
    cums[i] = s

cums = np.round(255 * np.array(cums)).astype(np.uint8)

# Vectorized mapping
dummy_img = cums[img]

cv2.imshow("Original", img)
cv2.imshow("Equalized", dummy_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
