import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image (grayscale for corner detection)
img = cv2.imread("smile.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

### 1. Harris Corner Detection ###
gray = np.float32(gray)
harris_corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# Dilate for visualization
harris_corners = cv2.dilate(harris_corners, None)

# Threshold to mark strong corners
img_harris = img.copy()
img_harris[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]  # red corners

### 2. FAST Corner Detection ###
fast = cv2.FastFeatureDetector_create()
keypoints = fast.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)

img_fast = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))  # green keypoints

### Visualization ###
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(img_harris, cv2.COLOR_BGR2RGB))
plt.title("Harris Corner Detection")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(img_fast, cv2.COLOR_BGR2RGB))
plt.title("FAST Corner Detection")

plt.show()
