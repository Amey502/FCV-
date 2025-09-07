import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# --- 1. Load the Image ---
image = cv2.imread('usain.jpg')
# Convert from BGR (OpenCV default) to RGB for displaying correctly
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --- 2. Prepare the Data for K-Means ---
# Reshape the image into a long list of pixels (rows, columns, 3) -> (rows * columns, 3)
pixel_values = image.reshape((-1, 3))
# Convert to float32, as required by the k-means algorithm
pixel_values = np.float32(pixel_values)

# --- 3. Define the Number of Clusters (Segments) ---
# This is the 'K' in K-Means. Try changing this number!
k = 2

# --- 4. Apply K-Means Clustering ---
# Define stopping criteria and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert the center colors from float back to uint8
centers = np.uint8(centers)

# Map the labels to the center colors
segmented_image = centers[labels.flatten()]

# Reshape the segmented image back to the original image dimensions
segmented_image = segmented_image.reshape(image.shape)

# --- 5. Display the Results ---
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f'Segmented Image (K={k})')
plt.imshow(segmented_image)
plt.axis('off')

plt.show()