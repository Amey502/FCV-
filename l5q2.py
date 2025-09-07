import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Custom simplified SIFT descriptor
# ----------------------------
def compute_custom_sift_descriptor(img, keypoints, patch_size=16, num_bins=8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    descriptors = []

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        radius = patch_size // 2

        # Extract patch around keypoint
        if x - radius < 0 or y - radius < 0 or x + radius >= gray.shape[1] or y + radius >= gray.shape[0]:
            continue
        patch = gray[y - radius:y + radius, x - radius:x + radius]

        # Compute gradients
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        # Divide into 4x4 regions
        desc = []
        step = patch_size // 4
        for i in range(0, patch_size, step):
            for j in range(0, patch_size, step):
                mag_patch = mag[i:i+step, j:j+step].flatten()
                ang_patch = ang[i:i+step, j:j+step].flatten()

                # Histogram of orientations
                hist, _ = np.histogram(ang_patch, bins=num_bins, range=(0, 360), weights=mag_patch)
                desc.extend(hist)

        # Normalize
        desc = np.array(desc, dtype=np.float32)
        desc /= (np.linalg.norm(desc) + 1e-7)
        descriptors.append(desc)

    return np.array(descriptors, dtype=np.float32)

# ----------------------------
# Compare with OpenCV SIFT
# ----------------------------
img = cv2.imread("smile.jpeg")

# Use OpenCV SIFT
sift = cv2.SIFT_create()
keypoints, descriptors_opencv = sift.detectAndCompute(img, None)

# Use custom descriptor
descriptors_custom = compute_custom_sift_descriptor(img, keypoints[:100])  # limit for speed

print("OpenCV SIFT descriptor shape:", descriptors_opencv.shape)
print("Custom SIFT descriptor shape:", descriptors_custom.shape)

# ----------------------------
# Visualization of keypoints
# ----------------------------
img_kp = cv2.drawKeypoints(img, keypoints[:100], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
plt.title("Keypoints with SIFT")
plt.show()
