import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# ---------------------------
# a) HOG feature extraction
# ---------------------------
def compute_hog(img, visualize=False):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Parameters: 8x8 pixels per cell, 2x2 cells per block
    features, hog_img = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True,
        feature_vector=True
    )
    
    if visualize:
        hog_img = exposure.rescale_intensity(hog_img, in_range=(0, 10))
        return features, hog_img
    else:
        return features


# ---------------------------
# b) Sliding window generator
# ---------------------------
def sliding_window(img, step_size, window_size):
    for y in range(0, img.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, img.shape[1] - window_size[0] + 1, step_size):
            yield (x, y, img[y:y+window_size[1], x:x+window_size[0]])


# ---------------------------
# c) Detection using similarity score
# ---------------------------
def detect_humans(img, reference_features, window_size=(64, 128), step_size=16, threshold=0.5):
    detections = []
    
    for (x, y, window) in sliding_window(img, step_size, window_size):
        if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
            continue
        
        # Extract HoG features
        features = compute_hog(window, visualize=False)
        
        # Cosine similarity
        denom = (np.linalg.norm(features) * np.linalg.norm(reference_features))
        if denom == 0:
            sim = 0
        else:
            sim = np.dot(features, reference_features) / denom
        
        if sim > threshold:
            detections.append((x, y, sim, window_size))
    
    return detections


# ---------------------------
# d) Non-Maximum Suppression (NMS)
# ---------------------------
def non_max_suppression(detections, overlapThresh=0.3):
    if len(detections) == 0:
        return []
    
    boxes = []
    scores = []
    for (x, y, score, (w, h)) in detections:
        boxes.append([x, y, x+w, y+h])
        scores.append(score)
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    pick = []
    x1 = boxes[:,0]; y1 = boxes[:,1]
    x2 = boxes[:,2]; y2 = boxes[:,3]
    
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)
    
    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)
        
        xx1 = np.maximum(x1[last], x1[idxs[:-1]])
        yy1 = np.maximum(y1[last], y1[idxs[:-1]])
        xx2 = np.minimum(x2[last], x2[idxs[:-1]])
        yy2 = np.minimum(y2[last], y2[idxs[:-1]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        overlap = (w * h) / area[idxs[:-1]]
        
        idxs = np.delete(idxs, np.concatenate(([len(idxs) - 1],
                     np.where(overlap > overlapThresh)[0])))
    
    return boxes[pick].astype("int")


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Load positive reference human sample (resize to match window size)
    ref_img = cv2.imread("usain.jpg")
    ref_img = cv2.resize(ref_img, (64, 128))
    ref_features, ref_hog_img = compute_hog(ref_img, visualize=True)
    
    # Show the reference HOG
    plt.imshow(ref_hog_img, cmap="gray")
    plt.title("Reference Human HOG")
    plt.show()
    
    # Test image
    test_img = cv2.imread("C:/Users/ameys/OneDrive/Desktop/CS/PYTHON/Sem5/CV/street.jpg")
    
    # Run detection
    detections = detect_humans(test_img, ref_features, threshold=0.6)
    boxes = non_max_suppression(detections)
    
    # Draw results
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(test_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    plt.title("Human Detection with HOG + Sliding Window")
    plt.show()

