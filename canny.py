import cv2 as cv
import numpy as np

x,y = np.mgrid[-1:2,-1:2]

gb = (1/2*np.pi*1)*(np.exp(-1*(x**2+y**2)/(2*1)))
gb = gb/np.sum(gb)
# print(gb)

img = cv.imread("usain.jpg",0)
gx = np.zeros_like(img,dtype=np.float32)
gy = np.zeros_like(img,dtype=np.float32)

img_pad = np.pad(img,1,mode = 'constant')
img_res = np.zeros_like(img,dtype=np.float32)

Gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
Gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

for i in range(img.shape[0]):
    for j in range(img.shape[1]):

        img_res[i][j] = np.sum(gb*img_pad[i:i+3,j:j+3])

img_res = np.pad(img_res,1,mode="constant")

for i in range(img.shape[0]):
    for j in range(img.shape[1]):

        gx[i][j] = np.sum(Gx*img_res[i:i+3,j:j+3])
        gy[i][j] = np.sum(Gy*img_res[i:i+3,j:j+3])
        

grad = np.sqrt(gx**2+gy**2)
theta = np.arctan2(gy,gx)

##NMS

theta = theta*180.0 / np.pi
theta[theta<0] +=180

nms_result = np.zeros_like(grad, dtype=np.float32)

for i in range(1,img.shape[0]-1):
    for j in range(1,img.shape[1]-1):

        p,q = 0,0

        angle = theta[i][j]

        if(0<=angle<22.5) or (157.5<=angle<=180):
            p = grad[i,j+1]
            q = grad[i,j-1]

        elif(22.5<=angle<67.5):
            p = grad[i-1,j+1]
            q = grad[i+1,j-1]

        elif(67.5<=angle<112.5):
            p = grad[i-1,j]
            q = grad[i+1,j]

        elif(112.5<=angle<157.5):
            p = grad[i-1,j-1]
            q = grad[i+1,j+1]

        if(grad[i,j]>=p) and (grad[i,j]>=q):
            nms_result[i, j] = grad[i, j]
        else:
            nms_result[i, j] = 0


high_threshold_ratio = 0.15
low_threshold_ratio = 0.05

# You can experiment with these ratios
high_threshold = np.max(nms_result) * high_threshold_ratio
low_threshold = high_threshold * low_threshold_ratio

# Create a new image for the result
result_img = np.zeros_like(nms_result, dtype=np.uint8)

# Define values for strong and weak pixels
STRONG_PIXEL = 255
WEAK_PIXEL = 75

# Find indices of strong and weak pixels
strong_i, strong_j = np.where(nms_result >= high_threshold)
weak_i, weak_j = np.where((nms_result >= low_threshold) & (nms_result < high_threshold))

# Set the pixels in the result image
result_img[strong_i, strong_j] = STRONG_PIXEL
result_img[weak_i, weak_j] = WEAK_PIXEL

# Iterate through the image (ignoring the 1-pixel border)
for i in range(1, result_img.shape[0] - 1):
    for j in range(1, result_img.shape[1] - 1):
        # If the pixel is strong, check its 8 neighbors
        if (result_img[i, j] == STRONG_PIXEL):
            # Check the 3x3 neighborhood
            if (result_img[i+1, j-1] == WEAK_PIXEL): result_img[i+1, j-1] = STRONG_PIXEL
            if (result_img[i+1, j] == WEAK_PIXEL):   result_img[i+1, j] = STRONG_PIXEL
            if (result_img[i+1, j+1] == WEAK_PIXEL): result_img[i+1, j+1] = STRONG_PIXEL
            if (result_img[i, j-1] == WEAK_PIXEL):   result_img[i, j-1] = STRONG_PIXEL
            if (result_img[i, j+1] == WEAK_PIXEL):   result_img[i, j+1] = STRONG_PIXEL
            if (result_img[i-1, j-1] == WEAK_PIXEL): result_img[i-1, j-1] = STRONG_PIXEL
            if (result_img[i-1, j] == WEAK_PIXEL):   result_img[i-1, j] = STRONG_PIXEL
            if (result_img[i-1, j+1] == WEAK_PIXEL): result_img[i-1, j+1] = STRONG_PIXEL

# Set any remaining weak pixels to 0
result_img[result_img == WEAK_PIXEL] = 0

# Your final Canny edge image!
cv.imshow("Final Canny Edges", result_img)
cv.waitKey(0)
cv.destroyAllWindows()