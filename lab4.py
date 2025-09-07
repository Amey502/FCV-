import cv2
import numpy as np

img = cv2.imread("smile.jpeg")
# x = cv2.resize(img,(500,500))

# for i in range(0,x.shape[0]):
#     for j in range(0,x.shape[1]):
#         # for z in range(0,x.shape[2]):

#             if(x[i][j]>=128):
#                 x[i][j]=255

#             else:
#                 x[i][j] = 0


# cv2.imshow('img',x)
# cv2.waitKey()
# cv2.destroyAllWindows()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150, apertureSize=3)

lines = cv2.HoughLinesP(edges, 1, np.pi/180, 1, minLineLength=20, maxLineGap=20)

line_img = np.copy(img)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2) 

cv2.imshow("Original Image", img)
cv2.imshow("Canny Edges", edges)
cv2.imshow("Detected Lines (Probabilistic Hough Transform)", line_img)
cv2.waitKey()
cv2.destroyAllWindows()
