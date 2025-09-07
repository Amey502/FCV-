import cv2


im = cv2.imread("C:/Users/ameys/OneDrive/Desktop/CS/PYTHON/Sem5/CV/20120725171132-1_0.jpg")
# cv2.imshow('p',im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# r,g,b = im[350,543]
# print(f"red : {r}\ngreen : {g}\nblue : {b}")

# cv2.rectangle(im,(0,0),(100,100),color=(255, 0, 0),thickness=2)
# half = cv2.resize(im, (50, 50))

w,h = im.shape[:2]
# print(w)

center = (w//2,h//2)
angle = 45
scale = 1.0

x = cv2.getRotationMatrix2D(center,angle,scale)

arb_angle = cv2.warpAffine(im,x,(w,h))
cv2.imshow('p',arb_angle)
cv2.waitKey(0)
cv2.destroyAllWindows()