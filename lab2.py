import cv2
import numpy as np

img = cv2.imread("C:/Users/ameys/OneDrive/Desktop/CS/PYTHON/Sem5/CV/pic.jpeg",0)
img = cv2.resize(img,(500,500),interpolation=cv2.INTER_CUBIC)
# img = np.array(img,np.float32)

#log transformation

# c = 255 / np.log(1+np.max(img))
# trans = c * np.log(1+img)
# trans = np.array(trans,dtype=np.uint8)

# cv2.imshow('transformed pic', trans)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#gamma transformation

# x = [0.1,0.5,1.2,2.2]

# for g in x:

#     gam = np.array(255*(img/255)**g, dtype = np.uint8)
#     # cv2.imwrite("gam"+str(gam)+".jpg",gam)
#     cv2.imshow('pic',gam)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def pix(p,r1,s1,r2,s2):

#     if(0<=p and p<=r1):
#         return (s1/r1)*p
    
#     elif(p>r1 and p<=r2):
#         return ((s2-s1)/(r2-r1))*(p-r1) + s1
    
#     else:
#         return ((255-s2)/(255-r2))*(p-r2)+s2
    
# z = np.vectorize(pix)

# i = z(img,70,0,140,255)


# cv2.imshow('pic',i)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#histogram equalization

# print(img.shape)

l = [0 for i in range(0,256)]

for i in range(0,500):
    for j in range(0,500):
        
        l[img[i][j]] = l[img[i][j]]+1

p = [ z/(500*500) for z in l]


cdf = np.cumsum(p)

sk = np.round(cdf*255).astype(np.uint8)
                              
# equalized = np.zeros_like(img)
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         equalized[i][j] = sk[img[i][j]]


# cv2.imshow('pic',equalized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(l)

#histogram specification

ref = cv2.imread("C:/Users/ameys/OneDrive/Desktop/CS/PYTHON/Sem5/CV/smile.jpeg",0)

ref_list = [0 for i in range(0,256)]

for i in range(0,ref.shape[0]):
    for j in range(0,ref.shape[1]):
        ref_list[ref[i][j]]+=1

ref_pdf = [x/(ref.shape[0]*ref.shape[1]) for x in ref_list]

ref_cdf = np.cumsum(ref_pdf)
fxn = np.round(ref_cdf*255).astype(np.uint8)

mapping = [0 for i in range(0,256)]

for i in range(0,256):
    x = np.abs(fxn-sk[i])
    mapping[i] = np.argmin(x)

new_img = np.zeros_like(img)

for i in range(0,img.shape[0]):
    for j in range(0,img.shape[1]):
        new_img[i][j] = mapping[img[i][j]]

cv2.imshow('ref',ref)
cv2.imshow('original', img)
cv2.imshow('pic',new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
