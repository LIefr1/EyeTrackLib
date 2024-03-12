import cv2 
import numpy as np 
import matplotlib.pyplot as plt

# image = cv2.imread('face.jpg')

# def cv_show(name,img,wait= 0):
# 	cv2.imshow(name, img)
# 	cv2.waitKey(wait)
# 	cv2.destroyAllWindows()

# # cv_show('eye', img=image)

# egdes_image = cv2.imwrite('edges_face.jpg', cv2.Canny(image, 200,300))

# cv_show('edges', cv2.imread('edges_face.jpg'))

img = cv2.imread('faces_portraits.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fig = plt.figure("Cage uncaged")
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.axis("off")
# plt.show()

face_cascade = cv2.CascadeClassifier(r'C:\Users\scher\AppData\Local\Programs\Python\Python39\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'C:\Users\scher\AppData\Local\Programs\Python\Python39\Lib\site-packages\cv2\data\haarcascade_eye.xml')
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.circle(roi_color, (ex+25,ey+25), 40, (0,255,0),2)

fig = plt.figure("Cage caged")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()