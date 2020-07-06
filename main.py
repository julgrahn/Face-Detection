import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("facephoto.jpg")
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(grayImage,        #built in method using haarcascade
scaleFactor = 1.05,                                     #creates a "scale pyramid", reduce size by 5% each step
minNeighbors = 7)                                       #eliminate false positives, using the value 7 works best in my case here after trial and error

for x, y, w, h in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3) #draws the rectangle from img, x,y coords for top-left etc, then 255 for green, last parameter for thickness

print(type(faces))
print(faces)

resized = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2))) #lowers res by 50%

cv2.imshow("Face", img)
#cv2.imshow("Gray", resized) #if user wants to lower the resolution by 50%
cv2.waitKey(0)
cv2.destroyAllWindows()