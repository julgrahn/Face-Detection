import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("facephoto.jpg")
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(grayImage,
scaleFactor = 1.05,
minNeighbors = 5)

print(type(faces))
print(faces)

cv2.imshow("Gray", grayImage)
cv2.waitKey(0)
cv2.destroyAllWindows()