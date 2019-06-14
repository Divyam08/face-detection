import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
global n1
global n2
global n3
global n4
global n5
global n6
global n7
global n8
global n9

knn = KNeighborsClassifier(n_neighbors=1)
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
i1 = cv2.imread("C:\\Users\\Divyam Bhayana\\Desktop\\modiji.jpg")
i2 = cv2.imread("C:\\Users\\Divyam Bhayana\\Desktop\\m1.jpg")
i3 = cv2.imread("C:\\Users\\Divyam Bhayana\\Desktop\\m2.jpg")
i4= cv2.imread("C:\\Users\\Divyam Bhayana\\Desktop\\m3.jpg")
i5 = cv2.imread("C:\\Users\\Divyam Bhayana\\Desktop\\ms1.jpg")
i6 = cv2.imread("C:\\Users\\Divyam Bhayana\\Desktop\\ms2.jpg")
i7 = cv2.imread("C:\\Users\\Divyam Bhayana\\Desktop\\ms3.jpg")
i8 = cv2.imread("C:\\Users\\Divyam Bhayana\\Desktop\\ms4.jpg")
g1 = cv2.cvtColor(i1,cv2.COLOR_RGB2GRAY)
g2 = cv2.cvtColor(i2,cv2.COLOR_RGB2GRAY)
g3 = cv2.cvtColor(i3,cv2.COLOR_RGB2GRAY)
g4 = cv2.cvtColor(i4,cv2.COLOR_RGB2GRAY)
g5 = cv2.cvtColor(i5,cv2.COLOR_RGB2GRAY)
g6 = cv2.cvtColor(i6,cv2.COLOR_RGB2GRAY)
g7 = cv2.cvtColor(i7,cv2.COLOR_RGB2GRAY)
g8 = cv2.cvtColor(i8,cv2.COLOR_RGB2GRAY)
m1 = detector.detectMultiScale(g1,scaleFactor=1.05,minNeighbors=1)
m2 = detector.detectMultiScale(g2,scaleFactor=1.05,minNeighbors=1)
m3 = detector.detectMultiScale(g3,scaleFactor=1.05,minNeighbors=1)
m4 = detector.detectMultiScale(g4,scaleFactor=1.05,minNeighbors=1)
m5 = detector.detectMultiScale(g5,scaleFactor=1.05,minNeighbors=1)
m6 = detector.detectMultiScale(g6,scaleFactor=1.05,minNeighbors=1)
m7 = detector.detectMultiScale(g7,scaleFactor=1.05,minNeighbors=1)
m8 = detector.detectMultiScale(g8,scaleFactor=1.05,minNeighbors=1)
if len(m1) > 0:
    v1 = m1[0]
    x,y,w,h = tuple(v1)
    img1 = cv2.rectangle(i1,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow("w1",img1)
    cv2.waitKey(2000)
    n1 = img1[y:y+h, x:x+w]
    n1 = cv2.resize(n1, (100, 100)).reshape((1,-1))
if len(m2) > 0:
    v2 = m2[0]
    x,y,w,h = tuple(v2)
    img2 = cv2.rectangle(i2,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow("w2",img2)
    cv2.waitKey(2000)
    n2 = img2[y:y+h, x:x+w]
    n2 = cv2.resize(n2, (100, 100)).reshape((1,-1))
if len(m3) > 0:
    v3 = m3[0]
    x,y,w,h = tuple(v3)
    img3 = cv2.rectangle(i3,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow("w3",img3)
    cv2.waitKey(2000)
    n3 = img3[y:y+h, x:x+w]
    n3 = cv2.resize(n3, (100, 100)).reshape((1,-1))
if len(m4) > 0:
    v4 = m4[0]
    x,y,w,h = tuple(v4)
    img4 = cv2.rectangle(i4,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow("w4",img4)
    cv2.waitKey(2000)
    n4 = img4[y:y+h, x:x+w]
    n4 = cv2.resize(n4, (100, 100)).reshape((1,-1))
if len(m5) > 0:
    v5 = m5[0]
    x,y,w,h = tuple(v5)
    img5 = cv2.rectangle(i5,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow("w5",img5)
    cv2.waitKey(2000)
    n5 = img5[y:y+h, x:x+w]
    n5 = cv2.resize(n5, (100, 100)).reshape((1,-1))
if len(m6) > 0:
    v6 = m6[0]
    x,y,w,h = tuple(v6)
    img6 = cv2.rectangle(i6,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow("w6",img6)
    cv2.waitKey(2000)
    n6 = img6[y:y+h, x:x+w]
    n6 = cv2.resize(n6, (100, 100)).reshape((1,-1))
if len(m7) > 0:
    v7 = m7[0]
    x, y, w, h = tuple(v7)
    img7 = cv2.rectangle(i7, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow("w7", img7)
    cv2.waitKey(2000)
    n7 = img7[y:y + h, x:x + w]
    n7 = cv2.resize(n7, (100, 100)).reshape((1, -1))
if len(m8) > 0:
    v8 = m8[0]
    x, y, w, h = tuple(v8)
    img8 = cv2.rectangle(i8, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow("w8", img8)
    cv2.waitKey(2000)
    n8 = img8[y:y + h, x:x + w]
    n8 = cv2.resize(n8, (100, 100)).reshape((1, -1))
X = np.concatenate([n1,n2,n3,n4,n5,n6,n7,n8]).reshape((8,-1))
y = np.array(['mo','mo','mo','mo','ma','ma','ma','ma'])
knn.fit(X,y)
test = cv2.imread("C:\\Users\\Divyam Bhayana\\Desktop\\m9.jpg")
te = cv2.cvtColor(test,cv2.COLOR_RGB2GRAY)
tf = detector.detectMultiScale(te,scaleFactor=1.05,minNeighbors=1)
if len(tf) > 0:
    v3 = tf[0]
    x,y,w,h = tuple(v3)
    img9 = cv2.rectangle(test,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow("w9",img9)
    n9 = img9[y:y + h, x:x + w]
    n9 = cv2.resize(n9, (100, 100)).reshape((1,-1))
cv2.waitKey(2000)
print(knn.predict(n9))
cv2.destroyAllWindows()







