import numpy as np
import cv2

EXIT_COND = list(map(ord, ['q', '\x1b']))

faceCascade = cv2.CascadeClassifier('./venv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # set Width
cap.set(4, 480)  # set Height

while not cv2.waitKey(1) in EXIT_COND:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(10, 10)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

    cv2.imshow('video', img)

cap.release()
cv2.destroyAllWindows()
