import cv2
import sys
import time
import numpy as np
import os

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)
recognizerFile = sys.argv[2]
faceID = sys.argv[3]
video_capture = cv2.VideoCapture(0)
recognizer = cv2.createLBPHFaceRecognizer(threshold=45.0)

if os.path.exists(recognizerFile):
    recognizer.load(recognizerFile)
else:
    recognizer.save(recognizerFile)
images = []
labels = []
print("3 Seconds Pause for Face: " + faceID)
timer = time.time() + 3
while (time.time() < timer):
    pass
timer = time.time() + 7
while time.time() < timer:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(150, 150),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        images.append(gray[y: y+h, x: x+w])
        labels.append(int(faceID))
    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
recognizer.update(images,np.array(labels))
recognizer.save(recognizerFile)
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()