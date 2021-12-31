import cv2
import mediapipe as mp
from HandTrackingModule import HandDetector

detector = HandDetector()

capture = cv2.VideoCapture(0)
while True:
    success, img = capture.read()

    lmlist, img = detector.lmlist(img)

    if lmlist:
        fingers, img = detector.fingersUp(img, lmlist)
        
        # print (fingers)
        
    cv2.imshow("Video feed", img)
    key = cv2.waitKey(1)
    if (key == 27):
        break


capture.release()
cv2.destroyAllWindows()