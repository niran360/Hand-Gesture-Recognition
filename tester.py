import cv2
import mediapipe as mp

mpHands =mp.solutions.hands
hands =mpHands.Hands()
drawTools = mp.solutions.drawing_utils

capture = cv2.VideoCapture(0)
while True:
    success, image = capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if (results.multi_hand_landmarks):
        for handlms in results.multi_hand_landmarks:
            drawTools.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Video feed", img)
    key = cv2.waitKey(1)
    if (key == 27):
        break


img = cv2.imread("hands.jpg")

cv2.imshow("hands",img)
cv2.waitKey()