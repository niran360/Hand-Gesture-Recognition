import cv2
import mediapipe as mp


mpHands = mp.solutions.hands
hands = mpHands.Hands()
drawTools = mp.solutions.drawing_utils

class HandDetector():
    def lmlist(self, img, draw = True ):
        lmlist = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if (results.multi_hand_landmarks):
            for handlms in results.multi_hand_landmarks:
                for id, lm in enumerate(handlms.landmark):
                    h,w,c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)

                    lmlist.append([id,cx,cy])
                
            if draw:
                drawTools.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)

        return lmlist, img


    def fingersUp(self, img, lmlist, draw = True):
        fingers=[]
        fingertips=[8,12,16,20]
        count=0
        if (lmlist[4][1] < lmlist[3][1]):
            fingers.append(0)
        else:
             fingers.append(1)
             count +=1

        for tip in fingertips:
            if (lmlist[tip][2] < lmlist[tip - 2] [2]):
                fingers.append(1)
                count += 1
            else:
                fingers.append(0)
        if draw:
            cv2.putText(img, str(count), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 3)

        return fingers, img
