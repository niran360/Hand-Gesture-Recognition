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