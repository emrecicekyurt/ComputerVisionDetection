import cv2
import mediapipe as mp
import time as tm

vCap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
fDraw = mp.solutions.drawing_utils

iTime = 0 # Initial (previous) Time
cTime = 0 # Current Time
while True:
    success, img = vCap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLM in results.multi_hand_landmarks:
            for id, lM in enumerate(handLM.landmark):
                #print(id, lM)
                height, weight, chan = img.shape
                cx, cy = int(lM.x*weight), int(lM.y*height)
                print(id, cx, cy)
            fDraw.draw_landmarks(img, handLM, mpHands.HAND_CONNECTIONS)
    cTime = tm.time()
    fps = 1/(cTime-iTime)
    iTime = cTime
    cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_ITALIC,2,(0,0,255), 2 )
    cv2.imshow("Video", img)
    cv2.waitKey(1)