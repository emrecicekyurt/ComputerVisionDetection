import cv2
import mediapipe as mp
import time as tm

class handDetector():
    def __init__(self, mode=False, maxHands = 2, detect_conf= 0.5, track_conf=0.5 ):
        self.mode=mode
        self.maxHands=maxHands
        self.detect_conf = detect_conf
        self.track_conf = track_conf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detect_conf, self.track_conf)
        self.fDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLM in self.results.multi_hand_landmarks:
                if draw:
                    self.fDraw.draw_landmarks(img, handLM, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPositions(self, img, numHand=0, draw=True):
        lMlist= []


        if self.results.multi_hand_landmarks:
            theHand = self.results.multi_hand_landmarks[numHand]
            for handLM in self.results.multi_hand_landmarks:
                for id, lM in enumerate(handLM.landmark):
                    # print(id, lM)
                    height, weight, chan = img.shape
                    cx, cy = int(lM.x * weight), int(lM.y * height)
                    # print(id, cx, cy)
                    lMlist.append([id,cx,cy])
                    if draw:
                        cv2.circle(img, (cx,cy), 8, (0,0,255),cv2.FILLED)
        return lMlist

def main():
    iTime = 0  # Initial (previous) Time
    cTime = 0  # Current Time
    vCap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = vCap.read()
        img = detector.findHands(img)
        list = detector.findPositions(img)
        if len(list)!=0:
            print(list[4])
        cTime = tm.time()
        fps = 1/(cTime-iTime)
        iTime = cTime
        cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_ITALIC,2,(0,0,255), 2 )
        cv2.imshow("Video", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()