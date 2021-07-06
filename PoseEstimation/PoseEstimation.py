import cv2
import mediapipe as mp
import time as tm

mpDraw = mp.solutions.drawing_utils
mpPose=mp.solutions.pose
pose = mpPose.Pose()


def rescale_frame(frame_input, percent=75):
    width = int(frame_input.shape[1] * percent / 100)
    height = int(frame_input.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame_input, dim, interpolation=cv2.INTER_AREA)

cap = cv2.VideoCapture("DanceVideo.mp4")
iTime = 0 # Initial (previous) Time
cTime = 0 # Current Time
while True:
    success, img = cap.read()
    imgRes= rescale_frame(img, 60)
    imgRGB = cv2.cvtColor(imgRes, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        for id, lM in enumerate(results.pose_landmarks.landmark):
            # print(id, lM)
            height, weight, chan = img.shape
            cx, cy = int(lM.x * weight), int(lM.y * height)
            print(id, cx, cy)
        mpDraw.draw_landmarks(imgRes, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    cTime = tm.time()
    fps = 1/(cTime-iTime)
    iTime = cTime
    cv2.putText(imgRes, str(int(fps)),(70,50), cv2.FONT_ITALIC,2,(0,0,255), 2 )
    cv2.imshow("Video", imgRes)
    cv2.waitKey(1)
