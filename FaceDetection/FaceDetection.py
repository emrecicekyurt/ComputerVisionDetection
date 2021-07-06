import cv2
import mediapipe as mp
import time as tm

def rescale_frame(frame_input, percent=75):
    width = int(frame_input.shape[1] * percent / 100)
    height = int(frame_input.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame_input, dim, interpolation=cv2.INTER_AREA)

cap = cv2.VideoCapture("elonmuskvideo.mp4")
iTime = 0 # Initial (previous) Time
cTime = 0 # Current Time

mpFace = mp.solutions.face_detection
mpDraw=mp.solutions.drawing_utils
faceDetect = mpFace.FaceDetection(0.75)

while True:
    success, img = cap.read()
    img = rescale_frame(img, 60)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetect.process(imgRGB)
    if results.detections:
        for id, detection in enumerate(results.detections):
            # print(id, detection)
            boundingBoxC = detection.location_data.relative_bounding_box
            ih, iw, ic =img.shape
            boundingBox = int(boundingBoxC.xmin*iw), int(boundingBoxC.ymin*ih), \
                          int(boundingBoxC.width * iw), int(boundingBoxC.height * ih) # that is how draw a rectangle
            cv2.rectangle(img , boundingBox, (0,255,255),2)

            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (boundingBox[0], boundingBox[1]-10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
            #cx, cy = int(lM.x * weight), int(lM.y * height)
           # print(id, cx, cy)
        #mpDraw.draw_detection(img, detection)
    cTime = tm.time()
    fps = 1 / (cTime - iTime)
    iTime = cTime
    cv2.putText(img, f"FPS : {int(fps)}", (70, 50), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(10)