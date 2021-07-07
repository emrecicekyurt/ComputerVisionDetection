import cv2
import mediapipe as mp
import time as tm

vCap = cv2.VideoCapture("elonmuskvideo.mp4")
iTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)# since video includes two people
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2) # It is to arragne the thickness of lines and points around face

def rescale_frame(frame_input, percent=75):
    width = int(frame_input.shape[1] * percent / 100)
    height = int(frame_input.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame_input, dim, interpolation=cv2.INTER_AREA)

while True:
    success, img = vCap.read()
    img = rescale_frame(img, 50)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for eachFace in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, eachFace, mpFaceMesh.FACE_CONNECTIONS, landmark_drawing_spec=drawSpec,)
            for id,lms in enumerate(eachFace.landmark):
                #print(lms)
                iw, ih, ic =img.shape
                x,y = int(lms.x*iw), int(lms.y*ih)
                print(f"The point id : {id} at the location {x,y}")
    cTime = tm.time()
    fps = 1/(cTime-iTime)
    iTime = cTime
    cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)