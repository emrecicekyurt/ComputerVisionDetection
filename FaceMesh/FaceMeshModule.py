import cv2
import mediapipe as mp
import time as tm

class FaceMeshDetector():
    def __init__(self, max_num_faces = 2):
        self.max_num_faces= max_num_faces
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.max_num_faces)  # since video includes two people
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)# It is to arragne the thickness of lines and points around face

    def drawFaceMesh(self, img):
        img = rescale_frame(img, 50)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        locations = []
        if results.multi_face_landmarks:
            for eachFace in results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, eachFace, self.mpFaceMesh.FACE_CONNECTIONS, landmark_drawing_spec=self.drawSpec, )
                for id, lms in enumerate(eachFace.landmark):
                    # print(lms)
                    iw, ih, ic = img.shape
                    x, y = int(lms.x * iw), int(lms.y * ih)
                    locations.append([id,x,y])
                    #print(f"The point id : {id} at the location {x, y}")
        return img, locations

def rescale_frame(frame_input, percent=75):
    width = int(frame_input.shape[1] * percent / 100)
    height = int(frame_input.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame_input, dim, interpolation=cv2.INTER_AREA)




def main():
    vCap = cv2.VideoCapture("elonmuskvideo.mp4")
    iTime = 0
    while True:
        success, img = vCap.read()
        faceDetector = FaceMeshDetector()
        img, locList = faceDetector.drawFaceMesh(img)
        print(locList)
        cTime = tm.time()
        fps = 1/(cTime-iTime)
        iTime = cTime
        cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
if __name__ == "__main__":
    main()