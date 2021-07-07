import cv2
import mediapipe as mp
import time as tm


def rescale_frame(frame_input, percent=75):
    width = int(frame_input.shape[1] * percent / 100)
    height = int(frame_input.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame_input, dim, interpolation=cv2.INTER_AREA)

class FaceDetector():
    def __init__(self, minDetectionConf= 0.5, ):
        self.minDetectionConf = minDetectionConf
        self.mpFace = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetect = self.mpFace.FaceDetection(self.minDetectionConf)

    def findFaces(self, img, draw=True):
        img = rescale_frame(img, 60)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetect.process(imgRGB)
        boxList = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # print(id, detection)
                boundingBoxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                boundingBox = int(boundingBoxC.xmin * iw), int(boundingBoxC.ymin * ih), \
                              int(boundingBoxC.width * iw), int(
                    boundingBoxC.height * ih)  # that is how draw a rectangle
                boxList.append([id, boundingBox, detection.score])
                self.prettyDraw(img, boundingBox)
                cv2.putText(img, f'{int(detection.score[0] * 100)}%', (boundingBox[0], boundingBox[1] - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
                # cx, cy = int(lM.x * weight), int(lM.y * height)
            # print(id, cx, cy)
            # mpDraw.draw_detection(img, detection) # You may see the points on face by using this
        return img, boxList

    def prettyDraw(self, img , boundingBox, l=30, t=3, rt=1):
        x,y,w,h = boundingBox
        x1, y1 =x+w, y+h
        cv2.rectangle(img, boundingBox, (255, 255, 0), rt)
        # left top
        cv2.line(img, (x,y), (x+l, y), (0,0,255), t)
        cv2.line(img, (x,y), (x , y+ l), (0, 0, 255), t)
        # rigth top
        cv2.line(img, (x1,y), (x1-l, y), (0,0,255), t)
        cv2.line(img, (x1,y), (x1 , y+l), (0, 0, 255), t)
        # right bot
        cv2.line(img, (x1, y1), (x1 - l, y1), (0, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (0, 0, 255), t)
        # left bot
        cv2.line(img, (x, y1), (x + l, y1), (0, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (0, 0, 255), t)
        return img
def main():
    cap = cv2.VideoCapture("elonmuskvideo.mp4")
    iTime = 0  # Initial (previous) Time
    cTime = 0  # Current Time
    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img, boxList = detector.findFaces(img)
        print(boxList)
        cTime = tm.time()
        fps = 1 / (cTime - iTime)
        iTime = cTime
        cv2.putText(img, f"FPS : {int(fps)}", (70, 50), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(10)

if __name__ == "__main__":
    main()