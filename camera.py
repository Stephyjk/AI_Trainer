import cv2
import numpy as np
import time
import math
import PoseModule as pm


class VideoCapture(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        detector = pm.poseDetector()
        count = 0
        dir = 0
        pTime = 0

        ret, frame = self.video.read()
        img = detector.findPose(frame, False)
        lmList = detector.findPosition(img, False)
        if len(lmList) != 0:
            # left arm
            angle = detector.findAngle(img, 11, 13, 15)
            # right arm
            # detector.findAngle(img, 12, 14, 16)
            per = np.interp(angle, (195, 280), (0, 100))
            bar = np.interp(angle, (200, 280), (650, 100))

            # check press ups
            color = (255, 0, 255)
            if per == 100:
                color = (0, 255, 0)
                if dir == 0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0, 255, 0)
                if dir == 1:
                    count += 0.5
                    dir = 0

            # draw bar
            cv2.rectangle(img, (50, 100), (75, 650), color, 3)
            cv2.rectangle(img, (50, int(bar)), (75, 650), color, cv2.FILLED)
            cv2.putText(img, f'{int(per)} %', (50, 75),
                        cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

            # press up count
            cv2.rectangle(img, (300, 50), (400, 150), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(count)}', (550, 120),
                        cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

        # cTime = time.time()
        # fps = 1 / (cTime - pTime)
        # pTime = cTime

        # cv2.putText(img, f'FPS: {int(fps)}', (400, 50),
        #             cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
        # cv2.imshow('img', img)

        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()
