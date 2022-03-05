import cv2
import time
import numpy as np
import PoseModule as pm

cap = cv2.VideoCapture('2.mp4')

detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0

while True:
    success, img = cap.read()
    # img = cv2.imread('Resources/Exercise/6.jpg')
    img = cv2.resize(img, (1100, 700))
    img = detector.findPose(img, False)
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
        cv2.rectangle(img, (900, 100), (975, 650), color, 3)
        cv2.rectangle(img, (900, int(bar)), (975, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)} %', (900, 75),
                    cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

        # press up count
        cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, 670),
                    cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
