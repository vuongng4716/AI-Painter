import cv2
import numpy as np
import time
import os
import HandTracking as htm

folderPath = "Header"
myList = os.listdir(folderPath)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]
cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector()

while True:
    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find hand Landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=True)
    if len(lmList) != 0:
        # print(lmList[4][1])

        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        #print(fingers)
        # 4. If selection mode - Two fingers are up
        if fingers[1] and fingers[2]:
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), (255, 0, 255), cv2.FILLED)

            # print('Selection Mode')
        # 5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            # print('Drawing Mode')
    img[0:251, 0:693] = header
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


