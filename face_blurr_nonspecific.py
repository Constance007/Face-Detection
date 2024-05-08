import cv2
from cvzone.FaceDetectionModule import FaceDetector

# Webcam
# cap = cv2.VideoCapture(0)
# cap.set(3, 480)  # width
# cap.set(4, 480)  # height

detector = FaceDetector(minDetectionCon=0.75)

while True:
    # Webcam
    # success, img = cap.read()

    # Read image from directory
    img = cv2.imread('Faces/constance.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, bboxes = detector.findFaces(img, draw=True)

    if bboxes:
        for i, bbox in enumerate(bboxes):
            x, y, w, h = bbox['bbox']
            if x < 0: x = 0
            if y < 0: y = 0

            imgCrop = img[y: y + h, x: x + w]
            imgBlur = cv2.blur(imgCrop, (35, 35))
            img[y: y + h, x: x + w] = imgBlur

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
