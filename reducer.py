from cv2 import cv2
import os

vidcap = cv2.VideoCapture("data/contains.mp4")
success, image = vidcap.read()
count = 0
os.mkdir('contain')
while success:
    cv2.imwrite("contain/frame%d.jpg" % count, image)
    success, image = vidcap.read()
    print("Read a new frame: ", success)
    count += 1