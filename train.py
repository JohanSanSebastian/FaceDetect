# Creating database
# It captures images and stores them in datasets
# folder under the folder name of sub_data
import cv2
import os
haar_file = 'haarcascade_frontalface_default.xml'

# All the faces data will be
# present this folder
datasets = 'datasets'
sub_data = input("Who's facial data would I be recording? ")

CWD_path = os.path.abspath(os.getcwd())
dataset_path = f"{CWD_path}\{datasets}"
path = f"{CWD_path}\{datasets}\{sub_data}"

if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

if not os.path.exists(path):
    os.mkdir(path)

# defining the size of images
(width, height) = (130, 100)

# '0' is used for the default video input source 
# (which in my case is the webcam),
# # if you've any other camera
#  attached use '1' like this
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

# The program loops until it has 30 images of the face.
count = 1
while count < 60:
    print(count)
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('% s/% s.png' % (path, count), face_resize)
    count += 1

    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)
    if key == 27:
        break
