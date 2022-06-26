#!pip install dlib 
import dlib
import matplotlib.pyplot as plt
import cv2
import numpy as np

def show_img(img_path1, img_path2):
    img = plt.imread(img_path1)
    img2 = plt.imread(img_path2)
    fig = plt.figure(figsize=(10, 10))

    rows = 2
    columns = 2
    fig.add_subplot(rows, columns, 1)

    # showing image
    plt.imshow(img)
    plt.axis('off')
    plt.title("First")
    
    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)
    
    # showing image
    plt.imshow(img2)
    plt.axis('off')
    plt.title("Second")

def search_landmark(image_path):
    img = plt.imread(image_path)
    #!git clone https://github.com/italojs/facial-landmarks-recognition
    face_detect = dlib.get_frontal_face_detector()
    landmark_detect = dlib.shape_predictor("facial-landmarks-recognition/shape_predictor_68_face_landmarks.dat")
    face = face_detect(img, 1)

    from itertools import chain
    merge_range = chain(range(1,30), range(36, 48)) #take only eyes- and eyebrows area
    # x=y=d=0
    landmark_tuple = []
    for k, d in enumerate(face):
        print("d: ", d)
        landmarks = landmark_detect(img, d)
        for n in merge_range:
            x1 = landmarks.part(n).x
            y1 = landmarks.part(n).y
            landmark_tuple.append((x1, y1))
            cv2.circle(img, (x1, y1), 2, (255, 255, 0), -1)
            x1 = y1 = 0
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    return landmark_tuple