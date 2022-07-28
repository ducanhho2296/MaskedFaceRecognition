#!pip install dlib 
import dlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pywebface260mmfr_implement import PyWebFace260M
import imutils
from imutils import face_utils

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
