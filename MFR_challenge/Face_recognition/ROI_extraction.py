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
    
    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)
    
    # showing image
    plt.imshow(img2)
    plt.axis('off')
    plt.title("Second")

#extract ROI using dlib detector
def search_landmark(image_path):
    img = plt.imread(image_path)
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

#------EXTRACT SELF-ROI OF FACES BY USING DLIB-68LANDMARKS-----#
from collections import OrderedDict

#extract only eye- and eyebrow region
FACIAL_LANDMARKS_IDXS = OrderedDict([("right_eyebrow", (17, 22)),
("left_eyebrow", (22, 27)),
("right_eye", (36, 42)),
("left_eye", (42, 48))])
roi_tuple = []
clone_tuple = []

def extract_ROI(img_path, shape_predictor_68_face_landmarks_path):
    img = cv2.imread(img_path)
    x = PyWebFace260M()
    bbox, det_feature = x.detect(img)
    #convert cv2 rectangle to dlib rectangle
    left = bbox[0][0]
    top = bbox[0][1]
    right = bbox[0][2]
    bottom = bbox[0][3]

    rect = dlib.rectangle(int(left), int(top), int(right), int(bottom)) 
    print("face: ", rect)

    landmark_detect = dlib.shape_predictor(shape_predictor_68_face_landmarks_path) #facial-landmarks-recognition/shape_predictor_68_face_landmarks.dat
    landmark_tuple = []
    
    shape_ROI = landmark_detect(img, rect)
    shape_ROI = face_utils.shape_to_np(shape_ROI) #convert landmark (x,y) to array

    for (name, (i,j)) in FACIAL_LANDMARKS_IDXS.items():
        img_clone = img.copy()

        for (x, y) in shape_ROI[i:j]:
            cv2.circle(img_clone, (x, y), 1, (0, 0, 255), -1)
        
        (x, y, w, h) = cv2.boundingRect(np.array([shape_ROI[i:j]]))
        roi = img[y:y + h, x:x + w]
        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
        roi_tuple.append(roi)
        clone_tuple.append(img_clone)

    ################################################################################
    #extract eye, eyebrow area histogram in LBP format

    hist0, img_lbp0 = calculate_LBP(roi_tuple[0])
    hist1, img_lbp1 = calculate_LBP(roi_tuple[1])
    hist2, img_lbp2 = calculate_LBP(roi_tuple[2])
    hist3, img_lbp3 = calculate_LBP(roi_tuple[3])

    #convert histogram in list 
    hist0 = list(hist0)
    hist1 = list(hist1)
    hist2 = list(hist2)
    hist3 = list(hist3)

    #concatenate histograms into a histogram
    hist = hist0 + hist1 + hist2 + hist3

    return hist
    

#------- encode ROI of faces in LBP and plot histogram-------------
def calculate_LBP(img_path):    
    img = cv2.imread(img_path)
    def assign_bit(image, x, y, c):  #comparing bit with threshold value of centre pixel
        bit = 0  
        try:          
            if image[x][y] >= c: 
                bit = 1         
        except: 
            pass
        return bit 
        
    def local_bin_val(image, x, y):  #calculating local binary pattern value of a pixel
        eight_bit_binary = []
        centre = image[x][y] 
        powers = [1, 2, 4, 8, 16, 32, 64, 128] 
        decimal_val = 0
        #starting from top right,assigning bit to pixels clockwise 
        eight_bit_binary.append(assign_bit(image, x-1, y + 1,centre)) 
        eight_bit_binary.append(assign_bit(image, x, y + 1, centre)) 
        eight_bit_binary.append(assign_bit(image, x + 1, y + 1, centre)) 
        eight_bit_binary.append(assign_bit(image, x + 1, y, centre)) 
        eight_bit_binary.append(assign_bit(image, x + 1, y-1, centre)) 
        eight_bit_binary.append(assign_bit(image, x, y-1, centre)) 
        eight_bit_binary.append(assign_bit(image, x-1, y-1, centre)) 
        eight_bit_binary.append(assign_bit(image, x-1, y, centre))     
        #calculating decimal value of the 8-bit binary number
        for i in range(len(eight_bit_binary)): 
            decimal_val += eight_bit_binary[i] * powers[i] 
            
        return decimal_val 
    m, n, _ = img.shape 
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #converting image to grayscale
    lbp_img = np.zeros((m, n),np.uint8) 
    # converting image to lbp
    for i in range(0,m): 
        for j in range(0,n): 
            lbp_img[i, j] = local_bin_val(gray_scale, i, j) 

    n_bins1 = int(lbp_img.max() + 1)
    hist, _ = np.histogram(lbp_img, density=True, bins=n_bins1, range=(0, n_bins1))   #get the histogram for distance calculation task
    return hist, lbp_img

