import sys
import os
import datetime
import numpy as np
import cv2
import argparse
import imutils
from imutils import face_utils
import dlib
import matplotlib.pyplot as plt 
from itertools import chain
import tensorflow as tf
import onnx 

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser(description='Run MFR online validation.')
parser.add_argument('--path', type=str, default='../',
                    help='mfr implementation path')
args = parser.parse_args()
_path = args.path

sys.path.append(_path)
from pywebface260mmfr_implement import PyWebFace260M

# x = PyWebFace260M()
# assets_path = os.path.join(_path, 'assets')
# x.load(assets_path)
# feat_len = x.feat_dim 
# print('feat length:', feat_len)
# img1 = []
# feat_list = []

#.............Print Model ONNX architecture.............
# assets_path_r50 = os.path.join(_path, 'assets', 'face_reg','R18.onnx')
# model_onnx = onnx.load(assets_path_r50)
# outputs = []
# for node in model_onnx.graph.node:
#     print(node.name, node.output)
#     outputs.extend(node.output)

#--------------load dataset
from imutils import paths
import numpy as np
import cv2
import os


def load_face_dataset(inputPath, minConfidence=0.5,
	minSamples=1):
  imagePaths = list(paths.list_images(inputPath))
  names = [p.split(os.path.sep)[-2] for p in imagePaths]
  (names, counts) = np.unique(names, return_counts=True)
  names = names.tolist()
 
#   faces = [] 
  labels = []

  for imagepath in imagePaths:
    # image = cv2.imread(imagepath)
    name = imagepath.split(os.path.sep)[-2]
    # print(name)
    if counts[names.index(name)] < minSamples:
      continue
    # faces.append(image)
    labels.append(name)
  
#   faces = np.array(faces)
  labels= np.array(labels)

  return labels

#load dataset with os.path.join
dataset_path = os.path.join(_path, "dataset")
b = load_face_dataset(dataset_path)
print(b)

#########################################################################
input("--------------press Enter to continue next part--------------")
for i in range(14):
    img_path = "{}.png".format(i)
    ta = datetime.datetime.now()
    img = cv2.imread(img_path)
    img1 = img.copy()
    feat = x.get_feature(img)    ##############can be used for recognition of faces after detection
    print("print type of img:", img.shape)
    tb = datetime.datetime.now()
    print('cost:', (tb - ta).total_seconds())
    feat_list.append(feat)

feat1 = feat_list[0]
feat2 = feat_list[1]

sim = x.get_sim(feat1, feat2)

print("sim: ", sim)

input("--------------press Enter to continue next part--------------")
#######################################################################3
img = cv2.imread("12.png")

bbox, det_feature = x.detect(img)
print ("bbox: ",bbox)
print("bbox type:", type(bbox))
print (">>>>\nfeature:", det_feature)
left = bbox[0][0]
top = bbox[0][1]
right = bbox[0][2]
bottom = bbox[0][3]

face = dlib.rectangle(int(left), int(top), int(right), int(bottom)) 
print("face: ", face)

# extract landmarks with dlib
from itertools import chain
merge_range = chain(range(18,29), range(37, 48)) #take only eyes- and eyebrows area

landmark_detect = dlib.shape_predictor("facial-landmarks-recognition/shape_predictor_68_face_landmarks.dat")
landmark_tuple = []


landmarks = landmark_detect(img, face)
for n in merge_range:
    x = landmarks.part(n).x
    y = landmarks.part(n).y
    landmark_tuple.append((x, y))
    cv2.circle(img, (x, y), 2, (255, 255, 0), -1)

cv2.imwrite("demo.jpg", img)


