import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import pandas as pd
import cv2
from sklearn import metrics
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description="Run Arcface model validation.")
parser.add_argument("--path", type=str, default="../", help="mfr implementation path")
args = parser.parse_args()
_path = args.path

sys.path.append(_path)
#---create a repository to store model.h5, create folder to store test-set
# !mkdir model
#!mkdir testsets
#---download model from gg drive
# import gdown
# !gdown 1LVB3CdVejpmGHM28BpqqkbZP5hDEcdZY

#load model
from resnet_arcface import loadModel

model_path = os.path.join(_path, 'model', 'arcface_weights.h5')
model = loadModel(model_path)
#model.summary()
#load dataset
# manual move some subfolder(each subfolder contains diffent persons) from COMASK20 dataset into testsets 
testsets_path = os.path.join(_path, 'testsets')
faces, labels, frames, base_imgs = load_face_dataset(testsets_path)

#extract facial Embeddings
embeddings = []
flags = []
for i in faces:
  embedding = model.predict(i)[0]
  embeddings.append(embedding)

#extract histogram of all faces (eye, eyebrow regions)
hists = []
for face in frames:
  hist = extract_ROI(face)
  hists.append(hist)

#find min distance of each pair of face [i] to all faces and return 1 for prediction (append 1 into flags)
for i in range (0, len(LBP_faces)-1, 2):
  hist_distance = lbp_recognizer(frames[i], frames[j])
