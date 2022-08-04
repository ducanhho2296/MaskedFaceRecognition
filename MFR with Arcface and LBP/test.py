import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import pandas as pd
import cv2
from sklearn import metrics
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from ROI_extraction import extract_ROI
from resnet_arcface import findCosineDistance


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

#---load dataset-----
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
  
#store all cos_distance into a tuple 
#if cos = 1 and label1 == label2 => True Positive
#if cos = 0 and label1 != label2 => True negative
#if cos = 0.5, put these faces into a tuple, after that run LBP to choose min distance.

label_flags = []  #tuple of ground true, y_test
label_flag = 0    #ground true, 2 faces have the same label => label_flag = 1 
flags = []        #tuple of y_pred (after finding Cosine similarity)
LBP_flags = []    #tuple of faces, which will be classify by using LBP
LBP_labels =[]    #store labels to use in LBP task

for i in range(len(embeddings)):
  hist_distances = []
  for j in range(len(embeddings) - 1, 0, -1):
    if i != j:
      flag = findCosineDistance(embeddings[i], embeddings[j])
      if flag != 0.5:
        flags.append(flag)

      elif flag == 0.5:
        if len(hists[i]) == len(hists[j]):
          dist = 1 - np.dot(hist[i], hist[j])
        else: dist = 10
        hist_distances.append(dist)

      #store labels for testing prediction  
      if labels[i] == labels[j] and flag != 0.5: 
        label_flag = 1
        label_flags.append(label_flag)
      elif labels[i] != labels[j] and flag != 0.5:
        label_flag = 0
        label_flags.append(label_flag)
      # else: print("this face must be verified by LBP")
      # label_flags.append(label_flag)
      # label_flag = 0

    else: continue  

#LBP prediction part
  if len(hist_distances) != 0:
    min_hist = min(hist_distances)
    min_index = hist_distances.index(min_hist)
    LBP_flags.append(1)
    if labels[min_index] == labels[min_index +1]: LBP_labels.append(1) #one hist have 2 faces, hist0 = dist(faces[0], faces[1])
    else: LBP_labels.append(0)

#-------Accuracy and ROC curve-----------#
y_pred = len([i for i, j in zip(flags, label_flags) if i == j])
y_test = len(flags)
acc = y_pred/y_test *100
print("accuracy without LBP is: ", acc)
print("F1 score without LBPis: ", f1_score(label_flags, flags))
print("\n")

fpr, tpr, _ = metrics.roc_curve(label_flags, flags)
auc = metrics.roc_auc_score(label_flags, flags)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.title("ROC CURVE without LBP")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

input("________Press Enter to continue..._________")

#-----Accuracy with LBP method----------#
total_flags = flags + LBP_flags
total_labels = label_flags + LBP_labels

y_pred_lbp = len([i for i, j in zip(total_flags, total_labels) if i == j])
y_test_lbp = len(total_flags)
acc = y_pred_lbp / y_test_lbp * 100
print("accuracy with LBP is: ", acc)
print("F1 score with LBP is: ", f1_score(total_labels, total_flags))
print("\n")
fpr, tpr, _ = metrics.roc_curve(total_labels, total_flags)
auc = metrics.roc_auc_score(total_labels, total_flags)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.title("ROC CURVE with LBP")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()