import os
import glob
import random
import time
from image_preprocessing import preprocess_img
from scipy.spatial.distance import pdist
from resnet_arcface import loadModel
import sys
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description="Run Test metric.")
parser.add_argument("--path", type=str, default="../", help="mfr implementation path")
args = parser.parse_args()
_path = args.path

sys.path.append(_path)
print(_path)
model_path = os.path.join(_path, 'model', 'Arcface-Res34.h5')
model = loadModel(model_path)


data_path = os.path.join(_path,"MLFWTestset")
sets = []
for dirname in os.listdir(data_path + "Genuine"):  #subfolder eg.10, 100,102
  genuine_path = data_path + "Genuine" + "/" + dirname + "/*"
  imposter_path = data_path + "Imposter" + "/*"
  genuine = random.sample([file for file in glob.glob(genuine_path)], 3)
  imposter = random.sample([file for file in glob.glob(imposter_path)], 1)

  sets.append(imposter + genuine)

correct_predictions = 0
genuine = 'Genuine'
imposter = 'Imposter'

for set in sets:
  faces = []
  frames = []
  for i in set:
    try:
      face, frame, base_img = preprocess_img(i, target_size=(112,112))
      faces.append(face)
    except:
      continue

  embeddings = []
  for face in faces:
        embedding = model.predict(face)[0]
        embeddings.append(embedding)
  
  y = pdist(embeddings, metric='cosine') 

#y[5] = [emb2-emb3]
  if y[0] < 0.55 or y[1] < 0.55 or y[2] < 0.55: 
    y_pred = 'Genuine'
  else: y_pred = 'Imposter'
  if y_pred in set[0]:
    correct_predictions += 1

print("Accuracy: ", correct_predictions / (len(os.listdir(data_path + "Genuine"))))


input('press Enter to continue test with another metric')
#-------------------------------------------------------------------------#
#another test metric
random.seed(2480)
data_path = "../MLFWTestset/"
correct_predictions = 0
time_per_prediction = 0
start_total = time.time()
bad_predictions = []

for dirname in os.listdir(data_path):
    genuine = [file for file in glob.glob(data_path + dirname + "/*")[0:6]]
    random.shuffle(genuine)
    imposter = random.sample([file for file in glob.glob(data_path + dirname + "/*")[6:]], 2)

    sets = [[imposter[0]] + genuine[0:3], [imposter[1]] + genuine[3:]]

    imposter = []
    faces = []
    embeddings = []
    acc_per_set = []
    for set in sets:
        faces = []
        for face in set:
            face, frame, base_img = preprocess_img(face, target_size=(112,112))
            faces.append(face)
        start_pred = time.time()
        # TODO add prediction here
        time_per_prediction += time.time() - start_pred
        for face in faces:
            embedding = model.predict(face)[0]
            embeddings.append(embedding)

    # TODO determine index of imposter and add to imposter list

    if imposter[0] == 0:
        correct_predictions += 1
    else:
        bad_predictions.append(sets[0])
    
    if imposter[1] == 0:
        correct_predictions += 1
    else:
        bad_predictions.append(sets[0])

print("Bad predictions: ", bad_predictions)

print("Accuracy: ", correct_predictions / (len(os.listdir(data_path)) * 2))
print("Total time: {}s\tPrediction time: {}s\tMean prediction time per set: {}s".format(time.time() - start_total, 
                                                                                        time_per_prediction, time_per_prediction / (len(os.listdir(data_path)) * 2)))

print("----------------------------------------")        
print("total Accuracy of all Test Batches: ", np.mean(acc_per_sets))