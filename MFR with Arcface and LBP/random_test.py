import os
import glob
import random
import time

random.seed(2480)


data_path = "../comask20_testset/"
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