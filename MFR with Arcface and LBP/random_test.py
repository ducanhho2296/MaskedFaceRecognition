import os
import glob
import random
import time

random.seed(2468)

data_path = "/content/drive/MyDrive/Colab Notebooks/comask20_testset"
for dirname in os.listdir(data_path):
    genuine = [file for file in glob.glob(data_path + dirname + "/*")[0:6]]
    random.shuffle(genuine)
    imposter = random.sample([file for file in glob.glob(data_path + dirname + "/*")[6:]], 2)

    sets = [[imposter[0]] + genuine[0:3], [imposter[1]] + genuine[3:]]

    imposter = []
    for set in sets:
        start_pred = time.time()
        # TODO add prediction here
        time_per_prediction += time.time() - start_pred

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