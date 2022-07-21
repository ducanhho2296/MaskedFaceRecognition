import os
import numpy as np
import pandas as pd
import cv2
import base64
from pathlib import Path
from PIL import Image
import requests
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import img_process



def load_image(img):
	exact_image = False; base64_img = False; url_img = False

	if type(img).__module__ == np.__name__:
		exact_image = True

	elif len(img) > 11 and img[0:11] == "data:image/":
		base64_img = True

	elif len(img) > 11 and img.startswith("http"):
		url_img = True

	#---------------------------

	if base64_img == True:
		img = loadBase64Img(img)

	elif url_img:
		img = np.array(Image.open(requests.get(img, stream=True).raw).convert('RGB'))

	elif exact_image != True: #image path passed as input
		if os.path.isfile(img) != True:
			raise ValueError("Confirm that ",img," exists")

		img = cv2.imread(img)

	return img


def preprocess_img(img, target_size=(112, 112), 
                   grayscale=False,
                   enforce_detection=True,
                   detector_backend='retinaface',
                   return_region=False,
                   align=True):
  img = load_image(img)
  base_img = img.copy()

  if img.shape[0] == 0 or img.shape[1] == 0:
    if enforce_detection == True:
      raise ValueError("Detected face shape is ", img.shape,
                    ". Consider to set enforce_detection argument to False.")
    else: #restore base image
      img = base_img.copy()
#---------------------------------------------------------------

  #extract face from images using OPENCV model

  (h, w) = img.shape[:2]
  blob = cv2.dnn.blobFromImage(cv2.resize(img, (112, 112)), 1.0, (112, 112), (104.0, 177.0, 123.0))
  protopath = "/Create-Face-Data-from-Images/model_data/deploy.prototxt"
  weightpath = "/Create-Face-Data-from-Images/model_dataweights.caffemodel"
  model = cv2.dnn.readNetFromCaffe(protopath, weightpath)

  model.setInput(blob)
  detections = model.forward()
  
  for i in range(0, detections.shape[2]):
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    confidence = detections[0, 0, i, 2]

    # If confidence > 0.5, show box around face
    if (confidence > 0.5):
      cv2.rectangle(img, (startX, startY), (endX, endY), (255, 255, 255), 2)

  for i in range(0, detections.shape[2]):
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    confidence = detections[0, 0, i, 2]

    # If confidence > 0.5, save it as a separate file
    if (confidence > 0.5):
      img = img[startY:endY, startX:endX]
  frame = img.copy()
	#--------------------------

	#post-processing
  if grayscale == True:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#---------------------------------------------------
	#resize image to expected shape


  if img.shape[0] > 0 and img.shape[1] > 0:
    factor_0 = target_size[0] / img.shape[0]
    factor_1 = target_size[1] / img.shape[1]
    factor = min(factor_0, factor_1)

    dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
    img = cv2.resize(img, dsize)

    # Then pad the other side to the target size by adding black pixels
    diff_0 = target_size[0] - img.shape[0]
    diff_1 = target_size[1] - img.shape[1]
    if grayscale == False:
			# Put the base image in the middle of the padded image
      img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), 
                      (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
    else:
      img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), 
                      (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

	#------------------------------------------

	#double check: if target image is not still the same size with target.
  if img.shape[0:2] != target_size:
    img = cv2.resize(img, target_size)

	#---------------------------------------------------
	#normalizing the image pixels

  img_pixels = img_process.img_to_array(img) #what this line doing? must?
  img_pixels = np.expand_dims(img_pixels, axis = 0)
  img_pixels /= 255 #normalize input in [0, 1]

	#---------------------------------------------------

	# if return_region == True:
	# 	return img_pixels, region
	# else:
  return img_pixels, frame, base_img


#############################################################################
#-------loop over all images and store images in a tuple, store labels of images into a tuple

def load_face_dataset(inputPath, minConfidence=0.5,
	minSamples=1):
	# grab the paths to all images in our input directory, extract
	# the name of the person (i.e., class label) from the directory
	# structure, and count the number of example images we have per
	# face
  imagePaths = list(paths.list_images(inputPath))
  # print(imagePaths)
  names = [p.split(os.path.sep)[-2] for p in imagePaths]
  (names, counts) = np.unique(names, return_counts=True)
  names = names.tolist()
  
  faces = [] #store face images after processing
  labels = [] #store labels
  frames = [] #show face image without processing
  base_imgs = []
  for imagepath in imagePaths:
    image, frame, base_img = preprocess_img(imagepath, target_size=(112,112))
    name = imagepath.split(os.path.sep)[-2]
    # print(name)
    if counts[names.index(name)] < minSamples:
      continue
    faces.append(image)
    labels.append(name)
    frames.append(frame)

  faces = np.array(faces)
  labels= np.array(labels)

  return faces, labels, frames, base_imgs