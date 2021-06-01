#!/usr/bin/python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 8th July 2018 - before Google inside look 2018 :)
# -------------------------------------------------------------------------

import cv2
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os
import os.path
import sys

# read the test image
try:
    source_image = cv2.imread(sys.argv[1])
except:
    source_image = cv2.imread('Surprised_Pikachu_HD.jpg')
prediction = 'n.a.'

# checking whether the training data is ready
PATH = './data/training.data'

if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    print('training data is ready, classifier is loading...')
else:
    print('training data is being created...')
    open('data/training.data', 'w')
    color_histogram_feature_extraction.training()
    print('training data is ready, classifier is loading...')

# get the prediction
color_histogram_feature_extraction.color_histogram_of_test_image(source_image)
prediction = knn_classifier.main('data/training.data', 'data/test.data')
print('Detected color is:', prediction)

cv2.rectangle(
    source_image,
    (10, 10),
    (250, 50),
    (255, 255, 255),
    -1,
)
cv2.putText(
    source_image,
    'Prediction: ' + prediction,
    (15, 40),
    0,
    0.8,
    (0, 0, 0),
    2,
    cv2.LINE_AA
)

# Display the resulting frame
cv2.imshow('color classifier', source_image)
cv2.waitKey(0)
