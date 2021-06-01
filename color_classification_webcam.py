#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 31st December 2017 - new year eve :)
# ----------------------------------------------

import cv2
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os
import os.path

cap = cv2.VideoCapture(0)
(ret, frame) = cap.read()
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

while True:

    # Capture frame-by-frame
    (ret, frame) = cap.read()

    cv2.rectangle(
        frame,
        (10, 10),
        (250, 50),
        (255, 255, 255),
        -1,
    )
    cv2.putText(
        frame,
        'Prediction: ' + prediction,
        (15, 40),
        0,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    # Display the resulting frame
    cv2.imshow('color classifier', frame)

    color_histogram_feature_extraction.color_histogram_of_test_image(frame)

    prediction = knn_classifier.main('data/training.data', 'data/test.data')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
