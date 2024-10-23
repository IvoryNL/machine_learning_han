import os
import glob
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import Bunch
from sklearn.metrics import accuracy_score
from skimage.io import imread
from skimage.transform import resize

def convertToBlackWhite(img):
    # Change image color space
    img_resized = cv.blur(img, (3, 3))
    img_hsv = cv.cvtColor(img_resized, cv.COLOR_BGR2HSV)

    # Define background color range in HSV space
    light_blue = (90, 0, 0)
    dark_blue = (130, 255, 255)

    # Mark pixels outside background color range
    img_BW = ~cv.inRange(img_hsv, light_blue, dark_blue)

    return img_BW

def extract_features(image):
    img_BW = convertToBlackWhite(image)
    cv.imshow('KNN input', img_BW)
    # extract features
    # get largest contour
    contours, hier = cv.findContours(img_BW.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        perimeter = 1
        total = 1
    else:
        contour = max(contours, key=cv.contourArea)
        contour = np.squeeze(contour)

        perimeter = cv.arcLength(contour, True)
        hull = cv.convexHull(contour, returnPoints=False)
        defects = cv.convexityDefects(contour, hull)
        if defects is None:
            total = 1
        else:
            defects = defects.squeeze()
            defects = defects[defects[:, -1] > 10000]
            total = cv.sumElems(defects[:, -1])[0]

    return [perimeter, total]

def extract_features_list(folder):
    features = []
    labels = []

    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        for filename in os.listdir(label_folder):
            img_path = os.path.join(label_folder, filename)
            img = cv.imread(img_path)
            img_features = extract_features(img)
            features.append(img_features)
            labels.append(label)
    return np.array(features), np.array(labels)

def annotate_image_with_accuracy(image, prediction, confidence, position=(10, 30)):
    annotated_image = image.copy()
    text = f'Prediction: {prediction}, Confidence: {confidence}'
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 0, 0)  # Blue color in BGR
    thickness = 2
    cv.putText(annotated_image, text, position, font, font_scale, color, thickness)
    return annotated_image

if __name__ == "__main__":
    # Load training data
    train_folder = 'C:/Users/roesth/Desktop/Dataset Simplified Hans/Train'
    X, y = extract_features_list(train_folder)

    # Standardize the features
    scaler = StandardScaler()
    X_val_scaled = scaler.fit_transform(X)

    # Create and train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_val_scaled, y)

    # Open webcam and try KNN on new pictures
    cap = cv.VideoCapture(0)
    while True:
        # from video
        ret, frame = cap.read()

        features = extract_features(frame)
        features_shaped = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_shaped)
        prediction = knn.predict(features_scaled)
        confidence = 100*np.max(knn.predict_proba(features_scaled))

        annotated_img = annotate_image_with_accuracy(frame, prediction, confidence)
        cv.imshow('KNN output', annotated_img)

        print(f'Predicted class for the new image: {prediction}')
        cv.waitKey(20)