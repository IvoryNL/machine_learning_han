import os
import cv2 as cv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

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

def convertToBlackWhiteVariables(img,hue_min,hue_max,sat_min,sat_max,val_min,val_max):
    # Change image color space
    img_resized = cv.blur(img, (3, 3))
    img_hsv = cv.cvtColor(img_resized, cv.COLOR_BGR2HSV)

    # Define background color range in HSV space
    #light_blue = (90, 0, 0)
    #dark_blue = (130, 255, 255)
    light_blue = (hue_min,sat_min,val_min)
    dark_blue = (hue_max,sat_max,val_max)

    # Mark pixels outside background color range
    img_BW = ~cv.inRange(img_hsv, light_blue, dark_blue)

    return img_BW

def extract_features(image):
    cv.imshow('KNN input', image)
    # extract features
    # get largest contour
    try:
        contours, hier = cv.findContours(image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv.contourArea)
        contour = np.squeeze(contour)
        perimeter = cv.arcLength(contour, True)
        hull = cv.convexHull(contour, returnPoints=False)
        defects = cv.convexityDefects(contour, hull)
        defects = defects.squeeze()
        defects = defects[defects[:, -1] > 10000]
        total = cv.sumElems(defects[:, -1])[0]
    except Exception as e:
        print(e)
        perimeter = 1
        total = 1

    return [perimeter, total]

def extract_features_list(folder):
    features = []
    labels = []

    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        for filename in os.listdir(label_folder):
            img_path = os.path.join(label_folder, filename)
            img = cv.imread(img_path)
            img_BW = convertToBlackWhite(img)
            img_features = extract_features(img_BW)
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
    data_folder1 = 'C:/Users/roesth/Desktop/Dataset Extended Hans/Data - Train'
    X_train, y_train = extract_features_list(data_folder1)

    # Create and train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=16)
    knn.fit(X_train, y_train)

    cap = cv.VideoCapture(0)

    win_name = "Image Filter"
    cv.namedWindow(win_name)
    # [name, default_value, max_value]
    sliders = [
        ["Hue Min", 90, 180],
        ["Hue Max", 130, 180],
        ["Sat Min", 0, 255],
        ["Sat Max", 255, 255],
        ["Val Min", 0, 255],
        ["Val Max", 255, 255],
    ]

    for slider in sliders:
        cv.createTrackbar(slider[0], win_name, slider[1], slider[2], lambda _: None)

    while True:
        # from video
        ret, frame = cap.read()

        hue_min, hue_max, sat_min, sat_max, val_min, val_max = [
            cv.getTrackbarPos(slider[0], win_name) for slider in sliders
        ]
        img_BW = convertToBlackWhiteVariables(frame,hue_min,hue_max,sat_min,sat_max,val_min,val_max)

        features = extract_features(img_BW)
        features_shaped = np.array(features).reshape(1, -1)
        prediction = knn.predict(features_shaped)
        confidence = 100*np.max(knn.predict_proba(features_shaped))
        annotated_img = annotate_image_with_accuracy(frame, prediction, confidence)

        cv.imshow('KNN output', annotated_img)
        cv.waitKey(200)

