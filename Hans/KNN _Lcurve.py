import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, auc
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve, train_test_split

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
    data_folder1 = 'C:/Users/roesth/Desktop/Dataset Simplified Hans/Data - Train'
    X_train, y_train = extract_features_list(data_folder1)

    data_folder2 = 'C:/Users/roesth/Desktop/Dataset Simplified Hans/Data - Test'
    X_test, y_test = extract_features_list(data_folder2)

    model = KNeighborsClassifier(n_neighbors=16)
    # Generate learning curve data
    train_sizes, train_scores, valid_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )

    # Calculate mean and standard deviation of training and validation scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)

    # Plot the learning curve
    plt.figure()
    plt.title('Learning Curve')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.grid()

    # Plot the training scores with error bars
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")

    # Plot the validation scores with error bars
    plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, valid_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()