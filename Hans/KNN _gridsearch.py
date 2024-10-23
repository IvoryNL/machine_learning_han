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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_train)

    # Create a pipeline for standardization and KNN
    pipeline = Pipeline([
        ('knn', KNeighborsClassifier())
    ])

    # Define the parameter grid for Grid Search
    param_grid = {
        'knn__n_neighbors': np.arange(1, 32)
    }

    # Perform Grid Search with Cross-Validation
    #grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    #grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='precision_weighted')
    #grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='recall_weighted')
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted')
    grid_search.fit(X_train_scaled, y_train)

    # Extract the results
    results = grid_search.cv_results_

    # Plot the cross-validation results
    plt.figure(figsize=(10, 6))
    plt.plot(param_grid['knn__n_neighbors'], results['mean_test_score'], marker='o')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Cross-Validated f1-score')
    plt.title('KNN Cross-Validation Results')
    plt.grid(True)
    plt.show()

    # Best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best parameters: {best_params}")
    print(f"Best cross-validated F1-score: {best_score:.2f}")

    # Make predictions on the test set
    y_pred = grid_search.predict(X_test_scaled)

    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels='12345')
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    # Compute performance measures
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=None))