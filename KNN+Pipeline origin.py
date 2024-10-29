import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
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

def extract_features(image):
    # extract features
    # get largest contour
    try:
        contours, _ = cv.findContours(image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv.contourArea)
        contour = np.squeeze(contour)
        perimeter = cv.arcLength(contour, True)
        hull = cv.convexHull(contour, returnPoints=False)
        defects = cv.convexityDefects(contour, hull)

        # Calculate the total defect length
        if defects is not None and defects.shape[0] > 0:
            defects = defects.squeeze()
            total_defects = np.sum(defects[:, -1]) if len(defects.shape) > 1 else defects[-1]
        else:
            total_defects = 0

    except Exception as e:
        print(f"Error extracting features: {e}")
        perimeter = 1
        total_defects = 1

    return [perimeter, total_defects]

def extract_features_list(folder):
    features = []
    labels = []
    original_images = []  # 用于保存原始图像，以便最后标注预测

    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        for filename in os.listdir(label_folder):
            img_path = os.path.join(label_folder, filename)
            img = cv.imread(img_path)
            if img is None:
                continue

            original_images.append(img)  # 保存原始图像
            img_BW = convertToBlackWhite(img)
            img_features = extract_features(img_BW)
            features.append(img_features)
            labels.append(label)
    return np.array(features), np.array(labels), original_images

if __name__ == "__main__":
    # Load training data
    train_folder = 'Pic_Data/Train'
    X_train, y_train, _ = extract_features_list(train_folder)

    test_folder = 'Pic_Data/Test'
    X_test, y_test, original_images = extract_features_list(test_folder)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create a pipeline for standardization and KNN
    pipeline = Pipeline([
        ('knn', KNeighborsClassifier())
    ])

    # Define the parameter grid for Grid Search
    param_grid = {
        'knn__n_neighbors': np.arange(1, 32)
    }

    # Perform Grid Search with Cross-Validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted')
    grid_search.fit(X_train_scaled, y_train)

    # Extract the results
    results = grid_search.cv_results_

    # Plot the cross-validation results
    os.makedirs('Result/KNN_methode_result', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(param_grid['knn__n_neighbors'], results['mean_test_score'], marker='o')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Cross-Validated f1-score')
    plt.title('KNN Cross-Validation Results')
    plt.grid(True)
    plt.savefig('Result/KNN_methode_result/cross_validation_results.png')
    plt.close()
    print("交叉验证结果图表已保存")

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
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('Result/KNN_methode_result/confusion_matrix.png')
    plt.close()
    print("混淆矩阵已保存")

    # Compute performance measures
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    report = classification_report(y_test, y_pred, target_names=np.unique(y_test))
    print("\nClassification Report:\n", report)

    # Save classification report
    report_path = 'Result/KNN_methode_result/KNN_results.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Best parameters: {best_params}\n")
        f.write(f"Best cross-validated F1-score: {best_score:.2f}\n")
        f.write(f"Accuracy: {accuracy:.2f}\n")
        f.write(f"Precision: {precision:.2f}\n")
        f.write(f"Recall: {recall:.2f}\n")
        f.write(f"F1 Score: {f1:.2f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
    print(f"分类报告已保存为 {report_path}")

    # 保存预测的图片结果
    output_model_folder = 'Result/Output_Data/KNN'
    os.makedirs(output_model_folder, exist_ok=True)
    for idx, (image, pred_label, true_label) in enumerate(zip(original_images, y_pred, y_test)):
        if image is None:
            continue

        output_image = image.copy()
        # Draw a solid border around the predicted area to ensure it stands out
        cv.rectangle(output_image, (0, 0), (output_image.shape[1], output_image.shape[0]), (0, 255, 0), 10)
        cv.putText(output_image, f"Predicted: {pred_label}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(output_image, f"True: {true_label}", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        output_image_path = os.path.join(output_model_folder, f"result_{idx}.png")
        cv.imwrite(output_image_path, output_image)

    print(f"预测结果已保存到文件夹 {output_model_folder}")

    # 绘制学习曲线
    print("正在绘制学习曲线...")
    train_sizes, train_scores, test_scores = learning_curve(grid_search.best_estimator_, X_train_scaled, y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve for KNN')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('Result/KNN_methode_result/learning_curve.png')
    plt.close()
    print("学习曲线已保存")
