import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
import random

# 设置当前工作目录为脚本所在的目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("当前工作目录:", os.getcwd())

# 数据加载和增强
def load_data(data_folder, augment=False):
    X, y, image_paths = [], [], []
    for label in os.listdir(data_folder):
        label_folder = os.path.join(data_folder, label)
        if os.path.isdir(label_folder):
            image_files = [f for f in os.listdir(label_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
            for file_name in image_files:
                image_path = os.path.join(label_folder, file_name)
                image = cv.imread(image_path)
                if image is None:
                    continue
                X.append(image)
                y.append(label)
                image_paths.append(image_path)

                # 图像增强
                if augment:
                    # 随机旋转
                    angle = random.choice([90, 180, 270])
                    rotated = cv.rotate(image, cv.ROTATE_90_CLOCKWISE if angle == 90 else (cv.ROTATE_180 if angle == 180 else cv.ROTATE_90_COUNTERCLOCKWISE))
                    X.append(rotated)
                    y.append(label)
                    image_paths.append(image_path)

                    # 随机翻转
                    flipped = cv.flip(image, 1)  # 水平翻转
                    X.append(flipped)
                    y.append(label)
                    image_paths.append(image_path)

                    # 随机缩放
                    scale = random.uniform(0.8, 1.2)
                    h, w = image.shape[:2]
                    scaled = cv.resize(image, (int(w * scale), int(h * scale)))
                    X.append(scaled)
                    y.append(label)
                    image_paths.append(image_path)

    return X, y, image_paths

# 使用 HSV 颜色空间移除背景的函数
def convert_to_black_white(img):
    img_resized = cv.blur(img, (3, 3))
    img_hsv = cv.cvtColor(img_resized, cv.COLOR_BGR2HSV)

    # 定义背景颜色的 HSV 范围
    light_blue = (90, 0, 0)
    dark_blue = (130, 255, 255)

    # 标记在背景颜色范围之外的像素
    mask = cv.inRange(img_hsv, light_blue, dark_blue)
    img_bw = cv.bitwise_not(mask)
    return img_bw

# 特征提取函数
def extract_features(image):
    try:
        contours, _ = cv.findContours(image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv.contourArea)
        contour = np.squeeze(contour)
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        hull = cv.convexHull(contour, returnPoints=False)
        defects = cv.convexityDefects(contour, hull)
        total_defects = defects.shape[0] if defects is not None else 0
    except Exception as e:
        area = 0
        perimeter = 0
        total_defects = 0

    return [area, perimeter, total_defects]

# 提取所有图像的特征（并行化处理）
def extract_features_list(folder, augment=False):
    features, labels, images_no_bg = [], [], []

    X, y, image_paths = load_data(folder, augment=augment)

    def process_image(img, label):
        img_no_bg = convert_to_black_white(img)  # 去除背景
        img_features = extract_features(img_no_bg)
        return img_features, label, img_no_bg

    results = Parallel(n_jobs=-1)(delayed(process_image)(img, lbl) for img, lbl in zip(X, y))

    for features_extracted, label, img_no_bg in results:
        features.append(features_extracted)
        labels.append(label)
        images_no_bg.append(img_no_bg)

    return features, labels, images_no_bg

if __name__ == "__main__":
    # 加载训练数据和测试数据
    train_folder = 'Pic_Data/Train'
    X_train, y_train, _ = extract_features_list(train_folder, augment=True)  # 增强训练数据

    test_folder = 'Pic_Data/Test'
    X_test, y_test, X_test_images_no_bg = extract_features_list(test_folder, augment=False)

    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 创建 KNN 模型
    knn = KNeighborsClassifier()

    # 超参数调优
    param_grid = {
        'n_neighbors': np.arange(1, 32)
    }
    grid_search = GridSearchCV(knn, param_grid, cv=10, scoring='f1_weighted', n_jobs=-1)  # 提高交叉验证折数到 10
    grid_search.fit(X_train_scaled, y_train)

    # 提取最优模型
    best_model = grid_search.best_estimator_

    # 交叉验证结果图表
    os.makedirs('Result/KNN_methode_result', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(param_grid['n_neighbors'], grid_search.cv_results_['mean_test_score'], marker='o')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Cross-Validated f1-score')
    plt.title('KNN Cross-Validation Results')
    plt.grid(True)
    plt.savefig('Result/KNN_methode_result/cross_validation_results.png')
    plt.close()
    print("交叉验证结果图表已保存")

    # 最优参数和分数
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best parameters: {best_params}")
    print(f"Best cross-validated F1-score: {best_score:.2f}")

    # 预测测试集
    y_pred = best_model.predict(X_test_scaled)

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    # 保存混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('Result/KNN_methode_result/confusion_matrix.png')
    plt.close()
    print("混淆矩阵已保存")

    # 计算性能指标
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

    # 保存分类报告
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

    # 保存预测结果的图像
    output_model_folder = 'Result/Output_Data/KNN'
    os.makedirs(output_model_folder, exist_ok=True)
    for idx, (image, pred_label, true_label) in enumerate(zip(X_test_images_no_bg, y_pred, y_test)):
        if image is None:
            continue
        output_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)  # 将黑白图像转换为 BGR 以便绘制彩色文本
        cv.putText(output_image, f"Predicted: {pred_label}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(output_image, f"True: {true_label}", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        output_image_path = os.path.join(output_model_folder, f"result_{idx}.png")
        cv.imwrite(output_image_path, output_image)
    print(f"预测结果已保存到文件夹 {output_model_folder}")

    # 绘制学习曲线
    print("正在绘制学习曲线...")
    train_sizes, train_scores, test_scores = learning_curve(best_model, X_train_scaled, y_train, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
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
