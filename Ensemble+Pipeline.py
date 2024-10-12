import os
import sys
import numpy as np
import cv2 as cv
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import Bunch
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# 设置当前工作目录为脚本所在的目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("当前工作目录:", os.getcwd())

# 自定义 Transformer，用于手部提取
class HandExtractionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        processed_images = []
        for image in X:
            if image is None:
                processed_images.append(None)
                continue

            hand = self._extract_hand(image)
            processed_images.append(hand)

        return np.array(processed_images)

    def _extract_hand(self, image):
        if image is None:
            return None

        # 转换为 Lab 色彩空间
        # Convert to Lab color space
        lab = cv.cvtColor(image, cv.COLOR_BGR2Lab)
        lab_channels = cv.split(lab)
        A = lab_channels[1]

        # 应用 Otsu 阈值分割
        # Apply Otsu's thresholding
        _, thresh = cv.threshold(A, 70, 235, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # 形态学操作
        # Morphological operations
        kernel_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
        result_open = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel_open)

        kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
        result_close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel_close)

        morphed_result = cv.bitwise_or(result_open, result_close)
        improve_with_open = cv.morphologyEx(morphed_result, cv.MORPH_OPEN, kernel_open)
        final_result = cv.morphologyEx(improve_with_open, cv.MORPH_CLOSE, kernel_close)

        # 应用掩膜
        # Apply mask
        final_image = cv.bitwise_and(image, image, mask=final_result)

        return final_image

# 自定义 Transformer，用于特征计算
class FeatureCalculationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feature_list = []
        self.processed_images = []  # 保存处理后的图像，用于后续保存预测结果
        for image in X:
            if image is None:
                feature_list.append(np.zeros(4))  # 假设有 4 个特征
                self.processed_images.append(None)
                continue

            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            contour = self.getLargestContour(gray)
            if contour is None:
                feature_list.append(np.zeros(4))
                self.processed_images.append(image)
                continue
            features = self.getContourFeatures(contour)
            feature_list.append(features)
            self.processed_images.append(image)  # 保存处理后的图像

        return np.array(feature_list)

    def getContourExtremes(self, contour):
        # 找到轮廓的极值点
        # Find the extreme points of the contour
        left = contour[contour[:, :, 0].argmin()]
        right = contour[contour[:, :, 0].argmax()]
        top = contour[contour[:, :, 1].argmin()]
        bottom = contour[contour[:, :, 1].argmax()]
        return np.array((left, right, top, bottom))

    def getConvexityDefects(self, contour):
        hull = cv.convexHull(contour, returnPoints=False)
        defects = cv.convexityDefects(contour, hull)
        if defects is not None:
            defects = defects.squeeze()
        return defects

    def getContourFeatures(self, contour):
        # 计算轮廓特征
        # Calculate contour features
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        x, y, w, h = cv.boundingRect(contour)
        rect_area = w * h
        extent = float(area) / rect_area if rect_area > 0 else 0

        defects = self.getConvexityDefects(contour)
        if defects is not None and defects.size > 0:
            if len(defects.shape) == 1:
                defects = defects.reshape(1, -1)
            defects = defects[defects[:, -1] > 10000]
            total_defects = np.sum(defects[:, -1]) if defects.size > 0 else 0
        else:
            total_defects = 0

        features = np.array([area, perimeter, total_defects, extent])
        return features

    def getLargestContour(self, img_BW):
        # 找到最大的轮廓
        # Find the largest contour
        contours, _ = cv.findContours(img_BW.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        contour = max(contours, key=cv.contourArea)
        return contour

# 设置数据集路径
# Set dataset path
dataset_folder = 'Pic_Data'  # 请确保将此路径替换为您的实际数据集路径

# 检查数据集路径是否存在
# Check if dataset path exists
if not os.path.exists(dataset_folder):
    print(f"数据集目录 {dataset_folder} 不存在。请检查路径是否正确。")
    sys.exit(1)

# 加载训练和测试数据
X_train = []
y_train = []
X_test = []
y_test = []
X_train_paths = []
X_test_paths = []

# 定义一个函数来加载数据
def load_data(data_folder, X, y, image_paths):
    if not os.path.exists(data_folder):
        print(f"目录 {data_folder} 不存在。")
        return
    # 遍历每个类别文件夹
    # Iterate through each class folder
    for label in os.listdir(data_folder):
        label_folder = os.path.join(data_folder, label)
        if os.path.isdir(label_folder):
            # 获取该类别下的所有图像文件
            # Get all image files in the class folder
            image_files = [f for f in os.listdir(label_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
            for file_name in image_files:
                image_path = os.path.join(label_folder, file_name)
                image = cv.imread(image_path)
                if image is None:
                    continue  # 如果图像加载失败，跳过该图像
                X.append(image)
                y.append(label)  # 子文件夹的名称作为标签
                image_paths.append(image_path)  # 保存图像路径

# 加载训练集数据
# Load training data
train_folder = os.path.join(dataset_folder, 'Train')
if not os.path.exists(train_folder):
    print(f"训练数据目录 {train_folder} 不存在。请检查路径是否正确。")
    sys.exit(1)
load_data(train_folder, X_train, y_train, X_train_paths)

# 加载测试集数据
# Load testing data
test_folder = os.path.join(dataset_folder, 'Test')
if not os.path.exists(test_folder):
    print(f"测试数据目录 {test_folder} 不存在。请检查路径是否正确。")
    sys.exit(1)
load_data(test_folder, X_test, y_test, X_test_paths)

# 将标签编码为数字形式
# Encode labels to numeric form
le = LabelEncoder()
# 组合训练和测试集的标签，以确保编码一致
# Combine labels from training and testing sets to ensure consistent encoding
all_labels = y_train + y_test
le.fit(all_labels)
# 分别转换训练和测试集的标签
# Transform labels for training and testing sets
y_train_encoded = le.transform(y_train)
y_test_encoded = le.transform(y_test)

# 定义结果保存目录
# Define the directory to save results
result_folder = os.path.join('Result', 'Ensemble_methode_result')
os.makedirs(result_folder, exist_ok=True)

# 定义预测结果图片保存目录
# Define the directory to save output images
output_folder = os.path.join('Result', 'Output_Data')
os.makedirs(output_folder, exist_ok=True)

# 定义所有的 Ensemble 方法
# Define all Ensemble methods
ensemble_methods = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Bagging': BaggingClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Voting': VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('svc', SVC(probability=True, random_state=42))
        ],
        voting='soft'
    ),
    'Stacking': StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('svc', SVC(probability=True, random_state=42)),
            ('dt', DecisionTreeClassifier(random_state=42))
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5
    )
}

# 创建一个字典来保存结果
# Create a dictionary to save results
results = {}

# 遍历每个 Ensemble 方法
# Iterate through each Ensemble method
for name, model in ensemble_methods.items():
    print(f"正在训练和评估模型：{name}")
    # 创建 Pipeline
    # Create Pipeline
    feature_transformer = FeatureCalculationTransformer()
    pipeline = Pipeline([
        ('hand_extraction', HandExtractionTransformer()),
        ('feature_calculation', feature_transformer),
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

    # 训练模型
    # Train the model
    pipeline.fit(X_train, y_train_encoded)

    # 在测试集上进行预测
    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)

    # 评估模型性能
    # Evaluate model performance
    accuracy = accuracy_score(y_test_encoded, y_pred)
    print(f"{name} 测试集准确率 (Test Accuracy): {accuracy:.3f}")

    # 将预测结果和真实标签转换回原始标签名称
    # Convert predicted and true labels back to original label names
    y_pred_labels = le.inverse_transform(y_pred)
    y_test_labels = le.inverse_transform(y_test_encoded)

    # 获取分类报告
    # Get classification report
    report = classification_report(y_test_labels, y_pred_labels, output_dict=True)

    # 使用交叉验证评估模型（使用训练集数据）
    # Evaluate the model using cross-validation (on training data)
    scores = cross_val_score(pipeline, X_train, y_train_encoded, cv=5, scoring='accuracy')
    cv_mean = scores.mean()
    cv_std = scores.std()
    print(f"{name} 交叉验证准确率 (Cross-Validation Accuracy): {cv_mean:.3f} ± {cv_std:.3f}")

    # 将结果保存到字典中
    # Save results to the dictionary
    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'classification_report': report
    }

    # 将结果保存到指定的目录中
    # Save results to the specified directory
    result_filename = os.path.join(result_folder, f"{name}_results.txt")
    with open(result_filename, 'w', encoding='utf-8') as f:
        f.write(f"模型名称 (Model Name): {name}\n")
        f.write(f"测试集准确率 (Test Accuracy): {accuracy:.3f}\n")
        f.write(f"交叉验证准确率 (Cross-Validation Accuracy): {cv_mean:.3f} ± {cv_std:.3f}\n")
        f.write("\n分类报告 (Classification Report):\n")
        f.write(classification_report(y_test_labels, y_pred_labels))
    print(f"{name} 的结果已保存到文件 {result_filename}\n")

    # 保存预测的图片结果
    # Save the predicted image results
    output_model_folder = os.path.join(output_folder, name)
    os.makedirs(output_model_folder, exist_ok=True)

    for idx, (image, pred_label, true_label, image_path) in enumerate(zip(feature_transformer.processed_images, y_pred_labels, y_test_labels, X_test_paths)):
        if image is None:
            continue  # 如果处理后的图像为 None，跳过
        # 在图像上标注预测结果
        # Annotate the image with prediction results
        output_image = image.copy()
        cv.putText(output_image, f"Predicted: {pred_label}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(output_image, f"True: {true_label}", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 保存图像
        # Save the image
        base_name = os.path.basename(image_path)
        output_image_path = os.path.join(output_model_folder, base_name)
        cv.imwrite(output_image_path, output_image)

    print(f"{name} 的预测结果已保存到文件夹 {output_model_folder}\n")

print("所有模型的训练和评估已完成。")
