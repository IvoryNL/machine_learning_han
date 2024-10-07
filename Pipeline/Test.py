import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

from PreprocessorV2 import PreprocessingPipeline

src = 'D:\\Data\\0. Machine Learning\\0. Mini_project_finger_counting\\0. Data\\TestSetIvan'
pipeline = PreprocessingPipeline()


""""
label = '1'
extracted_images = []
coded_labels = []

for filename in os.listdir(src):
    file_path = os.path.join(src, filename)

    # Check if it's a file (ignores subdirectories, if any)
    if os.path.isfile(file_path):
        image = cv2.imread(file_path)
        extracted_images.append(image)
        coded_labels.append(label)

pipeline = PreprocessingPipeline()
dataset = pipeline.process(extracted_images)
dataset.target = coded_labels

(trainX, testX, trainY, testY) = train_test_split(dataset.data, coded_labels, test_size=0.25, stratify=dataset.target, random_state=42)

print(dataset)
dataset.target.extend(coded_labels)
print(dataset.target)
print(len(dataset.data))
print(len(dataset.target))
"""

dataTrainX = []
dataTrainY = []
dataTestX = []
dataTestY = []

for folder_name in os.listdir(src):
    folder_path = os.path.join(src, folder_name)

    label = folder_name
    extracted_images = []
    coded_labels = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            image = cv2.imread(file_path)
            extracted_images.append(image)
            coded_labels.append(label)

    if len(extracted_images) == 0:
        continue

    dataset = pipeline.process(extracted_images)
    dataset.target = coded_labels

    (trainX, testX, trainY, testY) = train_test_split(dataset.data, coded_labels, test_size=0.25, stratify=dataset.target, random_state=42)

    dataTrainX.extend(trainX)
    dataTrainY.extend(trainY)
    dataTestX.extend(testX)
    dataTestY.extend(testY)
    print(dataset)

classifier = svm.SVC(kernel='rbf')  # You can change the kernel (e.g., 'rbf', 'poly')
classifier.fit(dataTrainX, dataTrainY)

test = []
test.append(dataTestX[30])
train_predictions = classifier.predict(test)
print(train_predictions)
print(test)

# cap = cv2.VideoCapture(0)
#
# if not cap.isOpened():
#     print("Error: Could not open video stream")
#     exit()
#
# print("Press ESC to exit")
#
# while True:
#     ret, frame = cap.read()
#
#     if not ret:
#         print("Failed to grab frame")
#         break
#
#     extracted_images = []
#     extracted_images.append(frame)
#     dataset = pipeline.process(extracted_images)  # This should return the processed data
#
#     print(dataset.data)
#
#     # prediction = classifier.predict(dataset)
#     #
#     # print(f"Prediction: {prediction}")
#
#     cv2.imshow('Video Stream', frame)
#
#     if cv2.waitKey(1) & 0xFF == 27:
#         print("ESC key pressed, exiting...")
#         break
#
# cap.release()
# cv2.destroyAllWindows()