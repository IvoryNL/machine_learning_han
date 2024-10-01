import os
import sys
import glob
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


# Custom transformer for cleaning the image (extract hand)
class HandExtractionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X is a list of image file paths
        processed_images = []
        for image_path in X:
            image = cv.imread(image_path)
            if image is None:
                continue
            # Logic to clean the image and extract hand
            hand = self._extract_hand(image)  # Placeholder for hand extraction logic
            processed_images.append(hand)

            cv.imwrite(image_path, hand)

        return np.array(X)

    def _extract_hand(self, image):
        # Check if the image is loaded successfully
        if image is None:
            return

        # Convert BGR image to Lab color space
        lab = cv.cvtColor(image, cv.COLOR_BGR2Lab)

        # Split the Lab channels
        lab_channels = cv.split(lab)
        A = lab_channels[1]

        # Apply Otsu's thresholding on the 'A' channel
        _, thresh = cv.threshold(A, 70, 235, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # Morphological operations
        kernel_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
        result_open = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel_open)

        kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
        result_close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel_close)

        # Combine the results of the open and close operations
        morphed_result = cv.bitwise_or(result_open, result_close)

        # Further improve the result with additional open and close operations
        improve_with_open = cv.morphologyEx(morphed_result, cv.MORPH_OPEN, kernel_open)
        final_result = cv.morphologyEx(improve_with_open, cv.MORPH_CLOSE, kernel_close)

        # Apply the final mask to the original image
        final_image = cv.bitwise_and(image, image, mask=final_result)

        return final_image

# Custom transformer for feature calculation
class FeatureCalculationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X is an array of processed images
        feature_names = ['area', 'perimeter', 'defects', 'extent']
        data = np.empty((0, len(feature_names)), float)
        target = []

        for filename in X:
            image = cv.imread(filename)
            if image is None:
                continue
            # load image and blur a bit to suppress noise
            img = cv.imread(filename)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            contour = self.getLargestContour(gray)

            # extract features from contour
            features = self.getContourFeatures(contour)

            # extract label from folder name and stor
            label = filename.split(os.path.sep)[-2]
            target.append(label)

            # append features to data matrix
            data = np.append(data, np.array([features]), axis=0)

        unique_targets = np.unique(target)
        dataset = Bunch(data=data,
                        target=target,
                        unique_targets=unique_targets,
                        feature_names=feature_names)

        le = LabelEncoder()
        coded_labels = le.fit_transform(dataset.target)

        (trainX, testX, trainY, testY) = train_test_split(dataset.data, coded_labels, test_size=0.25,
                                                          stratify=dataset.target, random_state=42)

        return trainX

    def getContourExtremes(self, contour):
        """ Return contour extremes as an tuple of 4 tuples """
        # determine the most extreme points along the contour
        left = contour[contour[:, 0].argmin()]
        right = contour[contour[:, 0].argmax()]
        top = contour[contour[:, 1].argmin()]
        bottom = contour[contour[:, 1].argmax()]

        return np.array((left, right, top, bottom))


    def getConvexityDefects(self, contour):
        """ Return convexity defects in a contour as an nd.array """
        hull = cv.convexHull(contour, returnPoints=False)
        defects = cv.convexityDefects(contour, hull)
        if defects is not None:
            defects = defects.squeeze()

        return defects


    def getContourFeatures(self, contour):
        """ Return some contour features
        """
        # basic contour features
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        extremePoints = self.getContourExtremes(contour)

        equi_diameter = np.sqrt(4 * area / np.pi)
        defects = self.getConvexityDefects(contour)

        defects = defects.squeeze()
        defects = defects[defects[:, -1] > 10000]
        total = cv.sumElems(defects[:, -1])[0]  # the sum of all the defects
        print(total)

        x, y, w, h = cv.boundingRect(contour)
        rect_area = w * h
        extent = float(area) / rect_area

        features = np.array((area, perimeter, total, extent))

        return (features)


    def getLargestContour(self, img_BW):
        """ Return largest contour in foreground as an nd.array """
        contours, hier = cv.findContours(img_BW.copy(), cv.RETR_EXTERNAL,
                                         cv.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv.contourArea)

        return np.squeeze(contour)

def process_images(input_folder, output_folder):
    # Get image file paths
    image_paths = [os.path.join(input_folder, img) for img in os.listdir(input_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

    # Define preprocessing pipeline
    pipeline = Pipeline([
        ('hand_extraction', HandExtractionTransformer()),
        ('feature_calculation', FeatureCalculationTransformer()),
        ('scaling', StandardScaler())  # Final step to scale the data
    ])

    # Fit and transform the data
    transformed_data = pipeline.fit_transform(image_paths)

    # Save the processed data to the output folder
    output_file = os.path.join(output_folder, 'processed_data.npy')
    np.save(output_file, transformed_data)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    sourcePath = input("Source path: ")
    outputPath = sourcePath + "\o"

    # Ensure output folder exists
    os.makedirs(outputPath, exist_ok=True)

    # Process the images
    process_images(sourcePath, outputPath)
