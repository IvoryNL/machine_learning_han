import numpy as np
import cv2 as cv
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch

class HandExtractionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        processed_images = []
        for image in X:
            if image is None:
                continue

            hand = self._extract_hand(image)
            processed_images.append(hand)

        return np.array(processed_images)

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

        for image in X:
            if image is None:
                continue

            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            contour = self.getLargestContour(gray)

            features = self.getContourFeatures(contour)

            data = np.append(data, np.array([features]), axis=0)

        unique_targets = np.unique(target)
        dataset = Bunch(data=data,
                        target=target,
                        unique_targets=unique_targets,
                        feature_names=feature_names)

        return dataset

    def getContourExtremes(self, contour):
        # determine the most extreme points along the contour
        left = contour[contour[:, 0].argmin()]
        right = contour[contour[:, 0].argmax()]
        top = contour[contour[:, 1].argmin()]
        bottom = contour[contour[:, 1].argmax()]

        return np.array((left, right, top, bottom))


    def getConvexityDefects(self, contour):
        hull = cv.convexHull(contour, returnPoints=False)
        defects = cv.convexityDefects(contour, hull)
        if defects is not None:
            defects = defects.squeeze()

        return defects


    def getContourFeatures(self, contour):
        # basic contour features
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        extremePoints = self.getContourExtremes(contour)

        equi_diameter = np.sqrt(4 * area / np.pi)
        defects = self.getConvexityDefects(contour)

        defects = defects.squeeze()
        defects = defects[defects[:, -1] > 10000]
        total = cv.sumElems(defects[:, -1])[0]

        x, y, w, h = cv.boundingRect(contour)
        rect_area = w * h
        extent = float(area) / rect_area

        features = np.array((area, perimeter, total, extent))

        return (features)


    def getLargestContour(self, img_BW):
        contours, hier = cv.findContours(img_BW.copy(), cv.RETR_EXTERNAL,
                                         cv.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv.contourArea)

        return np.squeeze(contour)

class PreprocessingPipeline:
    def process(self, X):
        pipeline = Pipeline([
            ('hand_extraction', HandExtractionTransformer()),
            ('feature_calculation', FeatureCalculationTransformer())
        ])

        dataset = pipeline.fit_transform(X)

        scalar = StandardScaler()
        dataset.data = scalar.fit_transform(dataset.data)

        return dataset