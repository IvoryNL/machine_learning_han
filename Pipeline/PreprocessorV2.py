import cv2 as cv
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
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
        if image is None:
            return

        sharpen_kernel = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]], dtype=np.float32)
        sharpened = cv.filter2D(image, -1, sharpen_kernel)

        blurred = cv.GaussianBlur(sharpened, (11, 11), 0)

        edges = cv.Canny(blurred, 50, 125)

        dilated_edges = cv.dilate(edges, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)), iterations=2)

        contours = cv.findContours(dilated_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv.contourArea)

        contour_image = np.zeros_like(image, dtype=np.uint8)
        cv.drawContours(contour_image, [largest_contour], -1, 255, cv.FILLED)

        image_height = image.shape[0]
        min_y = min(point[0][1] for point in largest_contour)
        max_y = max(point[0][1] for point in largest_contour)
        distance_to_bottom = image_height - max_y
        horizontal_edge_y = 0 if min_y < distance_to_bottom else image_height

        min_x = float('inf')
        max_x = float('-inf')

        for point in largest_contour:
            x, y = point[0]
            if (horizontal_edge_y == 0 and y <= 50) or (y >= image_height - 50):  # Applying 50-pixel offset
                min_x = min(min_x, x)
                max_x = max(max_x, x)

        center_of_contours_width = min_x + ((max_x - min_x) // 2)

        closest_point_left_edge = (-1, np.inf if horizontal_edge_y == 0 else -np.inf)
        closest_point_right_edge = (-1, np.inf if horizontal_edge_y == 0 else -np.inf)

        for point in largest_contour:
            x, y = point[0]
            if x < center_of_contours_width:
                if (horizontal_edge_y == 0 and y < closest_point_left_edge[1]) or (
                        horizontal_edge_y == image_height and y > closest_point_left_edge[1]):
                    closest_point_left_edge = (x, y)
            else:
                if (horizontal_edge_y == 0 and y < closest_point_right_edge[1]) or (
                        horizontal_edge_y == image_height and y > closest_point_right_edge[1]):
                    closest_point_right_edge = (x, y)

        cv.line(contour_image, closest_point_left_edge, closest_point_right_edge, 255, 5)

        contours, _ = cv.findContours(contour_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv.contourArea)

        result = np.zeros_like(image, dtype=np.uint8)
        cv.drawContours(result, [largest_contour], -1, 255, cv.FILLED)

        return result

class FeatureCalculationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
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

    def getConvexityDefects(self, contour):
        hull = cv.convexHull(contour, returnPoints=False)
        defects = cv.convexityDefects(contour, hull)
        if defects is not None:
            defects = defects.squeeze()

        return defects

    def getConvexHullAreas(contour):
        hull = cv.convexHull(contour)

        return cv.contourArea(hull)

    def getContourFeatures(self, contour):
        try:
            defects = self.getConvexityDefects(contour)
            defects = defects.squeeze()
            defects = defects[defects[:, -1] > 10000]
            totalDefects = cv.sumElems(defects[:, -1])[0]

            perimeter = cv.arcLength(contour, True)

            hullArea = cv.contourArea(contour)

            features = np.array((totalDefects, perimeter, hullArea))

            return features
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def getLargestContour(self, img_BW):
        contours, hier = cv.findContours(img_BW.copy(), cv.RETR_EXTERNAL,
                                         cv.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv.contourArea)

        return np.squeeze(contour)

class PreprocessingPipeline:
    def process(self, X):
        pipeline = Pipeline([
            ('hand_extraction', HandExtractionTransformer()),
            ('feature_calculation', FeatureCalculationTransformer()),

        ])

        dataset = pipeline.fit_transform(X)

        return dataset