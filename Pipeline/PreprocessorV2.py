import numpy as np
import cv2 as cv
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

        # Apply sharpening filter
        sharpen_kernel = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]], dtype=np.float32)
        sharpened = cv.filter2D(image, -1, sharpen_kernel)

        # Apply Gaussian Blur
        blurred = cv.GaussianBlur(sharpened, (11, 11), 0)

        # Detect edges
        edges = cv.Canny(blurred, 50, 125)

        # Dilate edges
        dilated_edges = cv.dilate(edges, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)), iterations=2)

        # Find contours
        contours, _ = cv.findContours(dilated_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv.contourArea)

        # Draw largest contour
        contour_image = np.zeros_like(image, dtype=np.uint8)
        cv.drawContours(contour_image, [largest_contour], -1, 255, cv.FILLED)

        # Determine minY, maxY, minX, maxX
        image_height = image.shape[0]
        min_y = min(point[0][1] for point in largest_contour)
        max_y = max(point[0][1] for point in largest_contour)
        distance_to_bottom = image_height - max_y
        horizontal_edge_y = 0 if min_y < distance_to_bottom else image_height

        min_x = min(point[0][0] for point in largest_contour)
        max_x = max(point[0][0] for point in largest_contour)
        center_of_contours_width = min_x + ((max_x - min_x) // 2)

        # Find closest points to left and right edges
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

        # Draw line between closest points
        cv.line(contour_image, closest_point_left_edge, closest_point_right_edge, 255, 5)

        # Find contours again on the updated contour image
        contours, _ = cv.findContours(contour_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv.contourArea)

        # Draw the largest contour in the final result image
        result = np.zeros_like(image, dtype=np.uint8)
        cv.drawContours(result, [largest_contour], -1, 255, cv.FILLED)

        return result

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
        try:
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
            ('feature_calculation', FeatureCalculationTransformer())
        ])

        dataset = pipeline.fit_transform(X)

        return dataset