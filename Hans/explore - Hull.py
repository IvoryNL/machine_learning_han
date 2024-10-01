import os
import glob
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

def getContourExtremes(contour):
    """ Return contour extremes as an tuple of 4 tuples """
    # determine the most extreme points along the contour
    left = contour[contour[:, 0].argmin()]
    right = contour[contour[:, 0].argmax()]
    top = contour[contour[:, 1].argmin()]
    bottom = contour[contour[:, 1].argmax()]

    return np.array((left, right, top, bottom))

def getConvexityDefects(contour):
    """ Return convexity defects in a contour as an nd.array """
    hull = cv.convexHull(contour, returnPoints=False)
    defects = cv.convexityDefects(contour, hull)
    if defects is not None:
        defects = defects.squeeze()

    return defects

def getContourFeatures(contour):
    """ Return some contour features
    """
    # basic contour features
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    extremePoints = getContourExtremes(contour)

    equi_diameter = np.sqrt(4 * area / np.pi)
    defects = getConvexityDefects(contour)

    defects = defects.squeeze()
    defects = defects[defects[:, -1] > 10000]
    total = cv.sumElems(defects[:, -1])[0] #the sum of all the defects
    #print(total)

    x,y,w,h = cv.boundingRect(contour)
    rect_area = w*h
    extent = float(area)/rect_area

    features = np.array((area, perimeter, total, extent))

    return (features)

def getLargestContour(img_BW):
    """ Return largest contour in foreground as an nd.array """
    contours, hier = cv.findContours(img_BW.copy(), cv.RETR_EXTERNAL,
                                     cv.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv.contourArea)

    return np.squeeze(contour)

def maskBlueBG(img):
    # Change image color space
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Define background color range in HSV space
    light_blue = (90, 0, 0)  # converted from HSV value obtained with colorpicker (150,50,0)
    dark_blue = (110, 255, 255)  # converted from HSV value obtained with colorpicker (250,100,100)

    # Mark pixels outside background color range
    mask = ~cv.inRange(img_hsv, light_blue, dark_blue)
    return mask

def fetch_data(data_path):
    # grab the list of images in our data directory
    print("[INFO] loading images...")
    p = os.path.sep.join([data_path, '**', '*.png'])

    file_list = [f for f in glob.iglob(p, recursive=True) if (os.path.isfile(f))]
    print("[INFO] images found: {}".format(len(file_list)))

    # initialize data matrix with correct number of features
    feature_names = ['area', 'perimeter', 'defects', 'extent']
    data = np.empty((0, len(feature_names)), float)
    target = []

    # loop over the image paths
    for filename in file_list:  # [::10]:
        # load image and blur a bit to suppress noise
        img = cv.imread(filename)
        img = cv.blur(img, (3, 3))

        # mask background
        img_BW = maskBlueBG(img)

        # perform a series of erosions and dilations to remove any small regions of noise
        img_BW = cv.erode(img_BW, None, iterations=2)
        img_BW = cv.dilate(img_BW, None, iterations=2)

        # check if foreground is actually there
        if cv.countNonZero(img_BW) == 0:
            continue

        contour = getLargestContour(img_BW)

        # extract features from contour
        features = getContourFeatures(contour)

        # extract label from folder name and stor
        label = filename.split(os.path.sep)[-2]
        target.append(label)

        # append features to data matrix
        data = np.append(data, np.array([features]), axis=0)

        k = cv.waitKey(1) & 0xFF

        # if the `q` key or ESC was pressed, break from the loop
        if k == ord("q") or k == 27:
            break

    unique_targets = np.unique(target)
    print("[INFO] targets found: {}".format(unique_targets))

    dataset = Bunch(data=data,
                    target=target,
                    unique_targets=unique_targets,
                    feature_names=feature_names)

    return dataset

if __name__ == "__main__":
    """feature exploration"""
    data_path = r'C:\Users\roesth\Desktop\Dataset Simplified Hans'

    # fetch the data
    gestures = fetch_data(data_path)

    # encode the categorical labels
    le = LabelEncoder()
    coded_labels = le.fit_transform(gestures.target)

    # partition the data into training and testing splits using 75% of the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(gestures.data, coded_labels,test_size=0.25, stratify=gestures.target, random_state=42)
    trainX = StandardScaler().fit_transform(trainX)
    print(gestures.feature_names)
    print(trainX)

    # show target distribution
    ax = sns.countplot(x=trainY, color="skyblue")
    ax.set_xticklabels(gestures.unique_targets)
    ax.set_title(data_path + ' count')
    plt.tight_layout()

    # show histograms of first 4 features
    fig0, ax0 = plt.subplots(2, 2)
    sns.histplot(trainX[:,0], color="teal", bins=10, ax=ax0[0,0])
    sns.histplot(trainX[:,1], color="olive", bins=10, ax=ax0[0,1])#, axlabel=gestures.feature_names[1])
    sns.histplot(trainX[:,2], color="gold", bins=10, ax=ax0[1,0])#, axlabel=gestures.feature_names[2])
    sns.histplot(trainX[:,3], color="skyblue", bins=10, ax=ax0[1,1])#, axlabel=gestures.feature_names[3])
    ax0[0,0].set_xlabel(gestures.feature_names[0])
    ax0[0,1].set_xlabel(gestures.feature_names[1])
    ax0[1,0].set_xlabel(gestures.feature_names[2])
    ax0[1,1].set_xlabel(gestures.feature_names[3])
    plt.tight_layout()
    
    # show scatter plot of features a and b
    a, b = 1, 2
    fig1 = plt.figure()
    ax1 = sns.scatterplot(x=trainX[:,a], y=trainX[:,b], hue=le.inverse_transform(trainY))
    ax1.set_title("Example of feature scatter plot")
    ax1.set_xlabel(gestures.feature_names[a])
    ax1.set_ylabel(gestures.feature_names[b])
    plt.tight_layout()

    # show boxplot for a single feature
    a = 2
    plt.figure()
    ax3 = sns.boxplot(x=le.inverse_transform(trainY), y=trainX[:,a])
    ax3.set_title(gestures.feature_names[a])
    ax3.set_ylabel(gestures.feature_names[a])
    plt.tight_layout()

    # show feature correlation heatmap
    plt.figure()
    corr = np.corrcoef(trainX, rowvar=False)
    ax4 = sns.heatmap(corr, annot=True, xticklabels=gestures.feature_names, yticklabels=gestures.feature_names)
    plt.tight_layout()
    plt.show(block=True)

