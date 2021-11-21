import cv2 as cv
import numpy as np
import glob
import pickle


def isFootSize(box): #OK
    width = box[1][0]
    height = box[1][1]
    footSize = width * height
    return isFootProportion(box[1][0], box[1][1]) and between(footSize, 3000, 16000)


def LoadFromFile(name):
    f = open(name + ".dat", "rb")
    var = pickle.load(f)
    f.close()
    return var


def isFootProportion(width, height): #OK
    minFootProportion = 1.4
    maxFootProportion = 7
    footProportionA = width / height
    footProportionB = height / width
    return between(footProportionA, minFootProportion, maxFootProportion) or between(footProportionB, minFootProportion,
                                                                                     maxFootProportion)


def between(value, min, max): #OK
    return value > min and value < max


def reduce(img): #OK
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            k = img[i, j]
            k = round(k / 32)
            img[i, j] = k * 32


class Preprocess:

    def calibrate(self):
        source = np.float32([[1300, 574], [2834, 302], [1114, 2521], [2596, 3136]])
        destination = np.float32([[0, 0], [2400, 0], [0, 2400], [2400, 2400]])
        self.matrix = cv.getPerspectiveTransform(source, destination)

        self.mtx = LoadFromFile("mtx")
        self.dist = LoadFromFile("dist")
        self.newcameramtx = LoadFromFile("newcameramtx")

    def preprocess(self, img):
        # Własciwy preprocess dla wszystkich obrazkow po kolei
        result = cv.undistort(img, self.mtx, self.dist, None, self.newcameramtx)
        return cv.warpPerspective(result, self.matrix, (2400, 2400))


def drawDetectedFeet(img, treshold_val):
    imageGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imageGray = cv.resize(imageGray, (900, 600))
    # imageGray = cv.fastNlMeansDenoising(imageGray)
    # cv.normalize(imageGray, imageGray, 0, 255, cv.NORM_MINMAX)
    # blur = cv.GaussianBlur(imageGray, (5, 5), 0)
    # ret3, thresh = cv.threshold(imageGray, treshold_val, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    k = 0.44
    kernel = [[k, k, k], [k, k, k], [k, k, k]]
    kernel = np.array(kernel)
    imageGray = cv.filter2D(imageGray, -1, kernel)
    imageGray = cv.medianBlur(imageGray, 7)
    #imageGray = cv.GaussianBlur(imageGray,(5,5),0)
    #ret, thresh = cv.threshold(imageGray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    thresh = cv.adaptiveThreshold(imageGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 55, 3)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cv.drawContours(imageGray, contours, -1, (0, 255, 0), 3)
    thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2RGB)


    for c in contours:
        if len(c) < 100:
            continue
        box = cv.fitEllipse(c)
        if box[1][0] > 0 and box[1][1] > 0 and isFootSize(box):
            cv.ellipse(thresh, box, (255, 0, 0))
    return thresh


def notNone(t):
    return t is not None

def notFeetResult(fileName: str):
    return not fileName.endswith("-feet.jpg")




treshold_max = 255
imageFiles = filter(notFeetResult, glob.glob('E:\\PG\\PBGotowe\\zcp\\*.jpg'))
print("Started...")
preprocessing = Preprocess()
print("Calibrating...")
preprocessing.calibrate()

for imageFileName in imageFiles:
    print(f"Processing {imageFileName}")
    image = cv.imread(imageFileName)
    image = preprocessing.preprocess(image)
    detected = drawDetectedFeet(image, 3)
    cv.imwrite(imageFileName.split('.')[0] + '-feet.jpg', detected)

print("Done")
