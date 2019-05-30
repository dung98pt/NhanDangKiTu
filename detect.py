import Letter
import cv2
import numpy as np
from matplotlib import pyplot as plt


def findCorners(bound):
    c1 = [bound[3][0], bound[0][1]]
    c2 = [bound[1][0], bound[0][1]]
    c3 = [bound[1][0], bound[2][1]]
    c4 = [bound[3][0], bound[2][1]]
    return [c1, c2, c3, c4]


def dist(P1, P2):
    return np.sqrt((P1[0] - P2[0]) ** 2 + (P1[1] - P2[1]) ** 2)


def mergeBoxes(c1, c2):
    newRect = []
    # find new corner for the top left
    cx = min(c1[0][0], c2[0][0])
    cy = min(c1[0][1], c2[0][1])
    newRect.append([cx, cy])
    # find new corner for the top right
    cx = max(c1[1][0], c2[1][0])
    cy = min(c1[1][1], c2[1][1])
    newRect.append([cx, cy])
    # find new corner for bottm right
    cx = max(c1[2][0], c2[2][0])
    cy = max(c1[2][1], c2[2][1])
    newRect.append([cx, cy])
    # find new corner for bottm left
    cx = min(c1[3][0], c2[3][0])
    cy = max(c1[3][1], c2[3][1])
    newRect.append([cx, cy])
    return newRect


def findCenterCoor(c1):
    width = abs(c1[0][0] - c1[1][0])
    height = abs(c1[0][1] - c1[3][1])
    return ([c1[0][0] + (width / 2.0), c1[0][1] + (height / 2.0)])


def findArea(c1):
    return abs(c1[0][0] - c1[1][0]) * abs(c1[0][1] - c1[3][1])
class Toa_Do:
    def __init__(self):
        pass

    def out_put(img):
        bndingBx = []  # holds bounding box of each countour
        corners = []
        
        #img = cv2.imread('1.png', 0)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        th3 = cv2.bitwise_not(th3)
        contours, heirar = cv2.findContours(th3, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        print('-------------------------------------')
        # find the rectangle around each contour
        for num in range(0, len(contours)):
            if (heirar[0][num][3] == -1):
                left = tuple(contours[num][contours[num][:, :, 0].argmin()][0])
                right = tuple(contours[num][contours[num][:, :, 0].argmax()][0])
                top = tuple(contours[num][contours[num][:, :, 1].argmin()][0])
                bottom = tuple(contours[num][contours[num][:, :, 1].argmax()][0])
                bndingBx.append([top, right, bottom, left])

        # find the edges of each bounding box
        for bx in bndingBx:
            corners.append(findCorners(bx))
        imgplot = plt.imshow(img, 'gray')
        plt.clf()
        err = 2  # error value for minor/major axis ratio
        # list will hold the areas of each bounding boxes
        Area = []
        # go through each corner and append its area to the list
        for corner in corners:
            Area.append(findArea(corner))
        Area = np.asarray(Area)  # organize list into array format
        avgArea = np.mean(Area)  # find average area
        stdArea = np.std(Area)  # find standard deviation of area
        outlier = (Area < avgArea - stdArea)  # find the out liers, these are probably the dots
        for num in range(0, len(outlier)):  # go through each outlier
            dot = False
            if (outlier[num]):
                black = np.zeros((len(img), len(img[0])), np.uint8)
                cv2.rectangle(black, (corners[num][0][0], corners[num][0][1]), (corners[num][2][0], corners[num][2][1]),
                              (255, 255), -1)
                fin = cv2.bitwise_and(th3, black)
                tempCnt, tempH = cv2.findContours(fin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                # loop, due to structure of countours
                for cnt in tempCnt:
                    rect = cv2.minAreaRect(cnt)
                    axis1 = rect[1][0] / 2.0
                    axis2 = rect[1][1] / 2.0
                    if (axis1 != 0 and axis2 != 0):  # do not perform if image has 0 dimension
                        ratio = axis1 / axis2  # calculate ratio of axis
                        # if ratio is close to 1 (circular), then most likely a dot
                        if ratio > 1.0 - err and ratio < err + 1.0:
                            dot = True
                if dot:
                    bestCorner = corners[num]
                    closest = np.inf
                    for crn in corners:
                        width = abs(crn[0][0] - crn[1][0])
                        height = abs(crn[0][1] - crn[3][1])
                        if (corners[num][0][1] > crn[0][1]):
                            continue
                        elif dist(corners[num][0], crn[0]) < closest and crn != corners[num]:
                            cent = findCenterCoor(crn)
                            bestCorner = crn
                            closest = dist(corners[num][0], crn[0])
                    # print(bestCorner)
                    newCorners = mergeBoxes(corners[num], bestCorner)
                    corners.append(newCorners)
                    # print(newCorners)
                    corners[num][0][0] = 0
                    corners[num][0][1] = 0
                    corners[num][1][0] = 0
                    corners[num][1][1] = 0
                    corners[num][2][0] = 0
                    corners[num][2][1] = 0
                    corners[num][3][0] = 0
                    corners[num][3][1] = 0
                    bestCorner[0][0] = 0
                    bestCorner[0][1] = 0
                    bestCorner[1][0] = 0
                    bestCorner[1][1] = 0
                    bestCorner[2][0] = 0
                    bestCorner[2][1] = 0
                    bestCorner[3][0] = 0
                    bestCorner[3][1] = 0

        ###############################################
        # Take letters and turn them into objects
        AllLetters = []
        ahihi = []
        counter = 0
        for bx in corners:
            width = abs(bx[1][0] - bx[0][0])
            height = abs(bx[3][1] - bx[0][1])
            if width*height == 0:
                continue
            #plt.plot([bx[0][0],bx[1][0]],[bx[0][1],bx[1][1]],'b-',linewidth=2)
            #plt.plot([bx[1][0],bx[2][0]],[bx[1][1],bx[2][1]],'b-',linewidth=2)
            #plt.plot([bx[2][0],bx[3][0]],[bx[2][1],bx[3][1]],'b-',linewidth=2)
            #plt.plot([bx[3][0],bx[0][0]],[bx[3][1],bx[0][1]],'b-',linewidth=2)
            #newLetter = Letter.Letter([bx[0][0],bx[0][1]],[height,width],counter)
            #AllLetters.append(newLetter)
            #counter+=1
            if width * height != 0:
                ahihi.append([bx[0][1], bx[2][1], bx[0][0], bx[2][0]])
            # print(bx)
        #plt.imshow(th3,'gray')
        #plt.show()
        #plt.clf()
        return th3, ahihi

