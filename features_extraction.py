# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 19:07:36 2023

@author: Xuze Ge
"""

import cv2
import skimage
import os
from skimage import io
import numpy as np
from PIL import Image
import scipy.signal as signal


def preprocessing():
    folders = ["aluminium", "copper", "PCB"]
    for name in folders:
        for i in range(1, 7):
            filename = "./datasets_raw/" + name + "/sample" + str(i) + ".png"
            img = Image.open(filename)
            img_region = img.crop((338, 88, 1589, 950))
            path_img_target = os.path.join("./datasets/" + name + "/sample" + str(i) + ".png")
            io.imsave(path_img_target, skimage.img_as_ubyte(img_region))


def contours():
    folders = ["aluminium", "copper", "PCB"]
    for name in folders:
        for i in range(1, 7):
            filename = "./datasets/" + name + "/sample" + str(i) + ".png"
            img = cv2.imread(filename, -1)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            thresh = signal.medfilt2d(thresh, 3)
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            thresh = cv2.erode(thresh, kernel, iterations=1)
            path_img_target = os.path.join("./datasets/binarized/" + name + "/sample" + str(i) + ".png")
            io.imsave(path_img_target, skimage.img_as_ubyte(thresh))

            # extract contours
            contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

            # Select the contour with the largest area
            areas = []
            for c in range(len(contours)) : areas.append(cv2.contourArea(contours[c]))
            max_id = areas.index(max(areas))
            cnt = contours[max_id]

            # draw outline
            draw_img = img.copy()
            res1 = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2)  # -1 means to draw all contours
            draw_img = img.copy()
            res2 = cv2.drawContours(draw_img, [cnt], -1, (255, 0, 0), 3)  # 0 means draw the first contour
            # save the results
            # res = np.hstack((res1, res2))
            path_img_target = os.path.join("./datasets/contours/" + name + "/sample" + str(i) + ".png")
            io.imsave(path_img_target, skimage.img_as_ubyte(res2))


def area_desc(cnt):
    area = cv2.contourArea(cnt)
    return area


def perimeter(cnt):
    return cv2.arcLength(cnt, True)


def Eccentricity(cnt):
    (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
    e = MA / ma
    return e



def homometer(img):
    avg_color = np.average(img, axis=0)
    avg_color = np.average(avg_color, axis=0)
    return avg_color


def lbp(src):
    src = np.resize(src, (32, 32))
    height, width = src.shape[:2]
    dst = np.zeros((height, width), dtype=np.uint8)

    lbp_value = np.zeros((1, 8), dtype=np.uint8)
    neighbours = np.zeros((1, 8), dtype=np.uint8)
    for row in range(1, height-1):
        for col in range(1, width-1):
            center = src[row, col]

            neighbours[0, 0] = src[row-1, col-1]
            neighbours[0, 1] = src[row-1, col]
            neighbours[0, 2] = src[row-1, col+1]
            neighbours[0, 3] = src[row, col+1]
            neighbours[0, 4] = src[row+1, col+1]
            neighbours[0, 5] = src[row+1, col]
            neighbours[0, 6] = src[row+1, col-1]
            neighbours[0, 7] = src[row, col-1]

            for i in range(8):
                if neighbours[0, i] > center:
                    lbp_value[0, i] = 1
                else:
                    lbp_value[0, i] = 0

            # 转成二进制数
            lbp = lbp_value[0, 0] * 1 + lbp_value[0, 1] * 2 + lbp_value[0, 2] * 4 + lbp_value[0, 3] * 8 \
                + lbp_value[0, 4] * 16 + lbp_value[0, 5] * 32 + lbp_value[0, 6] * 64 + lbp_value[0, 7] * 128
            dst[row, col] = lbp
    return dst


def color_moments(filename):
    img = cv2.imread(filename, 1)  # read an image
    if img is None:
        return
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert rgb to hsv
    h, s, v = cv2.split(hsv)
    color_feature = []  # initialize color features array

    # first moment（mean）
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    color_feature.append(h_mean)
    color_feature.append(s_mean)# Put the first moment into the feature array
    color_feature.append(v_mean)
    # second moment （std）
    h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    color_feature.append(h_std)
    color_feature.append(s_std)
    color_feature.append(v_std)# Put the second moment into the feature array

    # third moment （skewness）
    h_skewness = np.mean(abs(h - h.mean()) ** 3)
    s_skewness = np.mean(abs(s - s.mean()) ** 3)
    v_skewness = np.mean(abs(v - v.mean()) ** 3)
    h_thirdMoment = h_skewness ** (1. / 3)
    s_thirdMoment = s_skewness ** (1. / 3)
    v_thirdMoment = v_skewness ** (1. / 3)
    color_feature.append(h_thirdMoment)
    color_feature.append(s_thirdMoment)
    color_feature.append(v_thirdMoment)# Put the third moment into the feature array
    return color_feature


def truncate_fourier(fftCnt, pLowF=32):
    fftShift = np.fft.fftshift(fftCnt)  # Centralization, moving the low frequency components to the center of the frequency domain
    center = int(len(fftShift) / 2)
    low, high = center - int(pLowF / 2), center + int(pLowF / 2)
    fftshiftLow = fftShift[low:high]
    fftLow = np.fft.ifftshift(fftshiftLow)  # Decentralization
    return fftLow


def fourier_descriptors(cnt):
    cntPoints = np.squeeze(cnt)  # Delete array dimension with dimension 1, (2867, 1, 2)->(2867,2)
    # lenCnt = cnt.shape[0]  # number of contour points
    cntComplex = np.empty(cntPoints.shape[0], dtype=complex)
    cntComplex = cntPoints[:, 0] + 1j * cntPoints[:, 1]  # (xk,yk)->xk+j*yk
    fftCnt = np.fft.fft(cntComplex)  # Discrete Fourier transform, generating Fourier descriptors
    fftCnt = truncate_fourier(fftCnt)
    return fftCnt


if __name__ == "__main__":
    preprocessing()
    contours()
    fourier = []
    folders = ["aluminium", "copper", "PCB"]
    features = []

    for name in folders:
        for i in range(1, 7):
            lbp_features = []
            filename = "./datasets/" + name + "/sample" + str(i) + ".png"
            img = cv2.imread(filename, -1)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            thresh = cv2.erode(thresh, kernel, iterations=1)
            
            # extract contours
            contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

            # Select the contour with the largest area
            cnts = sorted(contours, key=cv2.contourArea, reverse=True)  # All contours sorted by area
            cnt = cnts[0]  # The 0th contour, the contour with the largest area
            # Call the above function and store the result in an array
            features.append(area_desc(cnt))
            features.append(perimeter(cnt))
            # features.append(max_length(cnt))
            # features.append(min_length(cnt))
            features.append(Eccentricity(cnt))
            features.append(homometer(thresh))
            features.extend(color_moments(filename))
            features.extend(abs(fourier_descriptors(cnt)))
            features.append('\n')
            # labels.append(name) # Store the label as the last element of the array if desired
            with open('D:\Dissertation\shape_features\\features.txt', 'w') as f:
                for data in features:
                    f.write(str(data) + " ")


            #print(features)
            #features.append(lbp(img_gray)) 
            #features.append(name)
            # features.append(fourier_desc(thresh, thresh.size))
    #print(features)
    #np.savetxt("./features.txt", features, fmt='%s', delimiter=', ')

            # lbp_features.append(lbp(img_gray))
            #lbp_features.append(lbp(img_gray))
            #lbp_features.append(name)
            #np.savetxt("./shape/" + name + '/' + name + str(i) + ".txt", features, fmt='%s', delimiter=',')