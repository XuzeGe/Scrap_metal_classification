# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 17:05:58 2023

@author: unreval
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def fourier_descriptor(img_path):
    img = cv2.imread(img_path, flags=1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # gray image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # Find the contour in the binary image, method=cv2.CHAIN_APPROX_NONE outputs each pixel of the contour
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # OpenCV4~
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)  # All contours sorted by area
    cnt = cnts[0]  # The 0th contour, the contour with the largest area
    cntPoints = np.squeeze(cnt)  # Delete array dimension with dimension 1, (2867, 1, 2)->(2867,2)
    lenCnt = cnt.shape[0]  # number of contour points
    # print("length of max contour:", lenCnt)
    # Discrete Fourier transform, generate Fourier descriptor fftCnt
    cntComplex = np.empty(cntPoints.shape[0], dtype=complex)  # declare complex array (2867,)
    cntComplex = cntPoints[:, 0] + 1j * cntPoints[:, 1]  # (xk,yk)->xk+j*yk
    # print("cntComplex", cntComplex.shape)
    fftCnt = np.fft.fft(cntComplex)
    scale = cntPoints.max()
    return fftCnt, scale


def truncFFT(fftCnt, pLowF=64):  # Truncated Fourier Descriptor
    fftShift = np.fft.fftshift(fftCnt)  # Centering, moving the low frequency components to the center of the frequency domain
    center = int(len(fftShift) / 2)
    low, high = center - int(pLowF / 2), center + int(pLowF / 2)
    fftshiftLow = fftShift[low:high]
    fftLow = np.fft.ifftshift(fftshiftLow)  # Decentralization
    return fftLow


def reconstruct(fftCnt, scale, ratio=1.0):  # Contour reconstruction from Fourier descriptors
    pLowF = int(fftCnt.shape[0] * ratio)  # Truncated length P<=K
    fftLow = truncFFT(fftCnt, pLowF)  # Truncate the Fourier descriptor to remove high-frequency coefficients
    ifft = np.fft.ifft(fftLow)  # Inverse Fourier Transform (P,)
    # cntRebuild = np.array([ifft.real, ifft.imag])  # complex number to array (2, P)
    # cntRebuild = np.transpose(cntRebuild)  # (P, 2)
    cntRebuild = np.stack((ifft.real, ifft.imag), axis=-1)  # complex number to array (P, 2)
    if cntRebuild.min() < 0:
        cntRebuild -= cntRebuild.min()
    cntRebuild *= scale / cntRebuild.max()
    cntRebuild = cntRebuild.astype(np.int32)
    print("ratio={}, fftCNT:{}, fftLow:{}".format(ratio, fftCnt.shape, fftLow.shape))

    # rebuild = np.ones(img.shape, np.uint8) * 255  # create blank image
    # cv2.polylines(rebuild, [cntRebuild], True, 0, thickness=2)  # draw polygons, closed curves
    return cntRebuild


if __name__ == "__main__":
    path = "D:\\Dissertation\\ML_algorithm\\data_f\\"
    image_dir = sorted(os.listdir(path))
    factor = np.zeros((18, 12))
    color = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
    rate = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01]
    rate_reverse = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    loss_mean = np.zeros(12)
    for i, image in enumerate(image_dir):
        img_path = os.path.join(path, image)
        # img = cv2.imread(img_path, flags=1)
        fftcnt, scale = fourier_descriptor(img_path)
        cntRebuild = reconstruct(fftcnt, scale)
        perimeter = cv2.arcLength(cntRebuild, True)
        j = 0
        for r in rate:
            cntRebuild_1 = reconstruct(fftcnt, scale, ratio=r)
            perimeter_1 = cv2.arcLength(cntRebuild_1, True)
            f = 1 - perimeter_1 / perimeter
            factor[i, j] = f
            # plt.scatter(r, f)
            j += 1
        #plt.plot(rate_reverse, factor[i, :])
    # plot the results
    # plt.scatter([0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01], factor)
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
        loss_mean[i] = np.mean(factor[:, i])
    plt.plot(rate_reverse, loss_mean)
    plt.xlabel('Truncated rate')
    plt.ylabel('average loss ratio')
    plt.title('')
    plt.grid()
    plt.show()

            
            
        
    
    


