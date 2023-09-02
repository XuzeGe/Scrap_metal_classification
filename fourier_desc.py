# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 19:07:36 2023

@author: Xuze Ge
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def truncFFT(fftCnt, pLowF=64):  # Truncated Fourier Descriptor
    fftShift = np.fft.fftshift(fftCnt)  # Centering, moving the low frequency components to the center of the frequency domain
    center = int(len(fftShift) / 2)
    low, high = center - int(pLowF / 2), center + int(pLowF / 2)
    fftshiftLow = fftShift[low:high]
    fftLow = np.fft.ifftshift(fftshiftLow)  # Decentralization
    return fftLow


def reconstruct(img, fftCnt, scale, ratio=1.0):  # Contour reconstruction from Fourier descriptors
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

    rebuild = np.ones(img.shape, np.uint8) * 255  # create blank image
    cv2.polylines(rebuild, [cntRebuild], True, 0, thickness=2)  # draw polygons, closed curves
    return rebuild


img = cv2.imread("./datasets/aluminium/sample1.png", flags=1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # gray image
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
# print(gray.shape)  # (727, 570)

# Find the contour in the binary image, method=cv2.CHAIN_APPROX_NONE outputs each pixel of the contour
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # OpenCV4~
cnts = sorted(contours, key=cv2.contourArea, reverse=True)  # All contours sorted by area
cnt = cnts[0]  # The 0th contour, the contour with the largest area
cntPoints = np.squeeze(cnt)  # Delete array dimension with dimension 1, (2867, 1, 2)->(2867,2)
lenCnt = cnt.shape[0]  # number of contour points
print("length of max contour:", lenCnt)
imgCnts = np.zeros(gray.shape[:2], np.uint8)  # create blank image
cv2.drawContours(imgCnts, cnt, -1, (255, 255, 255), 2)  # draw contours

# Discrete Fourier transform, generate Fourier descriptor fftCnt
cntComplex = np.empty(cntPoints.shape[0], dtype=complex)  # declare complex array (2867,)
cntComplex = cntPoints[:, 0] + 1j * cntPoints[:, 1]  # (xk,yk)->xk+j*yk
# print("cntComplex", cntComplex.shape)
fftCnt = np.fft.fft(cntComplex)  # Discrete Fourier transform, generating Fourier descriptors
# fftCnt = truncFFT(fftCnt)
# print(fftCnt)
# Contour reconstruction from all Fourier descriptors
scale = cntPoints.max()  # Scale factor
rebuild = reconstruct(img, fftCnt, scale)  # Inverse Fourier transform to reconstruct the contour curve, Fourier descriptor (2866,)
# Reconstruction of contour curves from truncated Fourier coefficients
rebuild1 = reconstruct(img, fftCnt, scale, ratio=0.5)  # Truncation ratio 20%
rebuild2 = reconstruct(img, fftCnt, scale, ratio=0.4)  # Truncation ratio 5%
rebuild3 = reconstruct(img, fftCnt, scale, ratio=0.3)  # Truncation ratio 2%
rebuild4 = reconstruct(img, fftCnt, scale, ratio=0.2)  # Truncation ratio 1%
rebuild5 = reconstruct(img, fftCnt, scale, ratio=0.1)
rebuild6 = reconstruct(img, fftCnt, scale, ratio=0.05)
# Plot the results, assess the quality of Fourier descriptors based on the results of reconstructed images
plt.figure(figsize=(9, 6))
plt.subplot(331), plt.axis('off'), plt.title("Origin")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(332), plt.axis('off'), plt.title("Contour")
plt.imshow(cv2.cvtColor(imgCnts, cv2.COLOR_BGR2RGB))
plt.subplot(333), plt.axis('off'), plt.title("rebuild (100%)")
plt.imshow(cv2.cvtColor(rebuild, cv2.COLOR_BGR2RGB))
plt.subplot(334), plt.axis('off'), plt.title("rebuild1 (50%)")
plt.imshow(cv2.cvtColor(rebuild1, cv2.COLOR_BGR2RGB))
plt.subplot(335), plt.axis('off'), plt.title("rebuild2 (40%)")
plt.imshow(cv2.cvtColor(rebuild2, cv2.COLOR_BGR2RGB))
plt.subplot(336), plt.axis('off'), plt.title("rebuild3 (30%)")
plt.imshow(cv2.cvtColor(rebuild3, cv2.COLOR_BGR2RGB))
plt.subplot(337), plt.axis('off'), plt.title("rebuild4 (20%)")
plt.imshow(cv2.cvtColor(rebuild4, cv2.COLOR_BGR2RGB))
plt.subplot(338), plt.axis('off'), plt.title("rebuild5 (10%)")
plt.imshow(cv2.cvtColor(rebuild5, cv2.COLOR_BGR2RGB))
plt.subplot(339), plt.axis('off'), plt.title("rebuild4 (5%)")
plt.imshow(cv2.cvtColor(rebuild6, cv2.COLOR_BGR2RGB))
plt.tight_layout()
plt.show()
