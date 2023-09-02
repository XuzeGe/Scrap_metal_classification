import sys
sys.path.append('/home/aistudio/external-libraries')
from skimage.feature import local_binary_pattern
import numpy as np
import matplotlib.pyplot as plt
import cv2


# The value of range radius in LBP algorithm
radius = 1
# Area pixel count
n_points = 8 * radius
image = cv2.imread('D:\Dissertation\shape_features\datasets\\PCB\sample1.png')
image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
lbp = local_binary_pattern(image, n_points, radius)
# Get the largest number in lbp
max_bins = int(lbp.max() + 1)
# test_hist is the number of a certain gray level, that is, the y coordinate. x is the abscissa.
test_hist, x = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
a = sum(test_hist)
plt.subplot(111)
plt.imshow(lbp, plt.cm.gray)
'''plt.subplot(111)
plt.plot(range(0, 256), test_hist)
plt.xlabel('Grey level')
plt.ylabel('Frequency')
plt.title('LBP feature spectrum')'''
plt.show()
