import cv2
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def find_marker(image):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # lower mask (0-10)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    mask = mask0 + mask1

    pos = np.empty(2)
    for k in range(2):
        nonempty = np.nonzero(np.any(mask, axis=1 - k))[0]
        first, last = nonempty.min(), nonempty.max()
        pos[1 - k] = -(- first - last) // 2

    return pos


f = 16.24
b = 350
p_size = .0055

imgL = cv2.cvtColor(cv.imread('0000_Speckle1_.png'), cv2.COLOR_BGR2RGB)
imgR = cv2.cvtColor(cv.imread('0000_Speckle3_.png'), cv2.COLOR_BGR2RGB)
half_size = np.array(imgL.shape[:2]) / 2

pos_l = find_marker(imgL)
pos_r = find_marker(imgR)

xl, yl = (pos_l - half_size) * p_size
xr, yr = (pos_r - half_size) * p_size
z = f * b / (xl - xr)
#z = f * b / np.mean([yl, yr])
x = xl * z / f
#y = np.mean([yl * z / f, yr * z / f])
y = yl * z / f

plt.clf()
plt.imshow(imgL, alpha=0.5)
plt.imshow(imgR, alpha=0.5)
plt.plot(*pos_l, '*b')
plt.plot(*pos_r, 'xg')
plt.show()

print(np.array([x, y, z],dtype=int))