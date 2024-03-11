import cv2
import os

dir = 'Dataset/test/surprise/'
list = os.listdir(dir)

for x in list:
    i = cv2.imread(os.path.join(dir, x))
    ri = cv2.resize(i, (192, 192))
    # gi = cv2.cvtColor(ri, cv2.COLOR_BGR2GRAY)

    o = os.path.join('out/test/surprise/', x)
    cv2.imwrite(o, ri)
