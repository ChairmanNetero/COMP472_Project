import cv2
import os

input_dir = 'Dataset/test/surprise/'
output_dir = 'out/test/surprise/'

# list files in input directory
list = os.listdir(input_dir)

for x in list:
    # read image
    i = cv2.imread(os.path.join(input_dir, x))

    # resize image
    ri = cv2.resize(i, (192, 192))

    # turn image black and white
    # gi = cv2.cvtColor(ri, cv2.COLOR_BGR2GRAY)

    # output the image
    o = os.path.join(output_dir, x)
    cv2.imwrite(o, ri)
