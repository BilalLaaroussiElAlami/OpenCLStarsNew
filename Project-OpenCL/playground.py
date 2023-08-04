from PIL import Image, ImageOps
import numpy as np
import time

x = np.array(
[[ [42,42,92],[ 29,1,69],[ 14,18,30]],
[[ 1,2,3],[4,5,6],[7,8,9]]])

print(x)
print(np.ravel(x))


def convert_to_2d(arr):
    unraveled = np.ravel(arr)
    r_values = unraveled[0:len(unraveled):3]
    g_values = unraveled[1:len(unraveled):3]
    b_values = unraveled[2:len(unraveled):3]
    print(r_values)
    print(g_values)
    print(b_values)

print("hope")
print(convert_to_2d(x))        