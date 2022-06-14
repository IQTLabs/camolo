'''
Create circular mask
https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array

Usage:
    python create_circle_mask.py size outpath
'''

import numpy as np
import skimage.io
import sys

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask
    

if __name__ == '__main__':
    size = sys.argv[1]
    outpath = sys.argv[2]
    
    mask_bool = create_circular_mask(int(size), int(size))
    mask = mask_bool.astype(int)
    print("mask:", mask)
    skimage.io.imsave(outpath, mask)
