import cv2
import numpy as np
from skimage.filters import threshold_otsu

def get_mask(image):
    
    image = 255 - image
    kernel = np.ones((1,1),np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    kernel2 = np.ones((1,1), np.uint8)
    kernel2 = kernel2/np.sum(kernel2)
    image_lp = cv2.filter2D(image,-1,kernel2)

    thresh = threshold_otsu(image_lp)
    mask = image_lp > thresh
    mask = (mask+0)*255
    
    return mask
