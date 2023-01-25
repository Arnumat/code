import cv2
import numpy as np
from fastiecm import fastiecm


def display(image, image_name):
    image = np.array(image, dtype=float)/float(255)
    shape = image.shape
    height = int(shape[0] / 2)
    width = int(shape[1] / 2)
    image = cv2.resize(image, (width, height))
    return image



#set contrast
def contrast_stretch(im):
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 95)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min
    return out

def calc_ndvi(image):
    b, g, r = cv2.split(image)
    
    bottom = (r.astype(float) + b.astype(float))
    bottom[bottom==0] = 0.01
    ndvi = (b.astype(float) - r) / bottom
    return ndvi


original = cv2.VideoCapture(0)
while True:
    _,frame = original.read()
    park = display(frame, 'Original')
    contrasted = contrast_stretch(park)
    ndvi = calc_ndvi(contrasted)
    ndvi_contrasted = contrast_stretch(ndvi)
    color_mapped_prep = ndvi_contrasted.astype(np.uint8)
    color_mapped_image = cv2.applyColorMap(color_mapped_prep, fastiecm)
    cv2.imshow("1",color_mapped_image)
    #print(np.array(color_mapped_image, dtype=float)/float(255))
    if(cv2.waitKey(500)& 0xFF==ord("q")):
        break

original.release()
cv2.destroyAllWindows()
